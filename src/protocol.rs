//! Wire protocol for the pipeline-parallel link.
//!
//! Every frame starts with a fixed 12-byte header so a receiver can route and
//! validate before touching the payload:
//!
//! ```text
//! offset  size  field
//!   0      1    version      (PROTOCOL_VERSION)
//!   1      1    opcode
//!   2      1    codec        (payload encoding for tensor frames)
//!   3      1    flags        (bit 0 = FLAG_FINAL)
//!   4      4    request_id   u32 LE
//!   8      4    seq_pos      u32 LE
//! ```
//!
//! `request_id` is the important one: it lets the coordinator match a reply to
//! the step that asked for it, so a late or duplicated frame can be dropped
//! instead of resolving the wrong pending promise.

/// Bump whenever a frame layout changes. The version byte is checked before any
/// payload is read, so a mismatched peer is reported as a version skew rather
/// than as a confusing length error.
pub const PROTOCOL_VERSION: u8 = 3;
pub const HEADER_LEN: usize = 12;

pub const OP_HELLO: u8 = 0x00;
pub const OP_ACTIVATION: u8 = 0x01;
pub const OP_TOKEN: u8 = 0x02;
pub const OP_RESET: u8 = 0x03;
pub const OP_ERROR: u8 = 0x04;

/// Marks the last frame of a generation.
pub const FLAG_FINAL: u8 = 0b0000_0001;

/// How a float payload is packed on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// Raw little-endian `f32`. 4 bytes per element.
    F32 = 0,
    /// Symmetric per-tensor int8 with an `f32` scale. ~4x smaller, slightly lossy.
    Q8 = 1,
}

impl Codec {
    pub fn from_u8(v: u8) -> Result<Self, ProtocolError> {
        match v {
            0 => Ok(Codec::F32),
            1 => Ok(Codec::Q8),
            other => Err(ProtocolError::UnknownCodec(other)),
        }
    }

    /// Encoded byte count for `n` floats, excluding the frame header.
    pub fn payload_len(self, n: usize) -> usize {
        match self {
            // count + data
            Codec::F32 => 4 + n * 4,
            // count + scale + data
            Codec::Q8 => 4 + 4 + n,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    Truncated { need: usize, got: usize },
    VersionMismatch { expected: u8, got: u8 },
    UnexpectedOpcode { expected: u8, got: u8 },
    UnknownCodec(u8),
    LengthMismatch { expected: usize, got: usize },
    Utf8,
}

impl core::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProtocolError::Truncated { need, got } => {
                write!(f, "frame truncated: need at least {need} bytes, got {got}")
            }
            ProtocolError::VersionMismatch { expected, got } => write!(
                f,
                "protocol version mismatch: peer speaks v{got}, this build speaks v{expected}"
            ),
            ProtocolError::UnexpectedOpcode { expected, got } => write!(
                f,
                "unexpected opcode 0x{got:02x} (expected 0x{expected:02x})"
            ),
            ProtocolError::UnknownCodec(c) => write!(f, "unknown codec id {c}"),
            ProtocolError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "payload length mismatch: expected {expected} bytes, got {got}"
                )
            }
            ProtocolError::Utf8 => write!(f, "payload is not valid utf-8"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub opcode: u8,
    pub codec: Codec,
    pub flags: u8,
    pub request_id: u32,
    pub seq_pos: u32,
}

impl Header {
    pub fn new(opcode: u8, request_id: u32, seq_pos: u32) -> Self {
        Header {
            version: PROTOCOL_VERSION,
            opcode,
            codec: Codec::F32,
            flags: 0,
            request_id,
            seq_pos,
        }
    }

    pub fn with_codec(mut self, codec: Codec) -> Self {
        self.codec = codec;
        self
    }

    pub fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    pub fn is_final(&self) -> bool {
        self.flags & FLAG_FINAL != 0
    }

    fn write_into(&self, out: &mut Vec<u8>) {
        out.push(self.version);
        out.push(self.opcode);
        out.push(self.codec as u8);
        out.push(self.flags);
        out.extend_from_slice(&self.request_id.to_le_bytes());
        out.extend_from_slice(&self.seq_pos.to_le_bytes());
    }

    /// Parse and validate a frame header without consuming the payload.
    pub fn parse(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.len() < HEADER_LEN {
            return Err(ProtocolError::Truncated {
                need: HEADER_LEN,
                got: bytes.len(),
            });
        }
        if bytes[0] != PROTOCOL_VERSION {
            return Err(ProtocolError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                got: bytes[0],
            });
        }
        Ok(Header {
            version: bytes[0],
            opcode: bytes[1],
            codec: Codec::from_u8(bytes[2])?,
            flags: bytes[3],
            request_id: read_u32(bytes, 4),
            seq_pos: read_u32(bytes, 8),
        })
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn expect_opcode(header: &Header, expected: u8) -> Result<(), ProtocolError> {
    if header.opcode != expected {
        return Err(ProtocolError::UnexpectedOpcode {
            expected,
            got: header.opcode,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor payload codecs
// ---------------------------------------------------------------------------

/// Symmetric per-tensor int8 quantisation. Returns the scale that maps the
/// int8 grid back to floats.
fn quantize_q8(values: &[f32], out: &mut Vec<u8>) -> f32 {
    let max_abs = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // An all-zero tensor has no scale; use 1.0 so dequantisation stays a no-op.
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv = scale.recip();
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in values {
        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    scale
}

fn dequantize_q8(bytes: &[u8], count: usize) -> Result<Vec<f32>, ProtocolError> {
    if bytes.len() < 4 + count {
        return Err(ProtocolError::Truncated {
            need: 4 + count,
            got: bytes.len(),
        });
    }
    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    Ok(bytes[4..4 + count]
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect())
}

// ---------------------------------------------------------------------------
// Frame builders / parsers
// ---------------------------------------------------------------------------

/// Capability handshake. Sent once when the data channel opens so two peers
/// running mismatched builds fail with a clear message instead of garbage.
///
/// `backend` is informational only -- it lets each side show what hardware its
/// peer is actually running on, and never participates in the match check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    pub dim: u32,
    pub n_layers: u32,
    pub vocab_size: u32,
    pub max_seq: u32,
    pub backend: String,
}

impl Hello {
    /// The fields that must agree for two peers to be able to talk.
    pub fn shape(&self) -> (u32, u32, u32, u32) {
        (self.dim, self.n_layers, self.vocab_size, self.max_seq)
    }
}

const HELLO_FIXED: usize = 20;

/// `request_id` distinguishes an announcement (`HELLO_ANNOUNCE`) from the answer
/// to one (`HELLO_REPLY`), so the exchange terminates rather than ping-ponging.
pub const HELLO_ANNOUNCE: u32 = 0;
pub const HELLO_REPLY: u32 = 1;

pub fn encode_hello(hello: &Hello, request_id: u32) -> Vec<u8> {
    let label = hello.backend.as_bytes();
    let mut out = Vec::with_capacity(HEADER_LEN + HELLO_FIXED + label.len());
    Header::new(OP_HELLO, request_id, 0).write_into(&mut out);
    out.extend_from_slice(&hello.dim.to_le_bytes());
    out.extend_from_slice(&hello.n_layers.to_le_bytes());
    out.extend_from_slice(&hello.vocab_size.to_le_bytes());
    out.extend_from_slice(&hello.max_seq.to_le_bytes());
    out.extend_from_slice(&(label.len() as u32).to_le_bytes());
    out.extend_from_slice(label);
    out
}

pub fn decode_hello(bytes: &[u8]) -> Result<(Header, Hello), ProtocolError> {
    let header = Header::parse(bytes)?;
    expect_opcode(&header, OP_HELLO)?;
    if bytes.len() < HEADER_LEN + HELLO_FIXED {
        return Err(ProtocolError::Truncated {
            need: HEADER_LEN + HELLO_FIXED,
            got: bytes.len(),
        });
    }

    let label_len = read_u32(bytes, HEADER_LEN + 16) as usize;
    let label_start = HEADER_LEN + HELLO_FIXED;
    if bytes.len() < label_start + label_len {
        return Err(ProtocolError::Truncated {
            need: label_start + label_len,
            got: bytes.len(),
        });
    }
    let backend = core::str::from_utf8(&bytes[label_start..label_start + label_len])
        .map_err(|_| ProtocolError::Utf8)?
        .to_string();

    Ok((
        header,
        Hello {
            dim: read_u32(bytes, HEADER_LEN),
            n_layers: read_u32(bytes, HEADER_LEN + 4),
            vocab_size: read_u32(bytes, HEADER_LEN + 8),
            max_seq: read_u32(bytes, HEADER_LEN + 12),
            backend,
        },
    ))
}

/// Hidden state travelling from stage N to stage N+1.
pub fn encode_activation(header: Header, values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + header.codec.payload_len(values.len()));
    let header = Header {
        opcode: OP_ACTIVATION,
        ..header
    };
    header.write_into(&mut out);
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    match header.codec {
        Codec::F32 => {
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Codec::Q8 => {
            quantize_q8(values, &mut out);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Activation {
    pub header: Header,
    pub values: Vec<f32>,
}

pub fn decode_activation(bytes: &[u8]) -> Result<Activation, ProtocolError> {
    let header = Header::parse(bytes)?;
    expect_opcode(&header, OP_ACTIVATION)?;
    if bytes.len() < HEADER_LEN + 4 {
        return Err(ProtocolError::Truncated {
            need: HEADER_LEN + 4,
            got: bytes.len(),
        });
    }
    let count = read_u32(bytes, HEADER_LEN) as usize;
    let expected = HEADER_LEN + header.codec.payload_len(count);
    if bytes.len() != expected {
        return Err(ProtocolError::LengthMismatch {
            expected,
            got: bytes.len(),
        });
    }

    let body = &bytes[HEADER_LEN + 4..];
    let values = match header.codec {
        Codec::F32 => body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Codec::Q8 => dequantize_q8(body, count)?,
    };

    Ok(Activation { header, values })
}

/// The sampled token flowing back to the coordinator, plus the worker's own
/// compute time so the coordinator can show a real per-stage breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenReply {
    pub token_id: u32,
    pub compute_us: u32,
}

pub fn encode_token(header: Header, reply: &TokenReply) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 8);
    let header = Header {
        opcode: OP_TOKEN,
        ..header
    };
    header.write_into(&mut out);
    out.extend_from_slice(&reply.token_id.to_le_bytes());
    out.extend_from_slice(&reply.compute_us.to_le_bytes());
    out
}

pub fn decode_token(bytes: &[u8]) -> Result<(Header, TokenReply), ProtocolError> {
    let header = Header::parse(bytes)?;
    expect_opcode(&header, OP_TOKEN)?;
    if bytes.len() < HEADER_LEN + 8 {
        return Err(ProtocolError::Truncated {
            need: HEADER_LEN + 8,
            got: bytes.len(),
        });
    }
    Ok((
        header,
        TokenReply {
            token_id: read_u32(bytes, HEADER_LEN),
            compute_us: read_u32(bytes, HEADER_LEN + 4),
        },
    ))
}

/// Tells the peer to drop its KV cache and start a fresh sequence.
pub fn encode_reset(request_id: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    Header::new(OP_RESET, request_id, 0).write_into(&mut out);
    out
}

/// Propagates a worker-side failure back to the coordinator instead of letting
/// the request time out.
pub fn encode_error(request_id: u32, seq_pos: u32, message: &str) -> Vec<u8> {
    let bytes = message.as_bytes();
    let mut out = Vec::with_capacity(HEADER_LEN + bytes.len());
    Header::new(OP_ERROR, request_id, seq_pos).write_into(&mut out);
    out.extend_from_slice(bytes);
    out
}

pub fn decode_error(bytes: &[u8]) -> Result<(Header, String), ProtocolError> {
    let header = Header::parse(bytes)?;
    expect_opcode(&header, OP_ERROR)?;
    let text = core::str::from_utf8(&bytes[HEADER_LEN..]).map_err(|_| ProtocolError::Utf8)?;
    Ok((header, text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect()
    }

    #[test]
    fn f32_activation_round_trips_exactly() {
        let values = ramp(128);
        let header = Header::new(OP_ACTIVATION, 42, 7).with_codec(Codec::F32);
        let frame = encode_activation(header, &values);

        assert_eq!(frame.len(), HEADER_LEN + 4 + 128 * 4);
        let decoded = decode_activation(&frame).expect("decode");
        assert_eq!(decoded.header.request_id, 42);
        assert_eq!(decoded.header.seq_pos, 7);
        assert_eq!(decoded.values, values, "f32 transport must be lossless");
    }

    #[test]
    fn q8_activation_is_four_times_smaller_and_stays_within_one_step() {
        let values = ramp(128);
        let f32_frame = encode_activation(Header::new(OP_ACTIVATION, 1, 0), &values);
        let q8_frame = encode_activation(
            Header::new(OP_ACTIVATION, 1, 0).with_codec(Codec::Q8),
            &values,
        );
        assert!(
            q8_frame.len() * 3 < f32_frame.len(),
            "q8 {} should be far under f32 {}",
            q8_frame.len(),
            f32_frame.len()
        );

        let decoded = decode_activation(&q8_frame).expect("decode");
        let max_abs = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let step = max_abs / 127.0;
        for (i, (&a, &b)) in values.iter().zip(decoded.values.iter()).enumerate() {
            assert!(
                (a - b).abs() <= step * 0.5 + 1e-6,
                "element {i}: {a} vs {b} exceeds half a quantisation step ({step})"
            );
        }
    }

    #[test]
    fn q8_handles_an_all_zero_tensor() {
        let values = vec![0.0f32; 16];
        let frame = encode_activation(
            Header::new(OP_ACTIVATION, 0, 0).with_codec(Codec::Q8),
            &values,
        );
        let decoded = decode_activation(&frame).expect("decode");
        assert!(decoded.values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn token_frame_round_trips_with_worker_timing() {
        let frame = encode_token(
            Header::new(OP_TOKEN, 9, 3).with_flags(FLAG_FINAL),
            &TokenReply {
                token_id: 61,
                compute_us: 1234,
            },
        );
        let (header, reply) = decode_token(&frame).expect("decode");
        assert_eq!(header.request_id, 9);
        assert!(header.is_final());
        assert_eq!(reply.token_id, 61);
        assert_eq!(reply.compute_us, 1234);
    }

    #[test]
    fn truncated_frames_are_rejected_rather_than_panicking() {
        let frame = encode_activation(Header::new(OP_ACTIVATION, 1, 0), &ramp(32));
        for cut in [
            0,
            1,
            HEADER_LEN - 1,
            HEADER_LEN,
            HEADER_LEN + 3,
            frame.len() - 1,
        ] {
            assert!(
                decode_activation(&frame[..cut]).is_err(),
                "truncating to {cut} bytes should fail"
            );
        }
    }

    #[test]
    fn version_and_opcode_mismatches_are_reported_precisely() {
        let mut frame = encode_token(
            Header::new(OP_TOKEN, 1, 0),
            &TokenReply {
                token_id: 1,
                compute_us: 0,
            },
        );
        assert_eq!(
            decode_activation(&frame),
            Err(ProtocolError::UnexpectedOpcode {
                expected: OP_ACTIVATION,
                got: OP_TOKEN
            })
        );

        frame[0] = 99;
        assert_eq!(
            Header::parse(&frame),
            Err(ProtocolError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                got: 99
            })
        );
    }

    #[test]
    fn declared_count_must_match_actual_payload_size() {
        let mut frame = encode_activation(Header::new(OP_ACTIVATION, 1, 0), &ramp(8));
        // Claim 9 elements while only shipping 8.
        frame[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&9u32.to_le_bytes());
        assert!(matches!(
            decode_activation(&frame),
            Err(ProtocolError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn hello_round_trips_including_its_backend_label() {
        let hello = Hello {
            dim: 128,
            n_layers: 2,
            vocab_size: 64,
            max_seq: 128,
            backend: "browser webgpu (adapter hidden)".to_string(),
        };
        let (header, decoded) = decode_hello(&encode_hello(&hello, HELLO_REPLY)).expect("decode");
        assert_eq!(decoded, hello);
        assert_eq!(header.request_id, HELLO_REPLY);
    }

    #[test]
    fn hello_handles_an_empty_backend_label() {
        let hello = Hello {
            dim: 128,
            n_layers: 2,
            vocab_size: 64,
            max_seq: 128,
            backend: String::new(),
        };
        let frame = encode_hello(&hello, HELLO_ANNOUNCE);
        assert_eq!(frame.len(), HEADER_LEN + 20);
        assert_eq!(decode_hello(&frame).expect("decode").1, hello);
    }

    #[test]
    fn hello_rejects_a_truncated_backend_label() {
        let hello = Hello {
            dim: 128,
            n_layers: 2,
            vocab_size: 64,
            max_seq: 128,
            backend: "webgpu".to_string(),
        };
        let frame = encode_hello(&hello, HELLO_ANNOUNCE);
        assert!(decode_hello(&frame[..frame.len() - 2]).is_err());
    }

    #[test]
    fn error_frames_carry_their_message() {
        let frame = encode_error(5, 2, "gpu device lost");
        let (header, msg) = decode_error(&frame).expect("decode");
        assert_eq!(header.request_id, 5);
        assert_eq!(msg, "gpu device lost");
    }
}
