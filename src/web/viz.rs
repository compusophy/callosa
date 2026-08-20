//! Canvas rendering for the activation tensor and the latency history.
//!
//! Both size their backing store to `devicePixelRatio` and to the element's real
//! CSS box, so the drawing stays sharp and never stretches when the layout
//! changes.

use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::dom::window;

pub struct Accent {
    pub positive: &'static str,
    pub negative: &'static str,
}

pub const NODE0_ACCENT: Accent = Accent {
    positive: "#22d3ee",
    negative: "rgba(34, 211, 238, 0.32)",
};

pub const NODE1_ACCENT: Accent = Accent {
    positive: "#a78bfa",
    negative: "rgba(167, 139, 250, 0.32)",
};

pub struct LatencySample {
    pub total_ms: f64,
    pub local_ms: f64,
    pub remote_ms: f64,
}

fn fit(canvas: &HtmlCanvasElement) -> Option<(CanvasRenderingContext2d, f64, f64)> {
    let dpr = window().device_pixel_ratio().max(1.0);
    let rect = canvas.get_bounding_client_rect();
    let (css_w, css_h) = (rect.width(), rect.height());
    if css_w <= 0.0 || css_h <= 0.0 {
        return None;
    }

    let want_w = (css_w * dpr).round() as u32;
    let want_h = (css_h * dpr).round() as u32;
    if canvas.width() != want_w {
        canvas.set_width(want_w);
    }
    if canvas.height() != want_h {
        canvas.set_height(want_h);
    }

    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .ok()??
        .dyn_into::<CanvasRenderingContext2d>()
        .ok()?;
    let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
    Some((ctx, css_w, css_h))
}

use wasm_bindgen::JsCast;

/// Diverging bar view of an activation vector, scaled to the tensor's own range
/// so a quiet layer stays legible.
pub fn draw_activations(canvas: &HtmlCanvasElement, values: &[f32], accent: &Accent) {
    let Some((ctx, width, height)) = fit(canvas) else {
        return;
    };
    ctx.clear_rect(0.0, 0.0, width, height);
    if values.is_empty() {
        return;
    }

    let mid = height / 2.0;
    ctx.set_stroke_style_str("rgba(255,255,255,0.08)");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(0.0, mid);
    ctx.line_to(width, mid);
    ctx.stroke();

    let peak = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let peak = if peak == 0.0 { 1.0 } else { peak };

    let bar_width = width / values.len() as f64;
    let gap = if bar_width > 3.0 { 1.0 } else { 0.0 };

    for (i, &value) in values.iter().enumerate() {
        let normalized = (value / peak) as f64;
        let bar_height = normalized.abs() * (mid - 2.0);
        let x = i as f64 * bar_width;
        let y = if normalized >= 0.0 {
            mid - bar_height
        } else {
            mid
        };

        let color = if normalized >= 0.0 {
            accent.positive
        } else {
            accent.negative
        };
        ctx.set_fill_style_str(color);
        ctx.fill_rect(x, y, (bar_width - gap).max(1.0), bar_height.max(1.0));
    }
}

/// Draw `text` as a QR code, sized to fill the canvas on whole-pixel modules.
///
/// Rendered here rather than fetched as an image so the page stays a single
/// self-contained wasm bundle with no network dependency.
pub fn draw_qr(canvas: &HtmlCanvasElement, text: &str) -> Result<(), String> {
    let code = qrcode::QrCode::new(text.as_bytes())
        .map_err(|e| format!("could not encode the invite as a qr code: {e}"))?;
    let colors = code.to_colors();
    let modules = code.width();

    let Some((ctx, width, height)) = fit(canvas) else {
        return Ok(());
    };

    // A quiet zone is part of the spec; scanners need it to find the symbol.
    const QUIET: usize = 2;
    let total = modules + QUIET * 2;
    let side = width.min(height);
    let scale = (side / total as f64).floor().max(1.0);
    let drawn = scale * total as f64;
    let origin_x = ((width - drawn) / 2.0).floor();
    let origin_y = ((height - drawn) / 2.0).floor();

    // White background, including the quiet zone: contrast is what gets scanned.
    ctx.set_fill_style_str("#ffffff");
    ctx.fill_rect(origin_x, origin_y, drawn, drawn);

    ctx.set_fill_style_str("#05060a");
    for (i, color) in colors.iter().enumerate() {
        if *color == qrcode::Color::Light {
            continue;
        }
        let x = (i % modules) + QUIET;
        let y = (i / modules) + QUIET;
        ctx.fill_rect(
            origin_x + x as f64 * scale,
            origin_y + y as f64 * scale,
            scale,
            scale,
        );
    }
    Ok(())
}

/// Rolling latency history: total round trip with the two compute halves stacked.
pub fn draw_latency(canvas: &HtmlCanvasElement, samples: &[LatencySample]) {
    let Some((ctx, width, height)) = fit(canvas) else {
        return;
    };
    ctx.clear_rect(0.0, 0.0, width, height);

    if samples.is_empty() {
        ctx.set_fill_style_str("rgba(255,255,255,0.25)");
        ctx.set_font("11px ui-monospace, monospace");
        let _ = ctx.fill_text("no samples yet", 8.0, height / 2.0 + 4.0);
        return;
    }

    let peak = samples.iter().fold(1.0f64, |m, s| m.max(s.total_ms));
    let slot = width / (samples.len().max(12) as f64);
    let bar_width = (slot - 2.0).max(1.0);
    let scale = (height - 4.0) / peak;

    for (i, sample) in samples.iter().enumerate() {
        let x = i as f64 * slot;
        let local = sample.local_ms * scale;
        let remote = sample.remote_ms * scale;
        let network = (sample.total_ms * scale - local - remote).max(0.0);

        let mut y = height;
        for (value, color) in [
            (local, "#22d3ee"),
            (network, "#fbbf24"),
            (remote, "#a78bfa"),
        ] {
            if value <= 0.0 {
                continue;
            }
            y -= value;
            ctx.set_fill_style_str(color);
            ctx.fill_rect(x, y, bar_width, value);
        }
    }

    ctx.set_fill_style_str("rgba(255,255,255,0.4)");
    ctx.set_font("10px ui-monospace, monospace");
    let _ = ctx.fill_text(&format!("peak {peak:.1} ms"), 6.0, 12.0);
}
