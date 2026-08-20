# Build
FROM rust:1-slim-bookworm AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# Run: no toolchain, no shell tooling, just the binary.
FROM debian:bookworm-slim
RUN useradd --system --uid 10001 relay
COPY --from=build /app/target/release/callosa-relay /usr/local/bin/callosa-relay
USER relay
ENV PORT=8080
EXPOSE 8080
CMD ["callosa-relay"]
