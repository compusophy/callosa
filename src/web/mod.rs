//! The browser client, written in Rust.
//!
//! There is no application JavaScript: the DOM, canvas rendering, WebRTC and
//! the pairing transports are all driven from here through `web-sys`. The only
//! script the page loads is the wasm-bindgen glue plus a two-line bootstrap.

mod app;
mod dom;
mod peer;
mod signal;
mod viz;

use wasm_bindgen::prelude::*;

/// Entry point. `wasm-bindgen(start)` runs this as soon as the module
/// initialises, so the page needs nothing beyond `await init()`.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    app::App::boot();
}
