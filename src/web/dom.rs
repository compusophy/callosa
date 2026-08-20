//! Thin DOM helpers.
//!
//! Elements are looked up once by their `data-ref` attribute and cached, so the
//! rest of the client works with typed handles instead of repeated queries.

use std::cell::RefCell;
use std::collections::HashMap;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    Document, Element, Event, EventTarget, HtmlCanvasElement, HtmlInputElement, HtmlSelectElement,
    HtmlTextAreaElement, Window,
};

pub fn window() -> Window {
    web_sys::window().expect("no window; this build only runs in a browser")
}

pub fn document() -> Document {
    window().document().expect("no document")
}

/// Monotonic microseconds since page load.
pub fn now_us() -> f64 {
    window()
        .performance()
        .map(|p| p.now() * 1000.0)
        .unwrap_or(0.0)
}

/// Every `[data-ref]` element on the page, keyed by its ref name.
pub struct Refs {
    map: HashMap<String, Element>,
}

impl Refs {
    pub fn collect() -> Self {
        let mut map = HashMap::new();
        let nodes = document()
            .query_selector_all("[data-ref]")
            .expect("query_selector_all failed");
        for i in 0..nodes.length() {
            if let Some(el) = nodes.item(i).and_then(|n| n.dyn_into::<Element>().ok()) {
                if let Some(name) = el.get_attribute("data-ref") {
                    map.insert(name, el);
                }
            }
        }
        Refs { map }
    }

    /// Panics when a ref is missing: the markup and this client ship together,
    /// so a missing ref is a build mistake, not a runtime condition.
    pub fn get(&self, name: &str) -> &Element {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("no element with data-ref=\"{name}\""))
    }

    pub fn canvas(&self, name: &str) -> HtmlCanvasElement {
        self.get(name).clone().unchecked_into()
    }

    pub fn set_text(&self, name: &str, value: &str) {
        let el = self.get(name);
        if el.text_content().as_deref() != Some(value) {
            el.set_text_content(Some(value));
        }
    }

    pub fn set_attr(&self, name: &str, attr: &str, value: &str) {
        let _ = self.get(name).set_attribute(attr, value);
    }

    pub fn set_hidden(&self, name: &str, hidden: bool) {
        if hidden {
            let _ = self.get(name).set_attribute("hidden", "");
        } else {
            let _ = self.get(name).remove_attribute("hidden");
        }
    }

    pub fn set_disabled(&self, name: &str, disabled: bool) {
        let el = self.get(name);
        if disabled {
            let _ = el.set_attribute("disabled", "");
        } else {
            let _ = el.remove_attribute("disabled");
        }
    }

    pub fn value(&self, name: &str) -> String {
        let el = self.get(name);
        if let Some(input) = el.dyn_ref::<HtmlInputElement>() {
            return input.value();
        }
        if let Some(select) = el.dyn_ref::<HtmlSelectElement>() {
            return select.value();
        }
        if let Some(area) = el.dyn_ref::<HtmlTextAreaElement>() {
            return area.value();
        }
        el.text_content().unwrap_or_default()
    }

    pub fn set_value(&self, name: &str, value: &str) {
        let el = self.get(name);
        if let Some(input) = el.dyn_ref::<HtmlInputElement>() {
            input.set_value(value);
        } else if let Some(select) = el.dyn_ref::<HtmlSelectElement>() {
            select.set_value(value);
        } else if let Some(area) = el.dyn_ref::<HtmlTextAreaElement>() {
            area.set_value(value);
        }
    }

    pub fn number(&self, name: &str, fallback: f64) -> f64 {
        self.value(name).trim().parse::<f64>().unwrap_or(fallback)
    }
}

/// Attach a listener that lives as long as the page.
///
/// The client is the page, so leaking these closures is the correct lifetime
/// rather than a leak to be cleaned up.
pub fn on_event<F>(target: &EventTarget, event: &str, handler: F)
where
    F: FnMut(Event) + 'static,
{
    let closure = Closure::<dyn FnMut(Event)>::new(handler);
    target
        .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
        .expect("add_event_listener failed");
    closure.forget();
}

/// A listener whose lifetime is owned by the caller, for objects that get torn
/// down and rebuilt (the peer connection and its data channel).
pub struct Listener {
    target: EventTarget,
    event: String,
    closure: Closure<dyn FnMut(Event)>,
}

impl Listener {
    pub fn attach<F>(target: &EventTarget, event: &str, handler: F) -> Self
    where
        F: FnMut(Event) + 'static,
    {
        let closure = Closure::<dyn FnMut(Event)>::new(handler);
        target
            .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
            .expect("add_event_listener failed");
        Listener {
            target: target.clone(),
            event: event.to_string(),
            closure,
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        let _ = self.target.remove_event_listener_with_callback(
            &self.event,
            self.closure.as_ref().unchecked_ref(),
        );
    }
}

/// Resolve after `millis`, for timeouts inside async flows.
pub async fn sleep(millis: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, millis)
            .expect("set_timeout failed");
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Run `handler` once after `millis`.
pub fn set_timeout<F>(millis: i32, handler: F) -> TimeoutHandle
where
    F: FnOnce() + 'static,
{
    let cell = RefCell::new(Some(handler));
    let closure = Closure::once(move || {
        if let Some(f) = cell.borrow_mut().take() {
            f();
        }
    });
    let id = window()
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            closure.as_ref().unchecked_ref(),
            millis,
        )
        .expect("set_timeout failed");
    TimeoutHandle {
        id,
        _closure: closure,
    }
}

pub struct TimeoutHandle {
    id: i32,
    _closure: Closure<dyn FnMut()>,
}

impl Drop for TimeoutHandle {
    fn drop(&mut self) {
        window().clear_timeout_with_handle(self.id);
    }
}

/// Repeat `handler` every `millis` until the handle is dropped.
pub fn set_interval<F>(millis: i32, handler: F) -> IntervalHandle
where
    F: FnMut() + 'static,
{
    let closure = Closure::<dyn FnMut()>::new(handler);
    let id = window()
        .set_interval_with_callback_and_timeout_and_arguments_0(
            closure.as_ref().unchecked_ref(),
            millis,
        )
        .expect("set_interval failed");
    IntervalHandle {
        id,
        _closure: closure,
    }
}

pub struct IntervalHandle {
    id: i32,
    _closure: Closure<dyn FnMut()>,
}

impl Drop for IntervalHandle {
    fn drop(&mut self) {
        window().clear_interval_with_handle(self.id);
    }
}

/// Query parameter from the current URL.
pub fn query_param(name: &str) -> Option<String> {
    let href = window().location().href().ok()?;
    let url = web_sys::Url::new(&href).ok()?;
    let value = url.search_params().get(name)?;
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

pub fn js_error_string(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(value, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"))
}
