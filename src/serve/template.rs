//! S.1: chat-template rendering (minijinja + pycompat).
//!
//! Loads the template from the model dir: `chat_template.jinja` if present,
//! else the `chat_template` string embedded in `tokenizer_config.json`.
//! GATE: byte-identical rendering vs HF transformers `apply_chat_template`
//! (tests/fixtures/template/). See docs/SERVE_PLAN.md S.1.
//!
//! ## Why the machinery below exists (parity traps, all proven by the gate)
//!
//! HF renders with `jinja2.ImmutableSandboxedEnvironment(trim_blocks=True,
//! lstrip_blocks=True)` and installs a custom `tojson` == Python
//! `json.dumps(x, ensure_ascii=False)` (default separators `", "` / `": "`).
//! To reproduce byte-for-byte we must, on the minijinja side:
//!   * enable `trim_blocks` + `lstrip_blocks` (matches transformers);
//!   * install the pycompat `unknown_method_callback` so `.lstrip/.rstrip/
//!     .split/.startswith/.endswith` work (the 35B template uses all of them);
//!   * override `tojson` — minijinja's builtin is COMPACT (no spaces) and
//!     HTML-escapes, both of which break parity;
//!   * define `raise_exception` (a transformers helper, not a jinja builtin);
//!   * preserve JSON object key ORDER through `tojson`.
//!
//! ## The key-order problem (and why we don't use `serde_json::Value`)
//!
//! HF's `tojson` preserves the JSON's INSERTION order (the `tool_defs` fixture
//! emits `{"type": ..., "function": ...}` — NOT alphabetical). But in this build
//! BOTH `serde_json::Value` AND `minijinja::Value` back their object maps with
//! `BTreeMap` (their respective `preserve_order` features are OFF and Cargo.toml
//! is frozen), so either would SORT the keys and break parity. The fix: an
//! order-preserving [`OrderedJson`] type (objects = `Vec<(String, _)>`) carried
//! into the template as an opaque minijinja [`Object`] ([`ToolValue`]); the
//! custom `tojson` downcasts it and serializes with insertion order intact.
//!
//! This applies to `messages` too, NOT just `tools`: the 35B template `tojson`s
//! message subtrees on the tool-replay path (`tool_call.arguments | tojson`), so
//! `messages` also enters as [`OrderedJson`] — routing it through
//! `Value::from_serialize(serde_json::Value)` would silently sort those argument
//! keys and diverge from HF. Conversion keeps JSON leaves as NATIVE minijinja
//! values (strings/numbers/bools/none stay primitive so `content is string`,
//! comparisons and iteration keep working; arrays become `Vec<Value>`); ONLY
//! objects need the opaque order-preserving [`ToolValue`] wrapper.
//!
//! ## Accepted limitation: huge integers
//!
//! `serde_json` is used WITHOUT `arbitrary_precision` (deliberately not enabled
//! crate-wide), so an integer literal in request JSON that exceeds `u64`/`i64`
//! range is parsed as `f64` at request-parse time and its exact digits do not
//! survive. HF (Python, arbitrary-precision ints) would keep them, so such
//! exotic tool JSON could diverge. Normal-range ints and floats are exact and
//! render Python-`repr`-identically (see [`py_format_f64`]).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use minijinja::value::{Enumerator, Object, ObjectRepr};
use minijinja::{Environment, Error as MjError, ErrorKind as MjErrorKind, Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from loading or rendering a chat template.
#[derive(Debug, Error)]
pub enum TemplateError {
    /// Neither `chat_template.jinja` nor a `chat_template` string in
    /// `tokenizer_config.json` was found in the model dir.
    #[error(
        "no chat template in {0}: neither chat_template.jinja nor \
         tokenizer_config.json:chat_template (NEVER hand-roll a ChatML fallback)"
    )]
    NotFound(String),
    /// Failed to read a file in the model dir.
    #[error("i/o error reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    /// `tokenizer_config.json` did not parse as JSON.
    #[error("parsing {path}: {source}")]
    ConfigJson {
        path: String,
        source: serde_json::Error,
    },
    /// The minijinja template failed to compile.
    #[error("compiling chat template: {0}")]
    Compile(MjError),
    /// The minijinja template failed to render.
    #[error("rendering chat template: {0}")]
    Render(MjError),
}

/// A loaded, self-contained chat template (owns its minijinja [`Environment`]).
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Load the chat template from a model directory with the EXACT HF priority:
    /// (a) `<dir>/chat_template.jinja` file if it exists (the 35B ships one);
    /// (b) else the `chat_template` STRING inside `<dir>/tokenizer_config.json`
    ///     (the 30B-instruct embeds a full 2630-char template this way).
    /// Errors if neither exists — there is deliberately NO built-in fallback.
    pub fn from_model_dir(dir: &Path) -> Result<Self, TemplateError> {
        let jinja_path = dir.join("chat_template.jinja");
        let source = if jinja_path.exists() {
            std::fs::read_to_string(&jinja_path).map_err(|e| TemplateError::Io {
                path: jinja_path.display().to_string(),
                source: e,
            })?
        } else {
            let cfg_path = dir.join("tokenizer_config.json");
            let text = match std::fs::read_to_string(&cfg_path) {
                Ok(t) => t,
                // Missing config == no template found (not an I/O surprise).
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(TemplateError::NotFound(dir.display().to_string()));
                }
                Err(e) => {
                    return Err(TemplateError::Io {
                        path: cfg_path.display().to_string(),
                        source: e,
                    });
                }
            };
            let cfg: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| TemplateError::ConfigJson {
                    path: cfg_path.display().to_string(),
                    source: e,
                })?;
            match cfg.get("chat_template").and_then(|v| v.as_str()) {
                Some(s) => s.to_owned(),
                None => return Err(TemplateError::NotFound(dir.display().to_string())),
            }
        };
        Self::from_source(source)
    }

    /// Build the environment from raw Jinja source.
    fn from_source(source: String) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        // Match transformers' jinja2 environment exactly.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        // Python str/list methods used by the 35B template.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Python json.dumps-compatible tojson (see module docs).
        env.add_filter("tojson", py_tojson);
        // transformers helper; only fires on error branches (never in valid inputs).
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat", source)
            .map_err(TemplateError::Compile)?;
        Ok(Self { env })
    }

    /// Render the chat template.
    ///
    /// * `messages` — OpenAI-style JSON array as an [`OrderedJson`] (NOT
    ///   `serde_json::Value`): the 35B template `tojson`s message subtrees on the
    ///   tool-replay path (`tool_call.arguments | tojson`), so object key order is
    ///   load-bearing for byte parity. Converted the same way as `tools`
    ///   ([`ordered_to_value`]): leaves stay native, objects become opaque
    ///   order-preserving [`ToolValue`]s.
    /// * `tools` — when `Some`, defines the `tools` variable; when `None`, leaves
    ///   it UNDEFINED (HF only passes the kwarg when provided, and both templates
    ///   gate on `{%- if tools %}`). Typed as [`OrderedJson`] rather than
    ///   `serde_json::Value` because tool-def key ORDER is load-bearing for byte
    ///   parity (see module docs) — expected to be an [`OrderedJson::Array`].
    /// * `add_generation_prompt` — always defined as a bool.
    /// * `enable_thinking` — `Some(b)` defines it as bool `b`; `None` leaves it
    ///   UNDEFINED so the templates' `enable_thinking is defined` checks see it
    ///   as absent (defining it as `none` would change behavior).
    pub fn render(
        &self,
        messages: &OrderedJson,
        tools: Option<&OrderedJson>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String, TemplateError> {
        let mut ctx: BTreeMap<&'static str, Value> = BTreeMap::new();
        // messages: OrderedJson -> minijinja Value the SAME way tools convert, so
        // objects keep insertion order (tool_calls[].arguments | tojson parity) and
        // leaves stay native (`content is string`, iteration, comparisons work).
        ctx.insert("messages", ordered_to_value(messages));
        ctx.insert("add_generation_prompt", Value::from(add_generation_prompt));
        if let Some(b) = enable_thinking {
            ctx.insert("enable_thinking", Value::from(b));
        }
        // else: leave `enable_thinking` UNDEFINED (do not insert).
        if let Some(tools) = tools {
            ctx.insert("tools", tools_to_value(tools));
        }
        // else: leave `tools` UNDEFINED.

        // BTreeMap<&str, Value> -> a live map Value. Passing it to render()
        // round-trips through minijinja's in-band value-handle signalling, so the
        // nested ToolValue objects survive intact (no re-serialization flattening).
        let context = Value::from_object(ctx);
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(TemplateError::Render)?;
        tmpl.render(context).map_err(TemplateError::Render)
    }
}

/// Convert an order-preserving `tools` value into the minijinja `tools` variable.
/// An array becomes a sequence of opaque [`ToolValue`] objects (so `for tool in
/// tools` iterates and `tool | tojson` preserves order); anything else is wrapped
/// as a single opaque object.
fn tools_to_value(tools: &OrderedJson) -> Value {
    match tools {
        OrderedJson::Array(items) => {
            let vals = items
                .iter()
                .map(|t| Value::from_object(ToolValue(t.clone())));
            Value::from_iter(vals)
        }
        other => Value::from_object(ToolValue(other.clone())),
    }
}

/// transformers' `raise_exception(msg)` helper. Not a jinja builtin; only ever
/// hit on the templates' error branches (unreachable for valid inputs).
fn raise_exception(msg: String) -> Result<Value, MjError> {
    Err(MjError::new(MjErrorKind::InvalidOperation, msg))
}

// ---------------------------------------------------------------------------
// tojson == Python json.dumps(x, ensure_ascii=False), default separators.
// ---------------------------------------------------------------------------

/// serde_json formatter that reproduces Python's DEFAULT separators: a space
/// after every `,` and every `:`. All string escaping is inherited from the
/// trait defaults, which already match Python's `ensure_ascii=False`
/// (`"` and `\` and control chars escaped; `/` and non-ASCII emitted raw).
///
/// It ALSO overrides float output: serde_json's default `write_f64` diverges
/// from Python `repr` (e.g. Rust `0.00001` vs Python `1e-05`), which would break
/// tojson byte parity for any tool JSON carrying floats. See [`py_format_f64`].
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn write_f64<W: ?Sized + std::io::Write>(&mut self, w: &mut W, value: f64) -> std::io::Result<()> {
        w.write_all(py_format_f64(value).as_bytes())
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Serialize anything with Python `json.dumps(..., ensure_ascii=False)` formatting.
fn to_python_json<T: Serialize>(value: &T) -> Result<String, MjError> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
    value
        .serialize(&mut ser)
        .map_err(|e| MjError::new(MjErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    String::from_utf8(buf)
        .map_err(|e| MjError::new(MjErrorKind::InvalidOperation, format!("tojson utf8: {e}")))
}

/// Format `x` exactly as Python's `repr(float)` / `json.dumps(float)` does, so
/// tojson stays byte-identical to HF for floats.
///
/// Method: Rust's `{:e}` already yields the SHORTEST round-trip mantissa + a
/// base-10 exponent (`value = d[0].d[1..] * 10^exp`). From it we recover the
/// significant-digit string and `decpt` (the decimal-point position, where
/// `value = 0.<digits> * 10^decpt`), then apply CPython `format_float_short`'s
/// 'r' rule:
///   * scientific notation iff `decpt <= -4 || decpt > 16`, exponent `= decpt-1`
///     rendered `e±NN` with at least two digits (`1e-05`, `1e+30`, `1e+100`);
///   * otherwise fixed notation, with a trailing `.0` for integral values
///     (`1.0`, `100.0`, `1000000000000000.0`).
/// Non-finite inputs match Python's json extension (`NaN` / `Infinity` /
/// `-Infinity`). (In practice serde_json routes non-finite floats to `write_null`
/// before reaching here, so those arms are defensive.)
fn py_format_f64(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-Infinity" } else { "Infinity" }.to_string();
    }

    let neg = x.is_sign_negative(); // catches -0.0 as well
    let sci = format!("{:e}", x.abs()); // e.g. "1.5e-5", "1e0", "9.999e15"
    let (mant, exp_str) = sci.split_once('e').expect("{:e} always contains 'e'");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is an integer");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.<digits> * 10^decpt
    let ndigits = digits.len() as i32;

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        // Scientific: <first>[.<rest>]e±NN
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        // 0.<zeros><digits>
        out.push_str("0.");
        out.extend(std::iter::repeat('0').take((-decpt) as usize));
        out.push_str(&digits);
    } else if decpt >= ndigits {
        // <digits><zeros>.0  (integral value)
        out.push_str(&digits);
        out.extend(std::iter::repeat('0').take((decpt - ndigits) as usize));
        out.push_str(".0");
    } else {
        // <digits[..decpt]>.<digits[decpt..]>
        let d = decpt as usize;
        out.push_str(&digits[..d]);
        out.push('.');
        out.push_str(&digits[d..]);
    }
    out
}

/// The `tojson` filter. For our order-preserving [`ToolValue`] objects it
/// serializes the underlying [`OrderedJson`] (insertion order intact); for any
/// other minijinja value it serializes the value directly (used by the
/// tool-call `arguments | tojson` path, which is not order-sensitive here).
/// Returns a safe string so autoescape never touches it.
fn py_tojson(value: &Value) -> Result<Value, MjError> {
    let s = if let Some(tool) = value.downcast_object_ref::<ToolValue>() {
        to_python_json(&tool.0)?
    } else {
        to_python_json(value)?
    };
    Ok(Value::from_safe_string(s))
}

// ---------------------------------------------------------------------------
// OrderedJson: an insertion-order-preserving JSON value.
// ---------------------------------------------------------------------------

/// A JSON value that preserves object key insertion order (objects are a
/// `Vec<(String, _)>`, unlike `serde_json::Value`/`minijinja::Value` which use
/// `BTreeMap` in this build and would SORT keys). Deserializing from any serde
/// data format keeps document order; serializing emits in that same order.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedJson>),
    Object(Vec<(String, OrderedJson)>),
}

impl Serialize for OrderedJson {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            OrderedJson::Null => s.serialize_unit(),
            OrderedJson::Bool(b) => s.serialize_bool(*b),
            OrderedJson::Number(n) => n.serialize(s),
            OrderedJson::String(v) => s.serialize_str(v),
            OrderedJson::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for it in items {
                    seq.serialize_element(it)?;
                }
                seq.end()
            }
            OrderedJson::Object(entries) => {
                let mut map = s.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?; // Vec order == emitted order.
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedJson;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }

            fn visit_bool<E>(self, v: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<OrderedJson, E> {
                Ok(serde_json::Number::from_f64(v)
                    .map(OrderedJson::Number)
                    .unwrap_or(OrderedJson::Null))
            }
            fn visit_str<E>(self, v: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v))
            }
            fn visit_none<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_some<D: serde::Deserializer<'de>>(
                self,
                d: D,
            ) -> Result<OrderedJson, D::Error> {
                OrderedJson::deserialize(d)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<OrderedJson, A::Error> {
                let mut items = Vec::new();
                while let Some(el) = seq.next_element()? {
                    items.push(el);
                }
                Ok(OrderedJson::Array(items))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<OrderedJson, A::Error> {
                // MapAccess yields entries in document order -> preserved in the Vec.
                // Duplicate keys are LAST-WINS to match Python `json.loads`, which
                // (like a Python dict) keeps the key's FIRST position but the LAST
                // value: overwrite the existing entry's value in place.
                let mut entries: Vec<(String, OrderedJson)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, OrderedJson>()? {
                    if let Some(slot) = entries.iter_mut().find(|(kk, _)| *kk == k) {
                        slot.1 = v;
                    } else {
                        entries.push((k, v));
                    }
                }
                Ok(OrderedJson::Object(entries))
            }
        }
        d.deserialize_any(V)
    }
}

/// Convert an [`OrderedJson`] into a minijinja [`Value`].
///
/// Leaves become NATIVE minijinja values (so `x is string`, numeric comparisons,
/// truthiness, and `content is none`/`is undefined` all behave like plain JSON);
/// arrays become a native `Vec<Value>` (a proper seq: iterable, `is not mapping`,
/// slicing, `| length`); ONLY objects need the opaque order-preserving
/// [`ToolValue`] wrapper (insertion-order field access, `in`/`is defined`,
/// `| items`, and `| tojson` via downcast).
fn ordered_to_value(j: &OrderedJson) -> Value {
    match j {
        OrderedJson::Null => Value::from(()),
        OrderedJson::Bool(b) => Value::from(*b),
        OrderedJson::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(u) = n.as_u64() {
                Value::from(u)
            } else if let Some(f) = n.as_f64() {
                Value::from(f)
            } else {
                Value::from(n.to_string())
            }
        }
        OrderedJson::String(s) => Value::from(s.clone()),
        OrderedJson::Array(items) => {
            Value::from(items.iter().map(ordered_to_value).collect::<Vec<Value>>())
        }
        OrderedJson::Object(_) => Value::from_object(ToolValue(j.clone())),
    }
}

/// An opaque, order-preserving minijinja object wrapping an [`OrderedJson`].
/// It behaves like a map (field access / iteration in insertion order) AND is
/// recognized by [`py_tojson`] via downcast so `| tojson` emits ordered JSON.
#[derive(Debug)]
struct ToolValue(OrderedJson);

impl Object for ToolValue {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        match self.0 {
            OrderedJson::Object(_) => ObjectRepr::Map,
            OrderedJson::Array(_) => ObjectRepr::Seq,
            _ => ObjectRepr::Plain,
        }
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        match &self.0 {
            OrderedJson::Object(entries) => {
                let k = key.as_str()?;
                entries
                    .iter()
                    .find(|(kk, _)| kk == k)
                    .map(|(_, v)| ordered_to_value(v))
            }
            OrderedJson::Array(items) => {
                let i = key.as_usize()?;
                items.get(i).map(ordered_to_value)
            }
            _ => None,
        }
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        match &self.0 {
            OrderedJson::Object(entries) => {
                let keys: Vec<Value> =
                    entries.iter().map(|(k, _)| Value::from(k.clone())).collect();
                Enumerator::Values(keys)
            }
            OrderedJson::Array(items) => Enumerator::Seq(items.len()),
            _ => Enumerator::NonEnumerable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_format_f64_matches_python_repr() {
        // Goldens computed with `python3 -c 'import json; print(json.dumps(v))'`.
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (1e30, "1e+30"),
            (123456789.123, "123456789.123"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (100.0, "100.0"),
            (-2.5e-8, "-2.5e-08"),
            (1e100, "1e+100"),
        ];
        for (v, want) in cases {
            assert_eq!(&py_format_f64(*v), want, "py_format_f64({v})");
        }
    }

    #[test]
    fn tojson_float_uses_python_formatting() {
        // Full tojson path exercises PyFormatter::write_f64 on a bare float leaf …
        let n = OrderedJson::Number(serde_json::Number::from_f64(1e-5).unwrap());
        assert_eq!(to_python_json(&n).unwrap(), "1e-05");
        // … and nested in an object (order + `", "`/`": "` separators + float).
        let obj: OrderedJson = serde_json::from_str(r#"{"scale": 1.5e-5, "n": 1.0}"#).unwrap();
        assert_eq!(
            to_python_json(&obj).unwrap(),
            r#"{"scale": 1.5e-05, "n": 1.0}"#
        );
    }

    #[test]
    fn ordered_json_duplicate_keys_last_wins() {
        // Python `json.loads` keeps a duplicate key's FIRST position but LAST value:
        //   json.dumps(json.loads('{"a":1,"b":2,"a":3}')) == '{"a": 3, "b": 2}'
        let j: OrderedJson = serde_json::from_str(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(to_python_json(&j).unwrap(), r#"{"a": 3, "b": 2}"#);
    }
}
