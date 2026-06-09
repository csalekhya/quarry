//! Plain-text rendering of query results.

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;

use crate::types::{format_date, ScalarValue};

/// Render one value the way the CLI and the README show it.
pub fn format_value(v: &ScalarValue) -> String {
    if v.is_null() {
        return "NULL".to_string();
    }
    match v {
        ScalarValue::Boolean(Some(b)) => b.to_string(),
        ScalarValue::Int64(Some(i)) => i.to_string(),
        ScalarValue::Float64(Some(f)) => format_f64(*f),
        ScalarValue::Utf8(Some(s)) => s.clone(),
        ScalarValue::Date32(Some(d)) => format_date(*d),
        _ => "NULL".to_string(),
    }
}

/// Up to four decimals, trailing zeros trimmed (keeping one decimal).
pub fn format_f64(f: f64) -> String {
    if !f.is_finite() {
        return f.to_string();
    }
    let s = format!("{f:.4}");
    let s = s.trim_end_matches('0');
    if s.ends_with('.') {
        format!("{s}0")
    } else {
        s.to_string()
    }
}

/// An ASCII table with a header row and right-aligned numbers.
pub fn format_table(schema: &SchemaRef, batches: &[RecordBatch]) -> String {
    let headers: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let mut rows: Vec<Vec<(String, bool)>> = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| match ScalarValue::from_array(c.as_ref(), i) {
                        Ok(v) => {
                            let numeric =
                                matches!(v, ScalarValue::Int64(_) | ScalarValue::Float64(_));
                            (format_value(&v), numeric)
                        }
                        Err(_) => ("?".to_string(), false),
                    })
                    .collect(),
            );
        }
    }
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for r in &rows {
        for (w, (s, _)) in widths.iter_mut().zip(r) {
            *w = (*w).max(s.chars().count());
        }
    }
    let sep: String = widths
        .iter()
        .map(|w| "-".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("+");
    let mut out = String::new();
    let line = |cells: Vec<String>| format!(" {}\n", cells.join(" | "));
    out.push_str(&line(
        headers
            .iter()
            .zip(&widths)
            .map(|(h, w)| format!("{h:<w$}"))
            .collect(),
    ));
    out.push_str(&format!("{sep}\n"));
    for r in &rows {
        out.push_str(&line(
            r.iter()
                .zip(&widths)
                .map(|((s, numeric), w)| {
                    if *numeric {
                        format!("{s:>w$}")
                    } else {
                        format!("{s:<w$}")
                    }
                })
                .collect(),
        ));
    }
    out.push_str(&format!(
        "({} row{})\n",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_formatting() {
        assert_eq!(format_f64(2.0), "2.0");
        assert_eq!(format_f64(0.1 + 0.2), "0.3");
        assert_eq!(format_f64(1234.56789), "1234.5679");
        assert_eq!(format_value(&ScalarValue::Date32(Some(0))), "1970-01-01");
        assert_eq!(format_value(&ScalarValue::Utf8(None)), "NULL");
    }
}
