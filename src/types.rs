//! SQL data types, scalar values, and calendar arithmetic.
//!
//! Quarry deliberately supports a small, closed set of SQL types. Every Arrow
//! column that enters the engine is normalized to one of them at scan time
//! (for example `Int32` becomes `Int64`, `Float32` becomes `Float64`), which
//! keeps the kernels and the type checker small.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    new_null_array, Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array,
    StringArray,
};
use arrow::datatypes::DataType as ArrowType;

use crate::error::{exec_err, QuarryError, Result};

/// The SQL types understood by the binder and the execution kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    Int64,
    Float64,
    Utf8,
    Date32,
    /// The type of an untyped `NULL` literal; coerces to anything.
    Null,
}

impl DataType {
    /// The Arrow physical type used to store values of this SQL type.
    pub fn to_arrow(self) -> ArrowType {
        match self {
            DataType::Boolean => ArrowType::Boolean,
            DataType::Int64 => ArrowType::Int64,
            DataType::Float64 => ArrowType::Float64,
            DataType::Utf8 => ArrowType::Utf8,
            DataType::Date32 => ArrowType::Date32,
            DataType::Null => ArrowType::Null,
        }
    }

    /// Map an Arrow type onto the SQL type it is normalized to, if supported.
    pub fn from_arrow(t: &ArrowType) -> Option<DataType> {
        Some(match t {
            ArrowType::Boolean => DataType::Boolean,
            ArrowType::Int8
            | ArrowType::Int16
            | ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::UInt8
            | ArrowType::UInt16
            | ArrowType::UInt32 => DataType::Int64,
            ArrowType::Float16 | ArrowType::Float32 | ArrowType::Float64 => DataType::Float64,
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => DataType::Utf8,
            ArrowType::Date32 | ArrowType::Date64 => DataType::Date32,
            ArrowType::Null => DataType::Null,
            _ => return None,
        })
    }

    pub fn is_numeric(self) -> bool {
        matches!(self, DataType::Int64 | DataType::Float64)
    }

    /// Whether values of this type have a total order usable by `<`, `MIN`, `ORDER BY`.
    pub fn is_orderable(self) -> bool {
        !matches!(self, DataType::Null)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Boolean => "BOOLEAN",
            DataType::Int64 => "BIGINT",
            DataType::Float64 => "DOUBLE",
            DataType::Utf8 => "VARCHAR",
            DataType::Date32 => "DATE",
            DataType::Null => "NULL",
        };
        f.write_str(s)
    }
}

/// A single typed value. `None` payloads are typed SQL NULLs.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Boolean(Option<bool>),
    Int64(Option<i64>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    Date32(Option<i32>),
    /// An untyped NULL.
    Null,
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
            ScalarValue::Date32(_) => DataType::Date32,
            ScalarValue::Null => DataType::Null,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(
            self,
            ScalarValue::Null
                | ScalarValue::Boolean(None)
                | ScalarValue::Int64(None)
                | ScalarValue::Float64(None)
                | ScalarValue::Utf8(None)
                | ScalarValue::Date32(None)
        )
    }

    /// A NULL of the given type.
    pub fn null_of(t: DataType) -> ScalarValue {
        match t {
            DataType::Boolean => ScalarValue::Boolean(None),
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Date32 => ScalarValue::Date32(None),
            DataType::Null => ScalarValue::Null,
        }
    }

    /// Materialize the value as an Arrow array repeating it `len` times.
    pub fn to_array(&self, len: usize) -> ArrayRef {
        match self {
            ScalarValue::Boolean(v) => Arc::new(BooleanArray::from(vec![*v; len])),
            ScalarValue::Int64(v) => Arc::new(Int64Array::from(vec![*v; len])),
            ScalarValue::Float64(v) => Arc::new(Float64Array::from(vec![*v; len])),
            ScalarValue::Utf8(v) => Arc::new(StringArray::from(vec![v.as_deref(); len])),
            ScalarValue::Date32(v) => Arc::new(Date32Array::from(vec![*v; len])),
            ScalarValue::Null => new_null_array(&ArrowType::Null, len),
        }
    }

    /// Read element `i` of a (normalized) Arrow array as a scalar.
    pub fn from_array(array: &dyn Array, i: usize) -> Result<ScalarValue> {
        let dt = DataType::from_arrow(array.data_type()).ok_or_else(|| {
            QuarryError::Execution(format!("unsupported array type {}", array.data_type()))
        })?;
        if array.is_null(i) {
            return Ok(ScalarValue::null_of(dt));
        }
        let any = array.as_any();
        Ok(match array.data_type() {
            ArrowType::Boolean => ScalarValue::Boolean(Some(
                any.downcast_ref::<BooleanArray>()
                    .map(|a| a.value(i))
                    .unwrap_or_default(),
            )),
            ArrowType::Int64 => {
                ScalarValue::Int64(any.downcast_ref::<Int64Array>().map(|a| a.value(i)))
            }
            ArrowType::Float64 => {
                ScalarValue::Float64(any.downcast_ref::<Float64Array>().map(|a| a.value(i)))
            }
            ArrowType::Utf8 => ScalarValue::Utf8(
                any.downcast_ref::<StringArray>()
                    .map(|a| a.value(i).to_string()),
            ),
            ArrowType::Date32 => {
                ScalarValue::Date32(any.downcast_ref::<Date32Array>().map(|a| a.value(i)))
            }
            ArrowType::Null => ScalarValue::Null,
            other => return exec_err(format!("array of type {other} is not normalized")),
        })
    }

    /// Numeric view used by statistics and selectivity estimation.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Int64(Some(v)) => Some(*v as f64),
            ScalarValue::Float64(Some(v)) => Some(*v),
            ScalarValue::Date32(Some(v)) => Some(*v as f64),
            _ => None,
        }
    }

    /// SQL ordering between two non-null values of compatible type.
    pub fn partial_cmp_sql(&self, other: &ScalarValue) -> Option<Ordering> {
        use ScalarValue::*;
        match (self, other) {
            (Boolean(Some(a)), Boolean(Some(b))) => a.partial_cmp(b),
            (Int64(Some(a)), Int64(Some(b))) => a.partial_cmp(b),
            (Float64(Some(a)), Float64(Some(b))) => a.partial_cmp(b),
            (Int64(Some(a)), Float64(Some(b))) => (*a as f64).partial_cmp(b),
            (Float64(Some(a)), Int64(Some(b))) => a.partial_cmp(&(*b as f64)),
            (Utf8(Some(a)), Utf8(Some(b))) => a.partial_cmp(b),
            (Date32(Some(a)), Date32(Some(b))) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl fmt::Display for ScalarValue {
    /// Renders the value as a SQL literal (used by EXPLAIN).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            return f.write_str("NULL");
        }
        match self {
            ScalarValue::Boolean(Some(b)) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            ScalarValue::Int64(Some(v)) => write!(f, "{v}"),
            ScalarValue::Float64(Some(v)) => write!(f, "{}", format_float(*v)),
            ScalarValue::Utf8(Some(s)) => write!(f, "'{}'", s.replace('\'', "''")),
            ScalarValue::Date32(Some(d)) => write!(f, "DATE '{}'", format_date(*d)),
            _ => f.write_str("NULL"),
        }
    }
}

/// Format a float so that integral values still show a decimal point.
pub fn format_float(v: f64) -> String {
    if v.is_finite() && v == v.trunc() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

// ---------------------------------------------------------------------------
// Calendar arithmetic (proleptic Gregorian, days since 1970-01-01).
// Algorithms from Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms".
// ---------------------------------------------------------------------------

/// Days since the Unix epoch for a civil date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = m as i64;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Civil date `(year, month, day)` for days since the Unix epoch.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(y) => 29,
        _ => 28,
    }
}

/// Parse `YYYY-MM-DD` into days since the epoch.
pub fn parse_date(s: &str) -> Option<i32> {
    let mut parts = s.trim().splitn(3, '-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    i32::try_from(days_from_civil(y, m, d)).ok()
}

/// Format days since the epoch as `YYYY-MM-DD`.
pub fn format_date(days: i32) -> String {
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Add a number of calendar months, clamping the day to the target month's end
/// (`2024-01-31 + 1 month = 2024-02-29`), then a number of days.
pub fn add_months_days(days: i32, months: i32, extra_days: i32) -> i32 {
    let (y, m, d) = civil_from_days(days as i64);
    let total = y * 12 + (m as i64 - 1) + months as i64;
    let ny = total.div_euclid(12);
    let nm = (total.rem_euclid(12) + 1) as u32;
    let nd = d.min(days_in_month(ny, nm));
    (days_from_civil(ny, nm, nd) + extra_days as i64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for z in [-800_000_i64, -1, 0, 1, 10_000, 19_000, 2_000_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
    }

    #[test]
    fn parse_and_format_dates() {
        let d = parse_date("1995-03-15").unwrap();
        assert_eq!(format_date(d), "1995-03-15");
        assert_eq!(parse_date("1995-02-29"), None);
        assert!(parse_date("1996-02-29").is_some());
        assert_eq!(parse_date("not a date"), None);
        assert_eq!(parse_date("1995-13-01"), None);
    }

    #[test]
    fn month_arithmetic_clamps() {
        let jan31 = parse_date("2024-01-31").unwrap();
        assert_eq!(format_date(add_months_days(jan31, 1, 0)), "2024-02-29");
        assert_eq!(format_date(add_months_days(jan31, 12, 0)), "2025-01-31");
        assert_eq!(format_date(add_months_days(jan31, -2, 0)), "2023-11-30");
        assert_eq!(format_date(add_months_days(jan31, 0, 1)), "2024-02-01");
    }

    #[test]
    fn scalar_display_and_arrays() {
        assert_eq!(
            ScalarValue::Utf8(Some("it's".into())).to_string(),
            "'it''s'"
        );
        assert_eq!(ScalarValue::Float64(Some(2.0)).to_string(), "2.0");
        assert_eq!(ScalarValue::Int64(None).to_string(), "NULL");
        let arr = ScalarValue::Date32(Some(5)).to_array(3);
        assert_eq!(arr.len(), 3);
        assert_eq!(
            ScalarValue::from_array(arr.as_ref(), 2).unwrap(),
            ScalarValue::Date32(Some(5))
        );
        let nulls = ScalarValue::Utf8(None).to_array(2);
        assert!(ScalarValue::from_array(nulls.as_ref(), 0)
            .unwrap()
            .is_null());
    }

    #[test]
    fn arrow_type_normalization() {
        assert_eq!(
            DataType::from_arrow(&ArrowType::Int32),
            Some(DataType::Int64)
        );
        assert_eq!(
            DataType::from_arrow(&ArrowType::Float32),
            Some(DataType::Float64)
        );
        assert_eq!(
            DataType::from_arrow(&ArrowType::LargeUtf8),
            Some(DataType::Utf8)
        );
        assert_eq!(DataType::from_arrow(&ArrowType::UInt64), None);
        assert_eq!(
            ScalarValue::Int64(Some(1)).partial_cmp_sql(&ScalarValue::Float64(Some(1.5))),
            Some(Ordering::Less)
        );
    }
}
