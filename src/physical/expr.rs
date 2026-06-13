//! Vectorized expression evaluation over Arrow record batches.
//!
//! A [`PhysicalExpr`] is a bound [`Expr`] whose column references have been
//! resolved to positions in the input batch. Evaluation returns a
//! [`ColumnarValue`]: either a full column or a single scalar that is
//! broadcast lazily, so `col < 10` compares an array against a scalar without
//! materializing a column of tens.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Date32Array, Float64Array, Int64Array, Scalar,
    StringArray,
};
use arrow::compute::cast;
use arrow::compute::kernels::{boolean, cmp, numeric, zip::zip};
use arrow::datatypes::{Date32Type, Float64Type, Int64Type};
use arrow::record_batch::RecordBatch;

use crate::error::{exec_err, plan_err, QuarryError, Result};
use crate::plan::expr::{BinaryOp, ColId, Expr, ScalarFunc};
use crate::plan::logical::ColumnDef;
use crate::types::{add_months_days, civil_from_days, DataType, ScalarValue};

/// Result of evaluating an expression against a batch.
#[derive(Debug, Clone)]
pub enum ColumnarValue {
    Array(ArrayRef),
    Scalar(ScalarValue),
}

impl ColumnarValue {
    /// Materialize as an array of `len` rows.
    pub fn into_array(self, len: usize) -> ArrayRef {
        match self {
            ColumnarValue::Array(a) => a,
            ColumnarValue::Scalar(s) => s.to_array(len),
        }
    }
}

/// An executable expression; columns are positions in the input batch.
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Column(usize),
    Literal(ScalarValue),
    Binary {
        left: Box<PhysicalExpr>,
        op: BinaryOp,
        right: Box<PhysicalExpr>,
    },
    Not(Box<PhysicalExpr>),
    Negative(Box<PhysicalExpr>),
    IsNull(Box<PhysicalExpr>),
    IsNotNull(Box<PhysicalExpr>),
    Like {
        expr: Box<PhysicalExpr>,
        pattern: Box<PhysicalExpr>,
        negated: bool,
    },
    InList {
        expr: Box<PhysicalExpr>,
        list: Vec<PhysicalExpr>,
        negated: bool,
    },
    Case {
        branches: Vec<(PhysicalExpr, PhysicalExpr)>,
        else_expr: Option<Box<PhysicalExpr>>,
        data_type: DataType,
    },
    Cast {
        expr: Box<PhysicalExpr>,
        to: DataType,
    },
    Function {
        func: ScalarFunc,
        args: Vec<PhysicalExpr>,
    },
}

/// A selection mask where NULL counts as false (SQL WHERE semantics).
pub fn null_as_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        mask.clone()
    } else {
        arrow::compute::prep_null_mask_filter(mask)
    }
}

/// Resolve column ids against the input schema.
pub fn create_physical_expr(e: &Expr, input: &[ColumnDef]) -> Result<PhysicalExpr> {
    let rec = |x: &Expr| create_physical_expr(x, input).map(Box::new);
    Ok(match e {
        Expr::Column(c) => PhysicalExpr::Column(column_index(c.id, input).ok_or_else(|| {
            QuarryError::Plan(format!("column {} (#{}) not found in input", c.name, c.id))
        })?),
        Expr::Literal(v) => PhysicalExpr::Literal(v.clone()),
        Expr::Binary { left, op, right } => PhysicalExpr::Binary {
            left: rec(left)?,
            op: *op,
            right: rec(right)?,
        },
        Expr::Not(x) => PhysicalExpr::Not(rec(x)?),
        Expr::Negative(x) => PhysicalExpr::Negative(rec(x)?),
        Expr::IsNull(x) => PhysicalExpr::IsNull(rec(x)?),
        Expr::IsNotNull(x) => PhysicalExpr::IsNotNull(rec(x)?),
        Expr::Like {
            expr,
            pattern,
            negated,
        } => PhysicalExpr::Like {
            expr: rec(expr)?,
            pattern: rec(pattern)?,
            negated: *negated,
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => PhysicalExpr::InList {
            expr: rec(expr)?,
            list: list
                .iter()
                .map(|x| create_physical_expr(x, input))
                .collect::<Result<_>>()?,
            negated: *negated,
        },
        Expr::Case {
            branches,
            else_expr,
            data_type,
        } => PhysicalExpr::Case {
            branches: branches
                .iter()
                .map(|(w, t)| {
                    Ok((
                        create_physical_expr(w, input)?,
                        create_physical_expr(t, input)?,
                    ))
                })
                .collect::<Result<_>>()?,
            else_expr: else_expr.as_deref().map(rec).transpose()?,
            data_type: *data_type,
        },
        Expr::Cast { expr, to } => PhysicalExpr::Cast {
            expr: rec(expr)?,
            to: *to,
        },
        Expr::Function { func, args, .. } => PhysicalExpr::Function {
            func: *func,
            args: args
                .iter()
                .map(|x| create_physical_expr(x, input))
                .collect::<Result<_>>()?,
        },
    })
}

pub fn column_index(id: ColId, input: &[ColumnDef]) -> Option<usize> {
    input.iter().position(|c| c.id == id)
}

/// Evaluate a constant expression (no column references) to a scalar.
pub fn evaluate_constant(e: &Expr) -> Result<ScalarValue> {
    if !e.is_constant() {
        return plan_err(format!("expression {e} is not constant"));
    }
    let pe = create_physical_expr(e, &[])?;
    let batch = RecordBatch::try_new_with_options(
        Arc::new(arrow::datatypes::Schema::empty()),
        vec![],
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    let arr = pe.evaluate(&batch)?.into_array(1);
    let v = ScalarValue::from_array(arr.as_ref(), 0)?;
    // Keep the declared type for typed NULLs produced from untyped inputs.
    if v.is_null() && v.data_type() != e.data_type() {
        return Ok(ScalarValue::null_of(e.data_type()));
    }
    Ok(v)
}

fn scalar_datum(s: &ScalarValue) -> Scalar<ArrayRef> {
    Scalar::new(s.to_array(1))
}

/// Apply a two-argument Datum kernel, keeping scalar ⊕ scalar as a scalar.
/// A two-argument Arrow kernel taking scalars or arrays.
type DatumKernel = dyn Fn(
    &dyn arrow::array::Datum,
    &dyn arrow::array::Datum,
) -> std::result::Result<ArrayRef, arrow::error::ArrowError>;

fn apply_datum(l: &ColumnarValue, r: &ColumnarValue, f: &DatumKernel) -> Result<ColumnarValue> {
    Ok(match (l, r) {
        (ColumnarValue::Array(a), ColumnarValue::Array(b)) => ColumnarValue::Array(f(a, b)?),
        (ColumnarValue::Array(a), ColumnarValue::Scalar(s)) => {
            ColumnarValue::Array(f(a, &scalar_datum(s))?)
        }
        (ColumnarValue::Scalar(s), ColumnarValue::Array(b)) => {
            ColumnarValue::Array(f(&scalar_datum(s), b)?)
        }
        (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => {
            let out = f(&scalar_datum(a), &scalar_datum(b))?;
            ColumnarValue::Scalar(ScalarValue::from_array(out.as_ref(), 0)?)
        }
    })
}

fn bool_result(
    f: fn(
        &dyn arrow::array::Datum,
        &dyn arrow::array::Datum,
    ) -> std::result::Result<BooleanArray, arrow::error::ArrowError>,
) -> impl Fn(
    &dyn arrow::array::Datum,
    &dyn arrow::array::Datum,
) -> std::result::Result<ArrayRef, arrow::error::ArrowError> {
    move |a, b| f(a, b).map(|x| Arc::new(x) as ArrayRef)
}

/// Element-wise map over one or two arrays where scalars are broadcast.
fn both_arrays(l: ColumnarValue, r: ColumnarValue, n: usize) -> (ArrayRef, ArrayRef, bool) {
    let scalar = matches!(
        (&l, &r),
        (ColumnarValue::Scalar(_), ColumnarValue::Scalar(_))
    );
    let len = if scalar { 1 } else { n };
    (l.into_array(len), r.into_array(len), scalar)
}

fn wrap(arr: ArrayRef, scalar: bool) -> Result<ColumnarValue> {
    if scalar {
        Ok(ColumnarValue::Scalar(ScalarValue::from_array(
            arr.as_ref(),
            0,
        )?))
    } else {
        Ok(ColumnarValue::Array(arr))
    }
}

impl PhysicalExpr {
    /// Evaluate against `batch`.
    pub fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let n = batch.num_rows();
        match self {
            PhysicalExpr::Column(i) => Ok(ColumnarValue::Array(Arc::clone(batch.column(*i)))),
            PhysicalExpr::Literal(v) => Ok(ColumnarValue::Scalar(v.clone())),
            PhysicalExpr::Binary { left, op, right } => {
                let l = left.evaluate(batch)?;
                let r = right.evaluate(batch)?;
                eval_binary(l, *op, r, n)
            }
            PhysicalExpr::Not(x) => map_unary(x.evaluate(batch)?, |a| {
                Ok(Arc::new(boolean::not(a.as_boolean())?) as ArrayRef)
            }),
            PhysicalExpr::Negative(x) => {
                map_unary(x.evaluate(batch)?, |a| Ok(numeric::neg(a.as_ref())?))
            }
            PhysicalExpr::IsNull(x) => map_unary(x.evaluate(batch)?, |a| {
                Ok(Arc::new(boolean::is_null(a.as_ref())?) as ArrayRef)
            }),
            PhysicalExpr::IsNotNull(x) => map_unary(x.evaluate(batch)?, |a| {
                Ok(Arc::new(boolean::is_not_null(a.as_ref())?) as ArrayRef)
            }),
            PhysicalExpr::Like {
                expr,
                pattern,
                negated,
            } => {
                let l = expr.evaluate(batch)?;
                let r = pattern.evaluate(batch)?;
                let f = if *negated {
                    bool_result(arrow::compute::kernels::comparison::nlike)
                } else {
                    bool_result(arrow::compute::kernels::comparison::like)
                };
                apply_datum(&l, &r, &f)
            }
            PhysicalExpr::InList {
                expr,
                list,
                negated,
            } => {
                let x = canonical_zero_value(expr.evaluate(batch)?);
                let mut acc: Option<ColumnarValue> = None;
                for item in list {
                    let item = canonical_zero_value(item.evaluate(batch)?);
                    let eq = apply_datum(&x, &item, &bool_result(cmp::eq))?;
                    acc = Some(match acc {
                        None => eq,
                        Some(prev) => eval_binary(prev, BinaryOp::Or, eq, n)?,
                    });
                }
                let acc = acc.ok_or_else(|| QuarryError::Plan("empty IN list".into()))?;
                if *negated {
                    map_unary(acc, |a| {
                        Ok(Arc::new(boolean::not(a.as_boolean())?) as ArrayRef)
                    })
                } else {
                    Ok(acc)
                }
            }
            PhysicalExpr::Case {
                branches,
                else_expr,
                data_type,
            } => {
                let mut acc = match else_expr {
                    Some(e) => e.evaluate(batch)?.into_array(n),
                    None => ScalarValue::null_of(*data_type).to_array(n),
                };
                for (cond, then) in branches.iter().rev() {
                    let c = cond.evaluate(batch)?.into_array(n);
                    let mask = null_as_false(c.as_boolean());
                    let t = then.evaluate(batch)?.into_array(n);
                    acc = zip(&mask, &t, &acc)?;
                }
                Ok(ColumnarValue::Array(acc))
            }
            PhysicalExpr::Cast { expr, to } => {
                let target = to.to_arrow();
                map_unary(expr.evaluate(batch)?, |a| Ok(cast(a.as_ref(), &target)?))
            }
            PhysicalExpr::Function { func, args } => {
                let vals = args
                    .iter()
                    .map(|a| a.evaluate(batch))
                    .collect::<Result<Vec<_>>>()?;
                eval_function(*func, vals, n)
            }
        }
    }
}

/// Replace every `-0.0` in a DOUBLE column with `0.0`; other columns pass
/// through untouched.
///
/// SQL (like IEEE 754) treats the two zeros as equal, but Arrow's comparison
/// kernels and its row format order floats totally, putting `-0.0` strictly
/// below `0.0`. Comparisons, IN lists, join keys, group keys, DISTINCT values
/// and sort keys therefore go through this first. Adding `0.0` maps `-0.0` to
/// `0.0` and leaves every other value (NaN included) unchanged.
pub fn canonical_zero(a: ArrayRef) -> ArrayRef {
    if a.data_type() != &arrow::datatypes::DataType::Float64 {
        return a;
    }
    let f = a.as_primitive::<Float64Type>();
    if !f.values().iter().any(|x| *x == 0.0 && x.is_sign_negative()) {
        return a;
    }
    Arc::new(f.unary::<_, Float64Type>(|x| x + 0.0))
}

/// [`canonical_zero`] for a column or a scalar.
pub fn canonical_zero_value(v: ColumnarValue) -> ColumnarValue {
    match v {
        ColumnarValue::Array(a) => ColumnarValue::Array(canonical_zero(a)),
        ColumnarValue::Scalar(ScalarValue::Float64(Some(x))) => {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(x + 0.0)))
        }
        s => s,
    }
}

fn map_unary(v: ColumnarValue, f: impl Fn(ArrayRef) -> Result<ArrayRef>) -> Result<ColumnarValue> {
    match v {
        ColumnarValue::Array(a) => Ok(ColumnarValue::Array(f(a)?)),
        ColumnarValue::Scalar(s) => wrap(f(s.to_array(1))?, true),
    }
}

fn eval_binary(
    l: ColumnarValue,
    op: BinaryOp,
    r: ColumnarValue,
    n: usize,
) -> Result<ColumnarValue> {
    let (l, r) = if op.is_comparison() {
        (canonical_zero_value(l), canonical_zero_value(r))
    } else {
        (l, r)
    };
    match op {
        BinaryOp::Eq => apply_datum(&l, &r, &bool_result(cmp::eq)),
        BinaryOp::NotEq => apply_datum(&l, &r, &bool_result(cmp::neq)),
        BinaryOp::Lt => apply_datum(&l, &r, &bool_result(cmp::lt)),
        BinaryOp::LtEq => apply_datum(&l, &r, &bool_result(cmp::lt_eq)),
        BinaryOp::Gt => apply_datum(&l, &r, &bool_result(cmp::gt)),
        BinaryOp::GtEq => apply_datum(&l, &r, &bool_result(cmp::gt_eq)),
        BinaryOp::Plus => apply_datum(&l, &r, &numeric::add),
        BinaryOp::Minus => apply_datum(&l, &r, &numeric::sub),
        BinaryOp::Multiply => apply_datum(&l, &r, &numeric::mul),
        BinaryOp::And | BinaryOp::Or => {
            let (a, b, scalar) = both_arrays(l, r, n);
            let (a, b) = (as_bool(&a)?, as_bool(&b)?);
            let out = if op == BinaryOp::And {
                boolean::and_kleene(&a, &b)?
            } else {
                boolean::or_kleene(&a, &b)?
            };
            wrap(Arc::new(out), scalar)
        }
        BinaryOp::Divide | BinaryOp::Modulo => {
            let (a, b, scalar) = both_arrays(l, r, n);
            wrap(divide(&a, &b, op == BinaryOp::Modulo)?, scalar)
        }
        BinaryOp::Concat => {
            let (a, b, scalar) = both_arrays(l, r, n);
            let (a, b) = (a.as_string::<i32>(), b.as_string::<i32>());
            let out: StringArray = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| match (x, y) {
                    (Some(x), Some(y)) => Some(format!("{x}{y}")),
                    _ => None,
                })
                .collect();
            wrap(Arc::new(out), scalar)
        }
    }
}

/// Boolean view of an array; an all-NULL untyped array becomes typed NULLs.
fn as_bool(a: &ArrayRef) -> Result<BooleanArray> {
    if a.data_type() == &arrow::datatypes::DataType::Null {
        return Ok(BooleanArray::from(vec![None; a.len()]));
    }
    Ok(a.as_boolean().clone())
}

/// Division and modulo with SQL semantics: integer division truncates and a
/// zero divisor yields NULL (as in SQLite) rather than an error.
fn divide(a: &ArrayRef, b: &ArrayRef, modulo: bool) -> Result<ArrayRef> {
    match a.data_type() {
        arrow::datatypes::DataType::Int64 => {
            let (x, y) = (a.as_primitive::<Int64Type>(), b.as_primitive::<Int64Type>());
            let out: Int64Array = x
                .iter()
                .zip(y.iter())
                .map(|(p, q)| match (p, q) {
                    (Some(p), Some(q)) if q != 0 => {
                        if modulo {
                            p.checked_rem(q)
                        } else {
                            p.checked_div(q)
                        }
                    }
                    _ => None,
                })
                .collect();
            Ok(Arc::new(out))
        }
        arrow::datatypes::DataType::Float64 => {
            let (x, y) = (
                a.as_primitive::<Float64Type>(),
                b.as_primitive::<Float64Type>(),
            );
            let out: Float64Array = x
                .iter()
                .zip(y.iter())
                .map(|(p, q)| match (p, q) {
                    (Some(p), Some(q)) if q != 0.0 => Some(if modulo { p % q } else { p / q }),
                    _ => None,
                })
                .collect();
            Ok(Arc::new(out))
        }
        arrow::datatypes::DataType::Null => Ok(Arc::clone(a)),
        other => exec_err(format!("cannot divide values of type {other}")),
    }
}

fn eval_function(func: ScalarFunc, args: Vec<ColumnarValue>, n: usize) -> Result<ColumnarValue> {
    let scalar = args.iter().all(|a| matches!(a, ColumnarValue::Scalar(_)));
    let len = if scalar { 1 } else { n };
    let arrays: Vec<ArrayRef> = args.into_iter().map(|a| a.into_array(len)).collect();
    let out: ArrayRef = match func {
        ScalarFunc::Abs => match arrays[0].data_type() {
            arrow::datatypes::DataType::Int64 => {
                let a = arrays[0].as_primitive::<Int64Type>();
                let mut overflow = false;
                let out: Int64Array = a
                    .iter()
                    .map(|v| {
                        v.and_then(|v| {
                            let r = v.checked_abs();
                            overflow |= r.is_none();
                            r
                        })
                    })
                    .collect();
                if overflow {
                    return exec_err("integer overflow in abs()");
                }
                Arc::new(out)
            }
            _ => {
                let a = arrays[0].as_primitive::<Float64Type>();
                Arc::new(a.unary::<_, Float64Type>(f64::abs))
            }
        },
        ScalarFunc::Lower | ScalarFunc::Upper => {
            let a = arrays[0].as_string::<i32>();
            let out: StringArray = a
                .iter()
                .map(|v| {
                    v.map(|s| {
                        if func == ScalarFunc::Lower {
                            s.to_lowercase()
                        } else {
                            s.to_uppercase()
                        }
                    })
                })
                .collect();
            Arc::new(out)
        }
        ScalarFunc::Length => {
            let a = arrays[0].as_string::<i32>();
            let out: Int64Array = a
                .iter()
                .map(|v| v.map(|s| s.chars().count() as i64))
                .collect();
            Arc::new(out)
        }
        ScalarFunc::Substr => {
            let s = arrays[0].as_string::<i32>();
            let start = arrays[1].as_primitive::<Int64Type>();
            let count = arrays.get(2).map(|a| a.as_primitive::<Int64Type>());
            let out: StringArray = (0..s.len())
                .map(|i| {
                    if s.is_null(i) || start.is_null(i) || count.is_some_and(|c| c.is_null(i)) {
                        return None;
                    }
                    Some(substr(
                        s.value(i),
                        start.value(i),
                        count.map(|c| c.value(i)),
                    ))
                })
                .collect();
            Arc::new(out)
        }
        ScalarFunc::Coalesce => {
            let mut iter = arrays.into_iter().rev();
            let mut acc = iter
                .next()
                .ok_or_else(|| QuarryError::Plan("coalesce() without arguments".into()))?;
            for a in iter {
                let mask = boolean::is_not_null(a.as_ref())?;
                acc = zip(&mask, &a, &acc)?;
            }
            acc
        }
        ScalarFunc::Round => {
            let x = arrays[0].as_primitive::<Float64Type>();
            let digits = arrays.get(1).map(|a| a.as_primitive::<Int64Type>());
            let out: Float64Array = (0..x.len())
                .map(|i| {
                    if x.is_null(i) || digits.is_some_and(|d| d.is_null(i)) {
                        return None;
                    }
                    let d = digits.map(|d| d.value(i)).unwrap_or(0).clamp(-15, 15) as i32;
                    let scale = 10f64.powi(d);
                    Some((x.value(i) * scale).round() / scale)
                })
                .collect();
            Arc::new(out)
        }
        ScalarFunc::ExtractYear | ScalarFunc::ExtractMonth | ScalarFunc::ExtractDay => {
            let a = arrays[0].as_primitive::<Date32Type>();
            let out: Int64Array = a
                .iter()
                .map(|v| {
                    v.map(|d| {
                        let (y, m, dd) = civil_from_days(d as i64);
                        match func {
                            ScalarFunc::ExtractYear => y,
                            ScalarFunc::ExtractMonth => m as i64,
                            _ => dd as i64,
                        }
                    })
                })
                .collect();
            Arc::new(out)
        }
        ScalarFunc::DateAdd { months, days } => {
            let a = arrays[0].as_primitive::<Date32Type>();
            let out: Date32Array = a.unary(|d| add_months_days(d, months, days));
            Arc::new(out)
        }
    };
    wrap(out, scalar)
}

/// SQL `substr` (1-based). A start before 1 consumes part of the length, as in
/// PostgreSQL: `substr('hello', 0, 2) = 'h'`.
fn substr(s: &str, start: i64, count: Option<i64>) -> String {
    let begin = start.max(1);
    let end = match count {
        Some(c) if c < 0 => return String::new(),
        Some(c) => start.saturating_add(c),
        None => i64::MAX,
    };
    if end <= begin {
        return String::new();
    }
    s.chars()
        .skip((begin - 1) as usize)
        .take(usize::try_from(end - begin).unwrap_or(usize::MAX))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::expr::ColumnRef;
    use arrow::datatypes::{DataType as ArrowType, Field, Schema};

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int64, true),
            Field::new("s", ArrowType::Utf8, true),
            Field::new("f", ArrowType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(7), Some(-3), None, Some(0)])),
                Arc::new(StringArray::from(vec![
                    Some("hello"),
                    Some("World"),
                    None,
                    Some(""),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(1.25),
                    Some(-2.5),
                    Some(0.0),
                    None,
                ])),
            ],
        )
        .unwrap()
    }

    fn defs() -> Vec<ColumnDef> {
        [
            ("a", DataType::Int64),
            ("s", DataType::Utf8),
            ("f", DataType::Float64),
        ]
        .iter()
        .enumerate()
        .map(|(i, (n, t))| ColumnDef {
            id: i + 1,
            name: n.to_string(),
            qualifier: None,
            data_type: *t,
            nullable: true,
        })
        .collect()
    }

    fn col(i: usize) -> Expr {
        let d = &defs()[i];
        Expr::Column(ColumnRef {
            id: d.id,
            name: d.name.clone(),
            data_type: d.data_type,
            nullable: true,
        })
    }

    fn int(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(v)))
    }

    fn eval(e: &Expr) -> Vec<ScalarValue> {
        let b = batch();
        let arr = create_physical_expr(e, &defs())
            .unwrap()
            .evaluate(&b)
            .unwrap()
            .into_array(b.num_rows());
        (0..arr.len())
            .map(|i| ScalarValue::from_array(arr.as_ref(), i).unwrap())
            .collect()
    }

    fn ints(v: &[Option<i64>]) -> Vec<ScalarValue> {
        v.iter().map(|x| ScalarValue::Int64(*x)).collect()
    }

    fn bools(v: &[Option<bool>]) -> Vec<ScalarValue> {
        v.iter().map(|x| ScalarValue::Boolean(*x)).collect()
    }

    #[test]
    fn division_by_zero_is_null() {
        let e = Expr::binary(int(10), BinaryOp::Divide, col(0));
        assert_eq!(eval(&e), ints(&[Some(1), Some(-3), None, None]));
        let e = Expr::binary(col(0), BinaryOp::Modulo, int(4));
        assert_eq!(eval(&e), ints(&[Some(3), Some(-3), None, Some(0)]));
    }

    #[test]
    fn three_valued_logic() {
        let gt = Expr::binary(col(0), BinaryOp::Gt, int(0));
        let or = Expr::binary(
            gt.clone(),
            BinaryOp::Or,
            Expr::Literal(ScalarValue::Boolean(Some(true))),
        );
        assert_eq!(eval(&or), bools(&[Some(true); 4]));
        let and = Expr::binary(
            gt,
            BinaryOp::And,
            Expr::Literal(ScalarValue::Boolean(Some(false))),
        );
        assert_eq!(eval(&and), bools(&[Some(false); 4]));
        let inl = Expr::InList {
            expr: Box::new(col(0)),
            list: vec![int(7), Expr::Literal(ScalarValue::Int64(None))],
            negated: true,
        };
        assert_eq!(eval(&inl), bools(&[Some(false), None, None, None]));
    }

    #[test]
    fn case_and_coalesce() {
        let e = Expr::Case {
            branches: vec![(Expr::binary(col(0), BinaryOp::Gt, int(0)), int(1))],
            else_expr: Some(Box::new(int(0))),
            data_type: DataType::Int64,
        };
        assert_eq!(eval(&e), ints(&[Some(1), Some(0), Some(0), Some(0)]));
        let e = Expr::Function {
            func: ScalarFunc::Coalesce,
            args: vec![col(0), int(42)],
            data_type: DataType::Int64,
        };
        assert_eq!(eval(&e), ints(&[Some(7), Some(-3), Some(42), Some(0)]));
    }

    #[test]
    fn string_functions() {
        let f = |func, extra: Vec<Expr>| {
            let mut args = vec![col(1)];
            args.extend(extra);
            eval(&Expr::Function {
                func,
                args,
                data_type: DataType::Utf8,
            })
        };
        let s = |v: &[Option<&str>]| {
            v.iter()
                .map(|x| ScalarValue::Utf8(x.map(String::from)))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            f(ScalarFunc::Upper, vec![]),
            s(&[Some("HELLO"), Some("WORLD"), None, Some("")])
        );
        assert_eq!(
            f(ScalarFunc::Substr, vec![int(2), int(3)]),
            s(&[Some("ell"), Some("orl"), None, Some("")])
        );
        assert_eq!(substr("hello", 0, Some(2)), "h");
        assert_eq!(substr("hello", 4, None), "lo");
        assert_eq!(substr("hello", 2, Some(-1)), "");
        let concat = Expr::binary(
            col(1),
            BinaryOp::Concat,
            Expr::Literal(ScalarValue::Utf8(Some("!".into()))),
        );
        assert_eq!(
            eval(&concat),
            s(&[Some("hello!"), Some("World!"), None, Some("!")])
        );
        let like = Expr::Like {
            expr: Box::new(col(1)),
            pattern: Box::new(Expr::Literal(ScalarValue::Utf8(Some("%o%".into())))),
            negated: false,
        };
        assert_eq!(
            eval(&like),
            bools(&[Some(true), Some(true), None, Some(false)])
        );
    }

    #[test]
    fn constants_fold_to_scalars() {
        let e = Expr::binary(int(2), BinaryOp::Multiply, int(21));
        assert_eq!(evaluate_constant(&e).unwrap(), ScalarValue::Int64(Some(42)));
        let date = Expr::Function {
            func: ScalarFunc::DateAdd {
                months: 12,
                days: 0,
            },
            args: vec![Expr::Literal(ScalarValue::Date32(
                crate::types::parse_date("1994-01-01"),
            ))],
            data_type: DataType::Date32,
        };
        assert_eq!(
            evaluate_constant(&date).unwrap(),
            ScalarValue::Date32(crate::types::parse_date("1995-01-01"))
        );
        let overflow = Expr::binary(int(i64::MAX), BinaryOp::Plus, int(1));
        assert!(evaluate_constant(&overflow).is_err());
        assert!(evaluate_constant(&col(0)).is_err());
        let round = Expr::Function {
            func: ScalarFunc::Round,
            args: vec![Expr::Literal(ScalarValue::Float64(Some(1.2345))), int(2)],
            data_type: DataType::Float64,
        };
        assert_eq!(
            evaluate_constant(&round).unwrap(),
            ScalarValue::Float64(Some(1.23))
        );
    }

    #[test]
    fn canonical_zero_folds_negative_zero_only() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(0.0),
            None,
            Some(-1.5),
            Some(f64::NAN),
        ]));
        let out = canonical_zero(a);
        let out = out.as_primitive::<Float64Type>();
        assert!(out.value(0) == 0.0 && out.value(0).is_sign_positive());
        assert!(out.value(1).is_sign_positive());
        assert!(out.is_null(2));
        assert_eq!(out.value(3), -1.5);
        assert!(out.value(4).is_nan());
        // Arrays without -0.0 and non-DOUBLE arrays come back as the same array.
        let plain: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 0.0]));
        assert!(Arc::ptr_eq(&canonical_zero(Arc::clone(&plain)), &plain));
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![0, -1]));
        assert!(Arc::ptr_eq(&canonical_zero(Arc::clone(&ints)), &ints));
        match canonical_zero_value(ColumnarValue::Scalar(ScalarValue::Float64(Some(-0.0)))) {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(x))) => {
                assert!(x == 0.0 && x.is_sign_positive())
            }
            other => panic!("{other:?}"),
        }
        // -0.0 = 0.0 is true and -0.0 < 0.0 is false, as in SQL.
        let neg = Expr::Literal(ScalarValue::Float64(Some(-0.0)));
        let pos = Expr::Literal(ScalarValue::Float64(Some(0.0)));
        let eq = Expr::binary(neg.clone(), BinaryOp::Eq, pos.clone());
        let lt = Expr::binary(neg, BinaryOp::Lt, pos);
        assert_eq!(
            evaluate_constant(&eq).unwrap(),
            ScalarValue::Boolean(Some(true))
        );
        assert_eq!(
            evaluate_constant(&lt).unwrap(),
            ScalarValue::Boolean(Some(false))
        );
    }
}
