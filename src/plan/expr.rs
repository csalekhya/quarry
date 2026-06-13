//! Bound, typed expressions used by the logical plan and the optimizer.
//!
//! Columns are referenced by a plan-wide unique [`ColId`] rather than by
//! position or name. That makes optimizer rewrites (moving a predicate across
//! a projection, reordering joins) a matter of moving expressions around: no
//! index arithmetic, no ambiguity between `n1.n_name` and `n2.n_name`.

use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::error::Result;
pub use crate::sql::ast::BinaryOp;
use crate::types::{DataType, ScalarValue};

/// Identifier of a column, unique within one planned query.
pub type ColId = usize;

/// A reference to a column produced by some plan node.
///
/// Equality and hashing use only `id`; `name` is for display.
#[derive(Debug, Clone)]
pub struct ColumnRef {
    pub id: ColId,
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl PartialEq for ColumnRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ColumnRef {}

impl Hash for ColumnRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Built-in scalar functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarFunc {
    Abs,
    Lower,
    Upper,
    Length,
    /// `substr(s, start [, len])`, 1-based like SQL.
    Substr,
    Coalesce,
    /// `round(x [, digits])`
    Round,
    ExtractYear,
    ExtractMonth,
    ExtractDay,
    /// `date + INTERVAL`: adds calendar months, then days.
    DateAdd {
        months: i32,
        days: i32,
    },
}

impl fmt::Display for ScalarFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ScalarFunc::Abs => "abs",
            ScalarFunc::Lower => "lower",
            ScalarFunc::Upper => "upper",
            ScalarFunc::Length => "length",
            ScalarFunc::Substr => "substr",
            ScalarFunc::Coalesce => "coalesce",
            ScalarFunc::Round => "round",
            ScalarFunc::ExtractYear => "extract_year",
            ScalarFunc::ExtractMonth => "extract_month",
            ScalarFunc::ExtractDay => "extract_day",
            ScalarFunc::DateAdd { months, days } => {
                return write!(f, "date_add[{months}mo,{days}d]");
            }
        };
        f.write_str(s)
    }
}

/// Aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggFunc {
    CountStar,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl fmt::Display for AggFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AggFunc::CountStar | AggFunc::Count => "count",
            AggFunc::Sum => "sum",
            AggFunc::Avg => "avg",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
        })
    }
}

/// A typed scalar expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(ColumnRef),
    Literal(ScalarValue),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Not(Box<Expr>),
    Negative(Box<Expr>),
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    Case {
        branches: Vec<(Expr, Expr)>,
        else_expr: Option<Box<Expr>>,
        data_type: DataType,
    },
    Cast {
        expr: Box<Expr>,
        to: DataType,
    },
    Function {
        func: ScalarFunc,
        args: Vec<Expr>,
        data_type: DataType,
    },
}

impl Expr {
    pub fn col(c: ColumnRef) -> Expr {
        Expr::Column(c)
    }

    pub fn lit(v: ScalarValue) -> Expr {
        Expr::Literal(v)
    }

    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    pub fn and(self, other: Expr) -> Expr {
        Expr::binary(self, BinaryOp::And, other)
    }

    /// The SQL type this expression evaluates to.
    pub fn data_type(&self) -> DataType {
        match self {
            Expr::Column(c) => c.data_type,
            Expr::Literal(v) => v.data_type(),
            Expr::Binary { left, op, right } => {
                if op.is_comparison() || matches!(op, BinaryOp::And | BinaryOp::Or) {
                    DataType::Boolean
                } else if *op == BinaryOp::Concat {
                    DataType::Utf8
                } else {
                    let (l, r) = (left.data_type(), right.data_type());
                    if l == DataType::Float64 || r == DataType::Float64 {
                        DataType::Float64
                    } else if l == DataType::Null {
                        r
                    } else {
                        l
                    }
                }
            }
            Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::Like { .. }
            | Expr::InList { .. } => DataType::Boolean,
            Expr::Negative(e) => e.data_type(),
            Expr::Case { data_type, .. } | Expr::Function { data_type, .. } => *data_type,
            Expr::Cast { to, .. } => *to,
        }
    }

    /// Conservative nullability: `false` only when NULL is impossible.
    pub fn nullable(&self) -> bool {
        match self {
            Expr::Column(c) => c.nullable,
            Expr::Literal(v) => v.is_null(),
            Expr::IsNull(_) | Expr::IsNotNull(_) => false,
            Expr::Binary { left, op, right } => {
                matches!(op, BinaryOp::Divide | BinaryOp::Modulo)
                    || left.nullable()
                    || right.nullable()
            }
            Expr::Not(e) | Expr::Negative(e) => e.nullable(),
            Expr::Like { expr, pattern, .. } => expr.nullable() || pattern.nullable(),
            Expr::InList { expr, list, .. } => expr.nullable() || list.iter().any(Expr::nullable),
            Expr::Case { .. } | Expr::Cast { .. } => true,
            Expr::Function { func, args, .. } => match func {
                ScalarFunc::Coalesce => args.iter().all(Expr::nullable),
                _ => args.iter().any(Expr::nullable),
            },
        }
    }

    /// Direct children, in evaluation order.
    pub fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Column(_) | Expr::Literal(_) => vec![],
            Expr::Binary { left, right, .. } => vec![left, right],
            Expr::Not(e) | Expr::Negative(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => vec![e],
            Expr::Cast { expr, .. } => vec![expr],
            Expr::Like { expr, pattern, .. } => vec![expr, pattern],
            Expr::InList { expr, list, .. } => {
                let mut v = vec![expr.as_ref()];
                v.extend(list.iter());
                v
            }
            Expr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut v = Vec::new();
                for (w, t) in branches {
                    v.push(w);
                    v.push(t);
                }
                if let Some(e) = else_expr {
                    v.push(e);
                }
                v
            }
            Expr::Function { args, .. } => args.iter().collect(),
        }
    }

    /// Collect every column id referenced by this expression.
    pub fn collect_columns(&self, out: &mut HashSet<ColId>) {
        if let Expr::Column(c) = self {
            out.insert(c.id);
        }
        for child in self.children() {
            child.collect_columns(out);
        }
    }

    pub fn column_ids(&self) -> HashSet<ColId> {
        let mut s = HashSet::new();
        self.collect_columns(&mut s);
        s
    }

    /// True when the expression references no columns (it may still be non-literal).
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Column(_) => false,
            _ => self.children().iter().all(|c| c.is_constant()),
        }
    }

    /// Rebuild the expression bottom-up, applying `f` to every node after its children.
    pub fn transform_up(self, f: &mut dyn FnMut(Expr) -> Result<Expr>) -> Result<Expr> {
        let rebuilt = match self {
            Expr::Column(_) | Expr::Literal(_) => self,
            Expr::Binary { left, op, right } => Expr::Binary {
                left: Box::new(left.transform_up(f)?),
                op,
                right: Box::new(right.transform_up(f)?),
            },
            Expr::Not(e) => Expr::Not(Box::new(e.transform_up(f)?)),
            Expr::Negative(e) => Expr::Negative(Box::new(e.transform_up(f)?)),
            Expr::IsNull(e) => Expr::IsNull(Box::new(e.transform_up(f)?)),
            Expr::IsNotNull(e) => Expr::IsNotNull(Box::new(e.transform_up(f)?)),
            Expr::Cast { expr, to } => Expr::Cast {
                expr: Box::new(expr.transform_up(f)?),
                to,
            },
            Expr::Like {
                expr,
                pattern,
                negated,
            } => Expr::Like {
                expr: Box::new(expr.transform_up(f)?),
                pattern: Box::new(pattern.transform_up(f)?),
                negated,
            },
            Expr::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: Box::new(expr.transform_up(f)?),
                list: list
                    .into_iter()
                    .map(|e| e.transform_up(f))
                    .collect::<Result<_>>()?,
                negated,
            },
            Expr::Case {
                branches,
                else_expr,
                data_type,
            } => Expr::Case {
                branches: branches
                    .into_iter()
                    .map(|(w, t)| Ok((w.transform_up(f)?, t.transform_up(f)?)))
                    .collect::<Result<_>>()?,
                else_expr: match else_expr {
                    Some(e) => Some(Box::new(e.transform_up(f)?)),
                    None => None,
                },
                data_type,
            },
            Expr::Function {
                func,
                args,
                data_type,
            } => Expr::Function {
                func,
                args: args
                    .into_iter()
                    .map(|e| e.transform_up(f))
                    .collect::<Result<_>>()?,
                data_type,
            },
        };
        f(rebuilt)
    }

    /// Replace column references using `lookup`; unmatched columns are kept.
    pub fn substitute(self, lookup: &dyn Fn(&ColumnRef) -> Option<Expr>) -> Expr {
        // `transform_up` only fails if the callback fails, and this one never does.
        self.transform_up(&mut |e| {
            Ok(match e {
                Expr::Column(c) => lookup(&c).unwrap_or(Expr::Column(c)),
                other => other,
            })
        })
        .unwrap_or_else(|_| unreachable!("infallible substitution"))
    }

    fn precedence(&self) -> u8 {
        match self {
            Expr::Binary { op, .. } => match op {
                BinaryOp::Or => 1,
                BinaryOp::And => 2,
                o if o.is_comparison() => 4,
                BinaryOp::Concat => 5,
                BinaryOp::Plus | BinaryOp::Minus => 6,
                _ => 7,
            },
            Expr::Not(_) => 3,
            Expr::IsNull(_) | Expr::IsNotNull(_) | Expr::Like { .. } | Expr::InList { .. } => 4,
            _ => 10,
        }
    }
}

/// Split `a AND b AND c` into `[a, b, c]`.
pub fn split_conjunction(expr: Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    fn walk(e: Expr, out: &mut Vec<Expr>) {
        match e {
            Expr::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                walk(*left, out);
                walk(*right, out);
            }
            other => out.push(other),
        }
    }
    walk(expr, &mut out);
    out
}

/// Split `a OR b OR c` into `[a, b, c]`.
pub fn split_disjunction(expr: Expr) -> Vec<Expr> {
    match expr {
        Expr::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } => {
            let mut v = split_disjunction(*left);
            v.extend(split_disjunction(*right));
            v
        }
        other => vec![other],
    }
}

/// Combine predicates with AND; `None` for an empty list.
pub fn conjunction(exprs: Vec<Expr>) -> Option<Expr> {
    exprs.into_iter().reduce(|a, b| a.and(b))
}

/// Combine predicates with OR; `None` for an empty list.
pub fn disjunction(exprs: Vec<Expr>) -> Option<Expr> {
    exprs
        .into_iter()
        .reduce(|a, b| Expr::binary(a, BinaryOp::Or, b))
}

fn fmt_child(
    f: &mut fmt::Formatter<'_>,
    child: &Expr,
    parent_prec: u8,
    strict: bool,
) -> fmt::Result {
    let p = child.precedence();
    if p < parent_prec || (strict && p == parent_prec) {
        write!(f, "({child})")
    } else {
        write!(f, "{child}")
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(c) => f.write_str(&c.name),
            Expr::Literal(v) => write!(f, "{v}"),
            Expr::Binary { left, op, right } => {
                let p = self.precedence();
                fmt_child(f, left, p, false)?;
                write!(f, " {op} ")?;
                // Right operand of a left-associative operator needs parens at equal precedence.
                fmt_child(f, right, p, !matches!(op, BinaryOp::And | BinaryOp::Or))
            }
            Expr::Not(e) => {
                f.write_str("NOT ")?;
                fmt_child(f, e, 4, false)
            }
            Expr::Negative(e) => {
                f.write_str("-")?;
                fmt_child(f, e, 10, false)
            }
            Expr::IsNull(e) => {
                fmt_child(f, e, 5, false)?;
                f.write_str(" IS NULL")
            }
            Expr::IsNotNull(e) => {
                fmt_child(f, e, 5, false)?;
                f.write_str(" IS NOT NULL")
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => {
                fmt_child(f, expr, 5, false)?;
                write!(f, " {}LIKE ", if *negated { "NOT " } else { "" })?;
                fmt_child(f, pattern, 5, false)
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                fmt_child(f, expr, 5, false)?;
                let items: Vec<String> = list.iter().map(|e| e.to_string()).collect();
                write!(
                    f,
                    " {}IN ({})",
                    if *negated { "NOT " } else { "" },
                    items.join(", ")
                )
            }
            Expr::Case {
                branches,
                else_expr,
                ..
            } => {
                f.write_str("CASE")?;
                for (w, t) in branches {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_expr {
                    write!(f, " ELSE {e}")?;
                }
                f.write_str(" END")
            }
            Expr::Cast { expr, to } => write!(f, "CAST({expr} AS {to})"),
            Expr::Function { func, args, .. } => {
                let items: Vec<String> = args.iter().map(|e| e.to_string()).collect();
                write!(f, "{func}({})", items.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: ColId, name: &str) -> Expr {
        Expr::Column(ColumnRef {
            id,
            name: name.into(),
            data_type: DataType::Int64,
            nullable: true,
        })
    }

    fn i(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(v)))
    }

    #[test]
    fn display_minimizes_parentheses() {
        let e = Expr::binary(
            Expr::binary(c(1, "a"), BinaryOp::Plus, i(1)),
            BinaryOp::Multiply,
            Expr::binary(c(2, "b"), BinaryOp::Minus, i(2)),
        );
        assert_eq!(e.to_string(), "(a + 1) * (b - 2)");
        let e = Expr::binary(
            c(1, "a"),
            BinaryOp::Minus,
            Expr::binary(c(2, "b"), BinaryOp::Minus, i(1)),
        );
        assert_eq!(e.to_string(), "a - (b - 1)");
        let p = Expr::binary(c(1, "a"), BinaryOp::Gt, i(1)).and(Expr::binary(
            Expr::binary(c(2, "b"), BinaryOp::Eq, i(1)),
            BinaryOp::Or,
            Expr::IsNull(Box::new(c(2, "b"))),
        ));
        assert_eq!(p.to_string(), "a > 1 AND (b = 1 OR b IS NULL)");
    }

    #[test]
    fn conjunction_round_trip() {
        let parts = vec![c(1, "a"), c(2, "b"), c(3, "c")];
        let joined = conjunction(parts.clone()).unwrap();
        assert_eq!(split_conjunction(joined), parts);
        assert!(conjunction(vec![]).is_none());
        let ors = disjunction(parts.clone()).unwrap();
        assert_eq!(split_disjunction(ors), parts);
    }

    #[test]
    fn columns_and_substitution() {
        let e = Expr::binary(c(1, "a"), BinaryOp::Plus, c(2, "b"));
        assert_eq!(e.column_ids(), HashSet::from([1, 2]));
        assert!(!e.is_constant());
        assert!(Expr::binary(i(1), BinaryOp::Plus, i(2)).is_constant());
        let s = e.substitute(&|r| (r.id == 1).then(|| i(10)));
        assert_eq!(s.to_string(), "10 + b");
        assert_eq!(s.data_type(), DataType::Int64);
        assert!(!Expr::IsNull(Box::new(c(1, "a"))).nullable());
    }
}
