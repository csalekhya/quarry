//! Abstract syntax tree for the supported SQL subset.
//!
//! The AST is purely syntactic: names are unresolved and nothing is typed.
//! The binder (`crate::binder`) turns it into a typed logical plan.

use std::fmt;

/// A top-level statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Query(Box<Query>),
    /// `EXPLAIN [ANALYZE] [VERBOSE] <query>`
    Explain {
        analyze: bool,
        verbose: bool,
        query: Box<Query>,
    },
}

/// `[WITH ...] SELECT ... [ORDER BY ...] [LIMIT n [OFFSET m]]`
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub ctes: Vec<Cte>,
    pub body: Select,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

/// One `name [(col, ...)] AS (query)` entry of a `WITH` clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Vec<String>,
    pub query: Box<Query>,
}

/// A single `SELECT` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Select {
    pub distinct: bool,
    pub projection: Vec<SelectItem>,
    /// Comma-separated `FROM` items; empty means `SELECT` without `FROM`.
    pub from: Vec<TableWithJoins>,
    pub selection: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// `*`
    Wildcard,
    /// `t.*`
    QualifiedWildcard(String),
    /// `expr [AS alias]`
    Expr { expr: Expr, alias: Option<String> },
}

/// A `FROM` item followed by any number of explicit joins.
#[derive(Debug, Clone, PartialEq)]
pub struct TableWithJoins {
    pub relation: TableFactor,
    pub joins: Vec<Join>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableFactor {
    /// A catalog table or CTE reference.
    Table { name: String, alias: Option<String> },
    /// `(subquery) [AS] alias [(col, ...)]`
    Derived {
        subquery: Box<Query>,
        alias: String,
        columns: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Join {
    pub relation: TableFactor,
    pub kind: JoinKind,
    /// `ON` condition; `None` only for `CROSS JOIN`.
    pub on: Option<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Cross,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    pub expr: Expr,
    pub asc: bool,
    /// Explicit `NULLS FIRST/LAST`; `None` means the engine default
    /// (NULLs sort as the smallest value, like SQLite).
    pub nulls_first: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    /// `DATE 'YYYY-MM-DD'`
    Date(String),
    /// `INTERVAL 'n' unit`
    Interval {
        value: i64,
        unit: IntervalUnit,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalUnit {
    Day,
    Month,
    Year,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateField {
    Year,
    Month,
    Day,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Concat,
}

impl BinaryOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
        )
    }

    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            BinaryOp::Plus
                | BinaryOp::Minus
                | BinaryOp::Multiply
                | BinaryOp::Divide
                | BinaryOp::Modulo
        )
    }

    /// The operator with its operands swapped (`a < b` == `b > a`).
    pub fn commute(self) -> Option<BinaryOp> {
        Some(match self {
            BinaryOp::Eq => BinaryOp::Eq,
            BinaryOp::NotEq => BinaryOp::NotEq,
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            BinaryOp::Plus => BinaryOp::Plus,
            BinaryOp::Multiply => BinaryOp::Multiply,
            BinaryOp::And => BinaryOp::And,
            BinaryOp::Or => BinaryOp::Or,
            _ => return None,
        })
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BinaryOp::Plus => "+",
            BinaryOp::Minus => "-",
            BinaryOp::Multiply => "*",
            BinaryOp::Divide => "/",
            BinaryOp::Modulo => "%",
            BinaryOp::Eq => "=",
            BinaryOp::NotEq => "<>",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
            BinaryOp::Concat => "||",
        })
    }
}

/// A SQL type name as written in `CAST(x AS type)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeName {
    Boolean,
    BigInt,
    Double,
    Varchar,
    Date,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// `col` or `qualifier.col`
    Identifier(Vec<String>),
    Literal(Literal),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    Case {
        operand: Option<Box<Expr>>,
        branches: Vec<(Expr, Expr)>,
        else_expr: Option<Box<Expr>>,
    },
    Cast {
        expr: Box<Expr>,
        data_type: TypeName,
    },
    Extract {
        field: DateField,
        expr: Box<Expr>,
    },
    /// Scalar or aggregate function call. `star` marks `count(*)`.
    Function {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
        star: bool,
    },
}

impl Expr {
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
            Expr::Unary {
                op: UnaryOp::Not, ..
            } => 3,
            Expr::IsNull { .. }
            | Expr::Between { .. }
            | Expr::InList { .. }
            | Expr::Like { .. } => 4,
            _ => 10,
        }
    }
}

fn child(f: &mut fmt::Formatter<'_>, e: &Expr, parent: u8, strict: bool) -> fmt::Result {
    let p = e.precedence();
    if p < parent || (strict && p == parent) {
        write!(f, "({e})")
    } else {
        write!(f, "{e}")
    }
}

impl fmt::Display for Expr {
    /// Canonical SQL rendering, used to derive default output column names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Identifier(parts) => write!(f, "{}", parts.join(".")),
            Expr::Literal(l) => match l {
                Literal::Integer(i) => write!(f, "{i}"),
                Literal::Float(v) => write!(f, "{v}"),
                Literal::String(s) => write!(f, "'{}'", s.replace('\'', "''")),
                Literal::Boolean(b) => write!(f, "{b}"),
                Literal::Null => write!(f, "NULL"),
                Literal::Date(d) => write!(f, "DATE '{d}'"),
                Literal::Interval { value, unit } => {
                    let u = match unit {
                        IntervalUnit::Day => "DAY",
                        IntervalUnit::Month => "MONTH",
                        IntervalUnit::Year => "YEAR",
                    };
                    write!(f, "INTERVAL '{value}' {u}")
                }
            },
            Expr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => {
                f.write_str("-")?;
                child(f, expr, 10, false)
            }
            Expr::Unary {
                op: UnaryOp::Not,
                expr,
            } => {
                f.write_str("NOT ")?;
                child(f, expr, 4, false)
            }
            Expr::Binary { left, op, right } => {
                let p = self.precedence();
                child(f, left, p, false)?;
                write!(f, " {op} ")?;
                child(f, right, p, !matches!(op, BinaryOp::And | BinaryOp::Or))
            }
            Expr::IsNull { expr, negated } => {
                child(f, expr, 5, false)?;
                write!(f, " IS {}NULL", if *negated { "NOT " } else { "" })
            }
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                child(f, expr, 5, false)?;
                write!(f, " {}BETWEEN ", if *negated { "NOT " } else { "" })?;
                child(f, low, 5, false)?;
                f.write_str(" AND ")?;
                child(f, high, 5, false)
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                child(f, expr, 5, false)?;
                let items: Vec<String> = list.iter().map(|e| e.to_string()).collect();
                write!(
                    f,
                    " {}IN ({})",
                    if *negated { "NOT " } else { "" },
                    items.join(", ")
                )
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => {
                child(f, expr, 5, false)?;
                write!(f, " {}LIKE ", if *negated { "NOT " } else { "" })?;
                child(f, pattern, 5, false)
            }
            Expr::Case {
                operand,
                branches,
                else_expr,
            } => {
                write!(f, "CASE")?;
                if let Some(op) = operand {
                    write!(f, " {op}")?;
                }
                for (w, t) in branches {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_expr {
                    write!(f, " ELSE {e}")?;
                }
                write!(f, " END")
            }
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type:?})"),
            Expr::Extract { field, expr } => write!(f, "EXTRACT({field:?} FROM {expr})"),
            Expr::Function {
                name,
                args,
                distinct,
                star,
            } => {
                if *star {
                    return write!(f, "{name}(*)");
                }
                let items: Vec<String> = args.iter().map(|e| e.to_string()).collect();
                write!(
                    f,
                    "{name}({}{})",
                    if *distinct { "DISTINCT " } else { "" },
                    items.join(", ")
                )
            }
        }
    }
}
