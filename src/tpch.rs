//! Deterministic generator for TPC-H-*like* data.
//!
//! This is not the official `dbgen`: it follows the TPC-H schema, table
//! cardinalities (per scale factor), key relationships and value domains
//! closely enough that the standard queries are meaningful, but text fields
//! are shorter and some distributions are simplified. The same scale factor
//! always produces byte-identical data (SplitMix64 with fixed seeds).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{lexsort_to_indices, take_record_batch, SortColumn};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::error::Result;
use crate::types::parse_date;

/// SplitMix64: tiny, fast, and stable across platforms and versions.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[lo, hi]`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }

    /// Uniform money value in `[lo, hi]` with two decimals.
    pub fn money(&mut self, lo: f64, hi: f64) -> f64 {
        let cents = self.range((lo * 100.0).round() as i64, (hi * 100.0).round() as i64);
        cents as f64 / 100.0
    }

    pub fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.range(0, items.len() as i64 - 1) as usize]
    }

    fn words(&mut self, n: usize) -> String {
        (0..n)
            .map(|_| self.pick(WORDS))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

const WORDS: &[&str] = &[
    "furiously",
    "special",
    "requests",
    "pending",
    "deposits",
    "carefully",
    "final",
    "accounts",
    "regular",
    "ideas",
    "quickly",
    "express",
    "packages",
    "blithely",
    "ironic",
    "theodolites",
    "bold",
    "foxes",
    "slyly",
    "unusual",
    "instructions",
    "even",
    "pinto",
    "beans",
    "Customer",
    "Complaints",
    "asymptotes",
    "dependencies",
    "silent",
    "platelets",
];

const REGIONS: [&str; 5] = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"];

const NATIONS: [(&str, i64); 25] = [
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
];

const SEGMENTS: &[&str] = &[
    "AUTOMOBILE",
    "BUILDING",
    "FURNITURE",
    "MACHINERY",
    "HOUSEHOLD",
];
const PRIORITIES: &[&str] = &["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"];
const INSTRUCTIONS: &[&str] = &[
    "DELIVER IN PERSON",
    "COLLECT COD",
    "NONE",
    "TAKE BACK RETURN",
];
const MODES: &[&str] = &["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"];
const TYPE1: &[&str] = &["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"];
const TYPE2: &[&str] = &["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"];
const TYPE3: &[&str] = &["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"];
const CONT1: &[&str] = &["SM", "LG", "MED", "JUMBO", "WRAP"];
const CONT2: &[&str] = &["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"];
const COLORS: &[&str] = &[
    "almond",
    "antique",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanched",
    "blue",
    "blush",
    "brown",
    "burlywood",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflower",
    "cream",
    "cyan",
    "dark",
    "forest",
    "frosted",
    "green",
    "honeydew",
    "khaki",
    "lavender",
    "lemon",
    "linen",
    "magenta",
    "maroon",
    "medium",
    "metallic",
    "midnight",
    "navy",
    "olive",
    "orange",
    "orchid",
    "pale",
    "peach",
    "pink",
    "plum",
    "powder",
    "puff",
    "purple",
    "red",
    "rose",
    "royal",
    "saddle",
    "salmon",
    "sandy",
    "seashell",
    "sienna",
    "sky",
    "slate",
    "smoke",
    "snow",
    "spring",
    "steel",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "yellow",
];

/// A generated table.
pub struct Table {
    pub name: &'static str,
    pub schema: SchemaRef,
    pub batch: RecordBatch,
}

fn schema(cols: &[(&str, DataType)]) -> SchemaRef {
    Arc::new(Schema::new(
        cols.iter()
            .map(|(n, t)| Field::new(*n, t.clone(), false))
            .collect::<Vec<_>>(),
    ))
}

fn ints(v: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(v))
}

fn floats(v: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(v))
}

fn strs(v: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(v))
}

fn dates(v: Vec<i32>) -> ArrayRef {
    Arc::new(Date32Array::from(v))
}

fn table(name: &'static str, cols: &[(&str, DataType)], arrays: Vec<ArrayRef>) -> Result<Table> {
    let schema = schema(cols);
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    Ok(Table {
        name,
        schema,
        batch,
    })
}

fn scaled(base: f64, sf: f64) -> usize {
    ((base * sf).round() as usize).max(1)
}

fn phone(rng: &mut Rng, nation: i64) -> String {
    format!(
        "{}-{}-{}-{}",
        nation + 10,
        rng.range(100, 999),
        rng.range(100, 999),
        rng.range(1000, 9999)
    )
}

/// The supplier of the `i`-th (0..4) partsupp row for `part`. Simplified
/// from the TPC-H formula so the four suppliers are distinct whenever there
/// are at least four suppliers.
fn supplier_for(part: i64, i: i64, suppliers: i64) -> i64 {
    (part + i * (suppliers / 4).max(1)) % suppliers + 1
}

fn retail_price(part: i64) -> f64 {
    (90_000 + (part / 10) % 20_001 + 100 * (part % 1000)) as f64 / 100.0
}

/// Generate all eight tables at scale factor `sf` (1.0 ≈ 6M lineitems).
pub fn generate(sf: f64) -> Result<Vec<Table>> {
    use DataType::{Date32 as D, Float64 as F, Int64 as I, Utf8 as S};
    let mut tables = Vec::new();

    let mut rng = Rng::new(1);
    tables.push(table(
        "region",
        &[("r_regionkey", I), ("r_name", S), ("r_comment", S)],
        vec![
            ints((0..5).collect()),
            strs(REGIONS.iter().map(|s| s.to_string()).collect()),
            strs((0..5).map(|_| rng.words(4)).collect()),
        ],
    )?);

    let mut rng = Rng::new(2);
    tables.push(table(
        "nation",
        &[
            ("n_nationkey", I),
            ("n_name", S),
            ("n_regionkey", I),
            ("n_comment", S),
        ],
        vec![
            ints((0..25).collect()),
            strs(NATIONS.iter().map(|(n, _)| n.to_string()).collect()),
            ints(NATIONS.iter().map(|(_, r)| *r).collect()),
            strs((0..25).map(|_| rng.words(5)).collect()),
        ],
    )?);

    let n_supp = scaled(10_000.0, sf) as i64;
    let mut rng = Rng::new(3);
    let mut s_nation = Vec::new();
    let (mut s_name, mut s_addr, mut s_phone, mut s_bal, mut s_comment) =
        (vec![], vec![], vec![], vec![], vec![]);
    for k in 1..=n_supp {
        let nk = rng.range(0, 24);
        s_name.push(format!("Supplier#{k:09}"));
        s_addr.push(format!("{} street {}", rng.pick(COLORS), rng.range(1, 999)));
        s_phone.push(phone(&mut rng, nk));
        s_bal.push(rng.money(-999.99, 9999.99));
        let n = rng.range(3, 6) as usize;
        s_comment.push(rng.words(n));
        s_nation.push(nk);
    }
    tables.push(table(
        "supplier",
        &[
            ("s_suppkey", I),
            ("s_name", S),
            ("s_address", S),
            ("s_nationkey", I),
            ("s_phone", S),
            ("s_acctbal", F),
            ("s_comment", S),
        ],
        vec![
            ints((1..=n_supp).collect()),
            strs(s_name),
            strs(s_addr),
            ints(s_nation),
            strs(s_phone),
            floats(s_bal),
            strs(s_comment),
        ],
    )?);

    let n_cust = scaled(150_000.0, sf) as i64;
    let mut rng = Rng::new(4);
    let (mut c_name, mut c_addr, mut c_nation, mut c_phone, mut c_bal, mut c_seg, mut c_comment) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for k in 1..=n_cust {
        let nk = rng.range(0, 24);
        c_name.push(format!("Customer#{k:09}"));
        c_addr.push(format!("{} avenue {}", rng.pick(COLORS), rng.range(1, 999)));
        c_nation.push(nk);
        c_phone.push(phone(&mut rng, nk));
        c_bal.push(rng.money(-999.99, 9999.99));
        c_seg.push(rng.pick(SEGMENTS).to_string());
        let n = rng.range(3, 6) as usize;
        c_comment.push(rng.words(n));
    }
    tables.push(table(
        "customer",
        &[
            ("c_custkey", I),
            ("c_name", S),
            ("c_address", S),
            ("c_nationkey", I),
            ("c_phone", S),
            ("c_acctbal", F),
            ("c_mktsegment", S),
            ("c_comment", S),
        ],
        vec![
            ints((1..=n_cust).collect()),
            strs(c_name),
            strs(c_addr),
            ints(c_nation),
            strs(c_phone),
            floats(c_bal),
            strs(c_seg),
            strs(c_comment),
        ],
    )?);

    let n_part = scaled(200_000.0, sf) as i64;
    let mut rng = Rng::new(5);
    let (
        mut p_name,
        mut p_mfgr,
        mut p_brand,
        mut p_type,
        mut p_size,
        mut p_cont,
        mut p_price,
        mut p_comment,
    ) = (
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    );
    for k in 1..=n_part {
        p_name.push(
            (0..5)
                .map(|_| rng.pick(COLORS))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let m = rng.range(1, 5);
        p_mfgr.push(format!("Manufacturer#{m}"));
        p_brand.push(format!("Brand#{m}{}", rng.range(1, 5)));
        p_type.push(format!(
            "{} {} {}",
            rng.pick(TYPE1),
            rng.pick(TYPE2),
            rng.pick(TYPE3)
        ));
        p_size.push(rng.range(1, 50));
        p_cont.push(format!("{} {}", rng.pick(CONT1), rng.pick(CONT2)));
        p_price.push(retail_price(k));
        let n = rng.range(2, 4) as usize;
        p_comment.push(rng.words(n));
    }
    tables.push(table(
        "part",
        &[
            ("p_partkey", I),
            ("p_name", S),
            ("p_mfgr", S),
            ("p_brand", S),
            ("p_type", S),
            ("p_size", I),
            ("p_container", S),
            ("p_retailprice", F),
            ("p_comment", S),
        ],
        vec![
            ints((1..=n_part).collect()),
            strs(p_name),
            strs(p_mfgr),
            strs(p_brand),
            strs(p_type),
            ints(p_size),
            strs(p_cont),
            floats(p_price),
            strs(p_comment),
        ],
    )?);

    let mut rng = Rng::new(6);
    let (mut ps_part, mut ps_supp, mut ps_qty, mut ps_cost, mut ps_comment) =
        (vec![], vec![], vec![], vec![], vec![]);
    for p in 1..=n_part {
        for i in 0..4 {
            ps_part.push(p);
            ps_supp.push(supplier_for(p, i, n_supp));
            ps_qty.push(rng.range(1, 9999));
            ps_cost.push(rng.money(1.0, 1000.0));
            let n = rng.range(3, 6) as usize;
            ps_comment.push(rng.words(n));
        }
    }
    tables.push(table(
        "partsupp",
        &[
            ("ps_partkey", I),
            ("ps_suppkey", I),
            ("ps_availqty", I),
            ("ps_supplycost", F),
            ("ps_comment", S),
        ],
        vec![
            ints(ps_part),
            ints(ps_supp),
            ints(ps_qty),
            floats(ps_cost),
            strs(ps_comment),
        ],
    )?);

    // Orders and lineitems are generated together (status and total price derive from lines).
    let n_orders = scaled(1_500_000.0, sf) as i64;
    let start = i64::from(parse_date("1992-01-01").unwrap_or_default());
    let end = i64::from(parse_date("1998-08-02").unwrap_or_default());
    let current = parse_date("1995-06-17").unwrap_or_default();
    let mut rng = Rng::new(7);
    let mut o = OrderCols::default();
    let mut l = LineCols::default();
    for ok in 1..=n_orders {
        let mut cust = rng.range(1, n_cust);
        if cust % 3 == 0 {
            // As in TPC-H, a third of customers never place orders.
            cust = if cust == n_cust { 1 } else { cust + 1 };
        }
        let odate = rng.range(start, end - 151) as i32;
        let lines = rng.range(1, 7);
        let mut total = 0.0;
        let (mut all_f, mut all_o) = (true, true);
        for ln in 1..=lines {
            let part = rng.range(1, n_part);
            let supp = supplier_for(part, rng.range(0, 3), n_supp);
            let qty = rng.range(1, 50) as f64;
            let price = (qty * retail_price(part) * 100.0).round() / 100.0;
            let disc = rng.range(0, 10) as f64 / 100.0;
            let tax = rng.range(0, 8) as f64 / 100.0;
            let ship = odate + rng.range(1, 121) as i32;
            let commit = odate + rng.range(30, 90) as i32;
            let receipt = ship + rng.range(1, 30) as i32;
            let rflag = if receipt <= current {
                if rng.range(0, 1) == 0 {
                    "R"
                } else {
                    "A"
                }
            } else {
                "N"
            };
            let status = if ship > current { "O" } else { "F" };
            all_f &= status == "F";
            all_o &= status == "O";
            total += price * (1.0 + tax) * (1.0 - disc);
            l.orderkey.push(ok);
            l.partkey.push(part);
            l.suppkey.push(supp);
            l.linenumber.push(ln);
            l.quantity.push(qty);
            l.extendedprice.push(price);
            l.discount.push(disc);
            l.tax.push(tax);
            l.returnflag.push(rflag.to_string());
            l.linestatus.push(status.to_string());
            l.shipdate.push(ship);
            l.commitdate.push(commit);
            l.receiptdate.push(receipt);
            l.shipinstruct.push(rng.pick(INSTRUCTIONS).to_string());
            l.shipmode.push(rng.pick(MODES).to_string());
            let n = rng.range(2, 4) as usize;
            l.comment.push(rng.words(n));
        }
        o.orderkey.push(ok);
        o.custkey.push(cust);
        o.status.push(
            if all_f {
                "F"
            } else if all_o {
                "O"
            } else {
                "P"
            }
            .to_string(),
        );
        o.totalprice.push((total * 100.0).round() / 100.0);
        o.orderdate.push(odate);
        o.priority.push(rng.pick(PRIORITIES).to_string());
        o.clerk.push(format!(
            "Clerk#{:09}",
            rng.range(1, (n_orders / 1500).max(1))
        ));
        o.shippriority.push(0);
        let n = rng.range(3, 7) as usize;
        o.comment.push(rng.words(n));
    }
    tables.push(table(
        "orders",
        &[
            ("o_orderkey", I),
            ("o_custkey", I),
            ("o_orderstatus", S),
            ("o_totalprice", F),
            ("o_orderdate", D),
            ("o_orderpriority", S),
            ("o_clerk", S),
            ("o_shippriority", I),
            ("o_comment", S),
        ],
        vec![
            ints(o.orderkey),
            ints(o.custkey),
            strs(o.status),
            floats(o.totalprice),
            dates(o.orderdate),
            strs(o.priority),
            strs(o.clerk),
            ints(o.shippriority),
            strs(o.comment),
        ],
    )?);
    tables.push(table(
        "lineitem",
        &[
            ("l_orderkey", I),
            ("l_partkey", I),
            ("l_suppkey", I),
            ("l_linenumber", I),
            ("l_quantity", F),
            ("l_extendedprice", F),
            ("l_discount", F),
            ("l_tax", F),
            ("l_returnflag", S),
            ("l_linestatus", S),
            ("l_shipdate", D),
            ("l_commitdate", D),
            ("l_receiptdate", D),
            ("l_shipinstruct", S),
            ("l_shipmode", S),
            ("l_comment", S),
        ],
        vec![
            ints(l.orderkey),
            ints(l.partkey),
            ints(l.suppkey),
            ints(l.linenumber),
            floats(l.quantity),
            floats(l.extendedprice),
            floats(l.discount),
            floats(l.tax),
            strs(l.returnflag),
            strs(l.linestatus),
            dates(l.shipdate),
            dates(l.commitdate),
            dates(l.receiptdate),
            strs(l.shipinstruct),
            strs(l.shipmode),
            strs(l.comment),
        ],
    )?);
    Ok(tables)
}

#[derive(Default)]
struct OrderCols {
    orderkey: Vec<i64>,
    custkey: Vec<i64>,
    status: Vec<String>,
    totalprice: Vec<f64>,
    orderdate: Vec<i32>,
    priority: Vec<String>,
    clerk: Vec<String>,
    shippriority: Vec<i64>,
    comment: Vec<String>,
}

#[derive(Default)]
struct LineCols {
    orderkey: Vec<i64>,
    partkey: Vec<i64>,
    suppkey: Vec<i64>,
    linenumber: Vec<i64>,
    quantity: Vec<f64>,
    extendedprice: Vec<f64>,
    discount: Vec<f64>,
    tax: Vec<f64>,
    returnflag: Vec<String>,
    linestatus: Vec<String>,
    shipdate: Vec<i32>,
    commitdate: Vec<i32>,
    receiptdate: Vec<i32>,
    shipinstruct: Vec<String>,
    shipmode: Vec<String>,
    comment: Vec<String>,
}

/// Output format for [`write_tables`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Csv,
    Parquet,
}

/// Write each table to `dir/<name>.csv` or `dir/<name>.parquet`.
///
/// Parquet files use `row_group_rows`-row row groups; `lineitem` is clustered
/// by `l_shipdate` and `orders` by `o_orderdate` so that date filters can
/// prune row groups (as a time-partitioned warehouse table would).
pub fn write_tables(
    tables: &[Table],
    dir: &Path,
    format: FileFormat,
    row_group_rows: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for t in tables {
        match format {
            FileFormat::Csv => {
                let file = File::create(dir.join(format!("{}.csv", t.name)))?;
                let mut w = arrow::csv::WriterBuilder::new()
                    .with_header(true)
                    .with_date_format("%Y-%m-%d".to_string())
                    .build(file);
                w.write(&t.batch)?;
            }
            FileFormat::Parquet => {
                let batch = match t.name {
                    "lineitem" => sort_by(&t.batch, "l_shipdate")?,
                    "orders" => sort_by(&t.batch, "o_orderdate")?,
                    _ => t.batch.clone(),
                };
                let file = File::create(dir.join(format!("{}.parquet", t.name)))?;
                let props = WriterProperties::builder()
                    .set_max_row_group_row_count(Some(row_group_rows.max(1)))
                    .build();
                let mut w = ArrowWriter::try_new(file, Arc::clone(&t.schema), Some(props))?;
                w.write(&batch)?;
                w.close()?;
            }
        }
    }
    Ok(())
}

fn sort_by(batch: &RecordBatch, col: &str) -> Result<RecordBatch> {
    let idx = batch.schema().index_of(col)?;
    let indices = lexsort_to_indices(
        &[SortColumn {
            values: Arc::clone(batch.column(idx)),
            options: None,
        }],
        None,
    )?;
    Ok(take_record_batch(batch, &indices)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_correctly_sized() {
        let a = generate(0.001).unwrap();
        let b = generate(0.001).unwrap();
        let rows: Vec<(&str, usize)> = a.iter().map(|t| (t.name, t.batch.num_rows())).collect();
        assert_eq!(
            rows[..7].iter().map(|r| r.1).collect::<Vec<_>>(),
            vec![5, 25, 10, 150, 200, 800, 1500]
        );
        assert!(rows[7].1 >= 1500 && rows[7].1 <= 1500 * 7);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.batch, y.batch, "{} differs between runs", x.name);
        }
    }

    #[test]
    fn partsupp_suppliers_are_distinct_per_part() {
        for p in 1..200 {
            let s: std::collections::HashSet<i64> =
                (0..4).map(|i| supplier_for(p, i, 10)).collect();
            assert_eq!(s.len(), 4);
        }
    }
}
