//! NULL-heavy feature tables shared by the feature and fuzz tests.

use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::{ArrayRef, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use quarry::tpch::Rng;
use quarry::types::parse_date;
use quarry::Engine;

use super::{load_sqlite, new_sqlite, Fixture};

const WORDS: &[&str] = &[
    "apple", "banana", "cherry", "", "date", "Elder", "fig", "grape", "ab", "b_c",
];

fn maybe<T>(rng: &mut Rng, pct_null: i64, v: T) -> Option<T> {
    (rng.range(0, 99) >= pct_null).then_some(v)
}

pub fn tables() -> Vec<(&'static str, SchemaRef, RecordBatch)> {
    let mut rng = Rng::new(42);
    let n = 300;
    let base = parse_date("1995-01-01").unwrap();
    let (mut a, mut b, mut f, mut s, mut d, mut g) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let av = rng.range(-20, 100);
        a.push(maybe(&mut rng, 15, av));
        let bv = rng.range(0, 100);
        b.push(maybe(&mut rng, 15, bv));
        let fv = rng.range(-40, 40) as f64 * 0.25;
        f.push(maybe(&mut rng, 15, fv));
        let sv = WORDS[rng.range(0, WORDS.len() as i64 - 1) as usize].to_string();
        s.push(maybe(&mut rng, 15, sv));
        let dv = base + rng.range(0, 365) as i32;
        d.push(maybe(&mut rng, 15, dv));
        let gv = ["x", "y", "z"][rng.range(0, 2) as usize].to_string();
        g.push(maybe(&mut rng, 10, gv));
    }
    let t_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("d", DataType::Date32, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((1..=n).collect::<Vec<i64>>())),
        Arc::new(Int64Array::from(a)),
        Arc::new(Int64Array::from(b)),
        Arc::new(Float64Array::from(f)),
        Arc::new(StringArray::from(s)),
        Arc::new(Date32Array::from(d)),
        Arc::new(StringArray::from(g)),
    ];
    let t = RecordBatch::try_new(t_schema.clone(), cols).unwrap();

    let m = 100;
    let (mut ua, mut ux) = (vec![], vec![]);
    for _ in 0..m {
        let av = rng.range(-20, 100);
        ua.push(maybe(&mut rng, 20, av));
        let xv = WORDS[rng.range(0, WORDS.len() as i64 - 1) as usize].to_string();
        ux.push(maybe(&mut rng, 20, xv));
    }
    let u_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int64, true),
        Field::new("x", DataType::Utf8, true),
    ]));
    let u = RecordBatch::try_new(
        u_schema.clone(),
        vec![
            Arc::new(Int64Array::from((1..=m).collect::<Vec<i64>>())),
            Arc::new(Int64Array::from(ua)),
            Arc::new(StringArray::from(ux)),
        ],
    )
    .unwrap();
    let e = RecordBatch::new_empty(u_schema.clone());
    vec![
        ("t", t_schema, t),
        ("u", u_schema.clone(), u),
        ("e", u_schema, e),
    ]
}

pub fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let mut engine = Engine::new();
        // Small batches so multi-batch code paths are exercised.
        engine.set_batch_size(64);
        let conn = new_sqlite();
        for (name, schema, batch) in tables() {
            engine
                .register_batches(name, schema.clone(), vec![batch.clone()])
                .unwrap();
            load_sqlite(&conn, name, &schema, &[batch]);
        }
        Fixture {
            engine,
            sqlite: Mutex::new(conn),
        }
    })
}
