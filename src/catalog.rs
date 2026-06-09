//! Table catalog and statistics.
//!
//! Registering a table runs a one-off `ANALYZE` pass: one streaming scan that
//! records the row count and, per column, the null count, min/max and the
//! number of distinct values (exact, via a set of 64-bit value hashes). The
//! optimizer's cardinality estimator reads these numbers.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::datatypes::{Date32Type, Field, Float64Type, Int64Type, Schema, SchemaRef};
use rustc_hash::{FxBuildHasher, FxHashSet};

use crate::error::{QuarryError, Result};
use crate::types::{DataType, ScalarValue};

/// Where a table's rows come from.
#[derive(Debug, Clone)]
pub enum TableSource {
    /// Record batches held in memory (already normalized to engine types).
    Memory(Vec<RecordBatch>),
    /// A CSV file read with the given Arrow file schema.
    Csv {
        path: PathBuf,
        has_header: bool,
        delimiter: u8,
        file_schema: SchemaRef,
    },
    /// A Parquet file.
    Parquet { path: PathBuf },
}

impl TableSource {
    pub fn kind(&self) -> &'static str {
        match self {
            TableSource::Memory(_) => "memory",
            TableSource::Csv { .. } => "csv",
            TableSource::Parquet { .. } => "parquet",
        }
    }
}

/// Per-column statistics gathered at registration.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    pub ndv: u64,
    pub null_count: u64,
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableStats {
    pub row_count: u64,
    pub columns: Vec<ColumnStats>,
}

/// A registered table.
#[derive(Debug)]
pub struct TableEntry {
    pub name: String,
    /// Normalized schema: lower-case column names, engine types only.
    pub schema: SchemaRef,
    pub source: TableSource,
    pub stats: TableStats,
}

impl TableEntry {
    pub fn column_type(&self, i: usize) -> DataType {
        DataType::from_arrow(self.schema.field(i).data_type()).unwrap_or(DataType::Null)
    }
}

/// Name → table map. Names are case-insensitive (stored lower case).
#[derive(Debug, Default, Clone)]
pub struct Catalog {
    tables: HashMap<String, Arc<TableEntry>>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<Arc<TableEntry>> {
        self.tables.get(&name.to_ascii_lowercase()).cloned()
    }

    pub fn insert(&mut self, entry: TableEntry) -> Arc<TableEntry> {
        let entry = Arc::new(entry);
        self.tables.insert(entry.name.clone(), Arc::clone(&entry));
        entry
    }

    pub fn deregister(&mut self, name: &str) -> bool {
        self.tables.remove(&name.to_ascii_lowercase()).is_some()
    }

    /// Table names in sorted order.
    pub fn table_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tables.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Map a source schema onto the engine's normalized schema.
pub fn normalize_schema(schema: &Schema) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut seen = FxHashSet::default();
    for f in schema.fields() {
        let name = f.name().to_ascii_lowercase();
        if !seen.insert(name.clone()) {
            return Err(QuarryError::Catalog(format!(
                "duplicate column name '{name}'"
            )));
        }
        let dt = DataType::from_arrow(f.data_type()).ok_or_else(|| {
            QuarryError::Catalog(format!(
                "column '{}' has unsupported type {}",
                f.name(),
                f.data_type()
            ))
        })?;
        // Untyped columns (all-NULL CSV columns) are treated as text.
        let dt = if dt == DataType::Null {
            DataType::Utf8
        } else {
            dt
        };
        fields.push(Field::new(name, dt.to_arrow(), true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Streaming statistics builder, fed batch by batch.
pub struct StatsCollector {
    types: Vec<DataType>,
    rows: u64,
    nulls: Vec<u64>,
    mins: Vec<Option<ScalarValue>>,
    maxs: Vec<Option<ScalarValue>>,
    hashes: Vec<FxHashSet<u64>>,
}

impl StatsCollector {
    pub fn new(schema: &Schema) -> Self {
        let n = schema.fields().len();
        StatsCollector {
            types: schema
                .fields()
                .iter()
                .map(|f| DataType::from_arrow(f.data_type()).unwrap_or(DataType::Null))
                .collect(),
            rows: 0,
            nulls: vec![0; n],
            mins: vec![None; n],
            maxs: vec![None; n],
            hashes: vec![FxHashSet::default(); n],
        }
    }

    pub fn update(&mut self, batch: &RecordBatch) -> Result<()> {
        self.rows += batch.num_rows() as u64;
        for (i, col) in batch.columns().iter().enumerate() {
            self.nulls[i] += col.null_count() as u64;
            self.update_column(i, col)?;
        }
        Ok(())
    }

    fn update_column(&mut self, i: usize, col: &ArrayRef) -> Result<()> {
        let set = &mut self.hashes[i];
        let (lo, hi): (Option<ScalarValue>, Option<ScalarValue>) = match self.types[i] {
            DataType::Int64 => {
                let a = col.as_primitive::<Int64Type>();
                a.iter().flatten().for_each(|v| {
                    set.insert(hash_of(v));
                });
                (
                    arrow::compute::min(a).map(|v| ScalarValue::Int64(Some(v))),
                    arrow::compute::max(a).map(|v| ScalarValue::Int64(Some(v))),
                )
            }
            DataType::Float64 => {
                let a = col.as_primitive::<Float64Type>();
                a.iter().flatten().for_each(|v| {
                    // -0.0 and 0.0 are one distinct value.
                    set.insert(hash_of((v + 0.0).to_bits()));
                });
                (
                    arrow::compute::min(a).map(|v| ScalarValue::Float64(Some(v))),
                    arrow::compute::max(a).map(|v| ScalarValue::Float64(Some(v))),
                )
            }
            DataType::Date32 => {
                let a = col.as_primitive::<Date32Type>();
                a.iter().flatten().for_each(|v| {
                    set.insert(hash_of(v));
                });
                (
                    arrow::compute::min(a).map(|v| ScalarValue::Date32(Some(v))),
                    arrow::compute::max(a).map(|v| ScalarValue::Date32(Some(v))),
                )
            }
            DataType::Utf8 => {
                let a = col.as_string::<i32>();
                a.iter().flatten().for_each(|v| {
                    set.insert(hash_of(v));
                });
                (
                    arrow::compute::min_string(a).map(|v| ScalarValue::Utf8(Some(v.to_string()))),
                    arrow::compute::max_string(a).map(|v| ScalarValue::Utf8(Some(v.to_string()))),
                )
            }
            DataType::Boolean => {
                let a = col.as_boolean();
                a.iter().flatten().for_each(|v| {
                    set.insert(hash_of(v));
                });
                (None, None)
            }
            DataType::Null => (None, None),
        };
        merge_bound(&mut self.mins[i], lo, std::cmp::Ordering::Less);
        merge_bound(&mut self.maxs[i], hi, std::cmp::Ordering::Greater);
        Ok(())
    }

    pub fn finish(self) -> TableStats {
        let columns = self
            .hashes
            .into_iter()
            .zip(self.nulls)
            .zip(self.mins.into_iter().zip(self.maxs))
            .map(|((h, null_count), (min, max))| ColumnStats {
                ndv: h.len() as u64,
                null_count,
                min,
                max,
            })
            .collect();
        TableStats {
            row_count: self.rows,
            columns,
        }
    }
}

fn hash_of<T: Hash>(v: T) -> u64 {
    FxBuildHasher.hash_one(v)
}

fn merge_bound(
    slot: &mut Option<ScalarValue>,
    cand: Option<ScalarValue>,
    keep: std::cmp::Ordering,
) {
    if let Some(c) = cand {
        let replace = match slot {
            None => true,
            Some(cur) => c.partial_cmp_sql(cur) == Some(keep),
        };
        if replace {
            *slot = Some(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::DataType as ArrowType;

    #[test]
    fn stats_over_multiple_batches() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int64, true),
            Field::new("s", ArrowType::Utf8, true),
            Field::new("f", ArrowType::Float64, true),
        ]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(3), None, Some(1)])),
                Arc::new(StringArray::from(vec![Some("b"), Some("a"), None])),
                Arc::new(Float64Array::from(vec![1.5, 2.5, 1.5])),
            ],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(9), Some(3)])),
                Arc::new(StringArray::from(vec![Some("z"), Some("a")])),
                Arc::new(Float64Array::from(vec![-1.0, 2.5])),
            ],
        )
        .unwrap();
        let mut c = StatsCollector::new(&schema);
        c.update(&b1).unwrap();
        c.update(&b2).unwrap();
        let s = c.finish();
        assert_eq!(s.row_count, 5);
        assert_eq!(s.columns[0].ndv, 3);
        assert_eq!(s.columns[0].null_count, 1);
        assert_eq!(s.columns[0].min, Some(ScalarValue::Int64(Some(1))));
        assert_eq!(s.columns[0].max, Some(ScalarValue::Int64(Some(9))));
        assert_eq!(s.columns[1].ndv, 3);
        assert_eq!(s.columns[1].max, Some(ScalarValue::Utf8(Some("z".into()))));
        assert_eq!(s.columns[2].ndv, 3);
        assert_eq!(s.columns[2].min, Some(ScalarValue::Float64(Some(-1.0))));
    }

    #[test]
    fn schema_normalization() {
        let s = Schema::new(vec![
            Field::new("ID", ArrowType::Int32, false),
            Field::new("x", ArrowType::Float32, false),
        ]);
        let n = normalize_schema(&s).unwrap();
        assert_eq!(n.field(0).name(), "id");
        assert_eq!(n.field(0).data_type(), &ArrowType::Int64);
        assert_eq!(n.field(1).data_type(), &ArrowType::Float64);
        let dup = Schema::new(vec![
            Field::new("a", ArrowType::Int64, true),
            Field::new("A", ArrowType::Int64, true),
        ]);
        assert!(normalize_schema(&dup).is_err());
        let bad = Schema::new(vec![Field::new("b", ArrowType::Binary, true)]);
        assert!(normalize_schema(&bad).is_err());
    }

    #[test]
    fn catalog_is_case_insensitive() {
        let mut cat = Catalog::new();
        let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int64, true)]));
        cat.insert(TableEntry {
            name: "t".into(),
            schema: schema.clone(),
            source: TableSource::Memory(vec![]),
            stats: StatsCollector::new(&schema).finish(),
        });
        assert!(cat.get("T").is_some());
        assert_eq!(cat.table_names(), vec!["t".to_string()]);
        assert!(cat.deregister("t"));
        assert!(cat.get("t").is_none());
    }
}
