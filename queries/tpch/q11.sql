-- Important stock identification (HAVING > scalar subquery rewritten as a join
-- against a one-row derived table; threshold fraction scaled for SF 0.01)
SELECT ps_partkey, value
FROM (SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
      FROM partsupp, supplier, nation
      WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
      GROUP BY ps_partkey) AS v,
     (SELECT sum(ps_supplycost * ps_availqty) * 0.001 AS threshold
      FROM partsupp, supplier, nation
      WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY') AS t
WHERE value > threshold
ORDER BY value DESC, ps_partkey
