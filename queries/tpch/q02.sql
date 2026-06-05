-- Minimum cost supplier (correlated subquery rewritten as a derived table)
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region,
     (SELECT ps_partkey AS mc_partkey, min(ps_supplycost) AS min_cost
      FROM partsupp, supplier, nation, region
      WHERE s_suppkey = ps_suppkey AND s_nationkey = n_nationkey
        AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
      GROUP BY ps_partkey) AS mc
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
  AND p_size = 15 AND p_type LIKE '%BRASS'
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
  AND mc_partkey = p_partkey AND ps_supplycost = min_cost
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100
