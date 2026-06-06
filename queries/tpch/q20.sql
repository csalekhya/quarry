-- Potential part promotion (nested IN subqueries rewritten as derived tables)
SELECT s_name, s_address
FROM supplier, nation,
     (SELECT DISTINCT ps_suppkey AS avail_suppkey
      FROM partsupp,
           (SELECT p_partkey AS forest_partkey FROM part WHERE p_name LIKE 'forest%') AS fp,
           (SELECT l_partkey AS agg_partkey, l_suppkey AS agg_suppkey, 0.5 * sum(l_quantity) AS half_qty
            FROM lineitem
            WHERE l_shipdate >= DATE '1994-01-01'
              AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
            GROUP BY l_partkey, l_suppkey) AS agg
      WHERE ps_partkey = forest_partkey AND agg_partkey = ps_partkey
        AND agg_suppkey = ps_suppkey AND ps_availqty > half_qty) AS avail
WHERE s_suppkey = avail_suppkey AND s_nationkey = n_nationkey AND n_name = 'CANADA'
ORDER BY s_name
