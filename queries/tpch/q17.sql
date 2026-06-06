-- Small-quantity-order revenue (correlated AVG rewritten as a grouped derived table)
SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part,
     (SELECT l_partkey AS agg_partkey, 0.2 * avg(l_quantity) AS avg_quantity
      FROM lineitem GROUP BY l_partkey) AS part_agg
WHERE p_partkey = l_partkey AND agg_partkey = l_partkey
  AND p_brand = 'Brand#23' AND p_container = 'MED BOX'
  AND l_quantity < avg_quantity
