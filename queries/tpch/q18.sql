-- Large volume customer (IN subquery rewritten as a join; threshold lowered for
-- the generator's 1-7 lines per order)
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) AS total_qty
FROM customer, orders, lineitem,
     (SELECT l_orderkey AS big_orderkey FROM lineitem
      GROUP BY l_orderkey HAVING sum(l_quantity) > 250) AS big
WHERE o_orderkey = big_orderkey AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate, o_orderkey
LIMIT 100
