-- Order priority checking (EXISTS rewritten as a join with a DISTINCT derived table)
SELECT o_orderpriority, count(*) AS order_count
FROM orders,
     (SELECT DISTINCT l_orderkey FROM lineitem WHERE l_commitdate < l_receiptdate) AS late
WHERE o_orderkey = late.l_orderkey
  AND o_orderdate >= DATE '1993-07-01'
  AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH
GROUP BY o_orderpriority
ORDER BY o_orderpriority
