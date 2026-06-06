-- Suppliers who kept orders waiting (EXISTS / NOT EXISTS rewritten with
-- per-order supplier counts)
SELECT s_name, count(*) AS numwait
FROM supplier, lineitem l1, orders, nation,
     (SELECT l_orderkey AS m_orderkey, count(DISTINCT l_suppkey) AS nsupp
      FROM lineitem GROUP BY l_orderkey) AS multi,
     (SELECT l_orderkey AS late_orderkey, count(DISTINCT l_suppkey) AS nlate
      FROM lineitem WHERE l_receiptdate > l_commitdate GROUP BY l_orderkey) AS late
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F'
  AND l1.l_receiptdate > l1.l_commitdate
  AND m_orderkey = l1.l_orderkey AND nsupp > 1
  AND late_orderkey = l1.l_orderkey AND nlate = 1
  AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100
