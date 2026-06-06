-- Top supplier (the view becomes a CTE referenced twice)
WITH revenue AS (
  SELECT l_suppkey AS supplier_no, sum(l_extendedprice * (1 - l_discount)) AS total_revenue
  FROM lineitem
  WHERE l_shipdate >= DATE '1996-01-01'
    AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
  GROUP BY l_suppkey)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue, (SELECT max(total_revenue) AS max_revenue FROM revenue) AS m
WHERE s_suppkey = supplier_no AND total_revenue = max_revenue
ORDER BY s_suppkey
