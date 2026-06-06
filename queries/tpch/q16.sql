-- Parts/supplier relationship (NOT IN subquery rewritten as a LEFT JOIN anti-join)
SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp
JOIN part ON p_partkey = ps_partkey
LEFT JOIN (SELECT s_suppkey AS bad_suppkey FROM supplier
           WHERE s_comment LIKE '%Customer%Complaints%') AS bad
  ON ps_suppkey = bad_suppkey
WHERE p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%'
  AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
  AND bad_suppkey IS NULL
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
