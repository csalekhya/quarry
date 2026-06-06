-- Global sales opportunity (NOT EXISTS rewritten as LEFT JOIN ... IS NULL)
SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
      FROM customer LEFT JOIN orders ON o_custkey = c_custkey,
           (SELECT avg(c_acctbal) AS avg_bal FROM customer
            WHERE c_acctbal > 0.00
              AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')) AS a
      WHERE substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND c_acctbal > avg_bal AND o_orderkey IS NULL) AS custsale
GROUP BY cntrycode
ORDER BY cntrycode
