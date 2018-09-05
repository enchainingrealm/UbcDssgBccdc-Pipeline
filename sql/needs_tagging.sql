SELECT test_key, result_key, result_full_description
FROM lab.dim_test_result_output_v1 AS table_0
WHERE NOT EXISTS (
    SELECT DISTINCT MM.test_key, MM.result_key
    FROM dbo.metamap AS MM
    WHERE table_0.test_key = MM.test_key
        AND table_0.result_key = MM.result_key
)
