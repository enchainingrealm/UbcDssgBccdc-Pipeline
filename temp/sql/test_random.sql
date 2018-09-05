SELECT R.test_key, R.result_key, R.result_full_description, MM.candidates
FROM dbo.tmp_random R, dbo.metamap MM
WHERE R.test_key = MM.test_key
    AND R.result_key = MM.result_key
