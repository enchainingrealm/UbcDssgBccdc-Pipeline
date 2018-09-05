SELECT C.test_key, C.result_key, C.result_full_description, MM.candidates
FROM dbo.tmp_culture C, dbo.metamap MM
WHERE C.test_key = MM.test_key
    AND C.result_key = MM.result_key
