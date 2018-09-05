SELECT NIH.test_key, NIH.result_key, NIH.result_full_description, MM.candidates
FROM dbo.tmp_nih NIH, dbo.tmp_nih_metamap MM
WHERE NIH.test_key = MM.test_key
    AND NIH.result_key = MM.result_key
