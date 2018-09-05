SELECT *
FROM dbo.predictions
WHERE test_performed_pred = 'no'
    AND (test_outcome_pred = 'positive'
        OR test_outcome_pred = 'negative')
