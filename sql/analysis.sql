
SELECT
    COUNT(*) AS total_records,
    ROUND(AVG(target_subscribed) * 100.0, 2) AS subscription_rate_percent
FROM bank_marketing;

SELECT
    job,
    COUNT(*) AS customers,
    ROUND(AVG(target_subscribed) * 100.0, 2) AS subscription_rate_percent
FROM bank_marketing
GROUP BY job
ORDER BY subscription_rate_percent DESC;

WITH job_rates AS (
    SELECT
        job,
        COUNT(*) AS customers,
        AVG(target_subscribed::numeric) AS subscription_rate
    FROM bank_marketing
    GROUP BY job
)
SELECT
    job,
    customers,
    ROUND(100 * subscription_rate, 2) AS rate_percent,
    DENSE_RANK() OVER (
        ORDER BY subscription_rate DESC
    ) AS conversion_rank
FROM job_rates
ORDER BY conversion_rank;

SELECT
    record_id,
    age,
    job,
    balance,
    campaign,
    previous,
    target_subscribed
FROM bank_marketing
ORDER BY record_id
LIMIT 10;
