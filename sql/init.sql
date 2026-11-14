
CREATE TABLE IF NOT EXISTS bank_marketing (
    record_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    age INTEGER NOT NULL,
    job TEXT NOT NULL,
    marital TEXT NOT NULL,
    education TEXT NOT NULL,
    default_status TEXT NOT NULL,
    balance INTEGER NOT NULL,
    housing TEXT NOT NULL,
    loan TEXT NOT NULL,
    contact TEXT NOT NULL,
    day INTEGER NOT NULL,
    month TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL,
    campaign INTEGER NOT NULL,
    pdays INTEGER NOT NULL,
    previous INTEGER NOT NULL,
    poutcome TEXT NOT NULL,
    target_subscribed SMALLINT NOT NULL CHECK (target_subscribed IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_bank_target
ON bank_marketing (target_subscribed);

CREATE INDEX IF NOT EXISTS idx_bank_job
ON bank_marketing (job);
