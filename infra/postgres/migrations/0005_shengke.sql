CREATE TABLE IF NOT EXISTS tts_voices (
    id uuid PRIMARY KEY,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tts_evaluations (
    id uuid PRIMARY KEY,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tts_scores (
    evaluation_id uuid NOT NULL REFERENCES tts_evaluations(id),
    listener_id text NOT NULL,
    sample_id uuid NOT NULL,
    payload jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (evaluation_id, listener_id, sample_id)
);
