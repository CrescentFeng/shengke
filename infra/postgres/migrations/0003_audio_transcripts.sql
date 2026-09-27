CREATE TABLE IF NOT EXISTS audio_transcripts (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL UNIQUE REFERENCES audio_jobs(id) ON DELETE CASCADE,
    backend text NOT NULL,
    model text NOT NULL,
    language text,
    transcript_uri text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_transcripts_language_idx ON audio_transcripts (language);
