CREATE TABLE IF NOT EXISTS audio_diarizations (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL UNIQUE REFERENCES audio_jobs(id) ON DELETE CASCADE,
    input_asset_id uuid NOT NULL REFERENCES audio_assets(id) ON DELETE CASCADE,
    backend text NOT NULL,
    model text NOT NULL,
    exclusive boolean NOT NULL,
    speaker_count integer NOT NULL CHECK (speaker_count >= 0),
    diarization_uri text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_diarizations_backend_idx
    ON audio_diarizations (backend, model);
