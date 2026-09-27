CREATE TABLE IF NOT EXISTS audio_segments (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES audio_jobs(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES audio_assets(id) ON DELETE CASCADE,
    slice_asset_id uuid REFERENCES audio_assets(id) ON DELETE SET NULL,
    start_ms bigint NOT NULL CHECK (start_ms >= 0),
    end_ms bigint NOT NULL CHECK (end_ms > start_ms),
    speaker text,
    language text,
    transcript text,
    confidence double precision CHECK (confidence >= 0 AND confidence <= 1),
    labels jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_segments_job_timeline_idx
    ON audio_segments (job_id, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS audio_segments_asset_idx ON audio_segments (asset_id);
