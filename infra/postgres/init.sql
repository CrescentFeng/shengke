CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS audio_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    input_uri text NOT NULL,
    operations jsonb NOT NULL,
    policy_id text NOT NULL,
    idempotency_key text NOT NULL UNIQUE,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    state text NOT NULL CHECK (state IN (
        'pending', 'validating', 'queued', 'running', 'succeeded', 'failed', 'cancelled'
    )),
    progress double precision NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    error_code text,
    error_message text,
    locked_at timestamptz,
    locked_by text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_jobs_queue_idx
    ON audio_jobs (created_at)
    WHERE state = 'queued';

CREATE TABLE IF NOT EXISTS job_events (
    id bigserial PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES audio_jobs(id) ON DELETE CASCADE,
    state text NOT NULL,
    message text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events (job_id, id);

CREATE TABLE IF NOT EXISTS audio_assets (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES audio_jobs(id) ON DELETE CASCADE,
    role text NOT NULL,
    uri text NOT NULL,
    filename text,
    sha256 char(64),
    size_bytes bigint CHECK (size_bytes >= 0),
    probe jsonb,
    source_asset_id uuid REFERENCES audio_assets(id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_assets_job_idx ON audio_assets (job_id);
CREATE INDEX IF NOT EXISTS audio_assets_sha_idx ON audio_assets (sha256);

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

CREATE TABLE IF NOT EXISTS tool_runs (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES audio_jobs(id) ON DELETE CASCADE,
    tool_name text NOT NULL,
    tool_version text NOT NULL,
    operation text NOT NULL,
    status text NOT NULL,
    input_asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    output_asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz,
    finished_at timestamptz,
    error_code text,
    error_message text
);

CREATE INDEX IF NOT EXISTS tool_runs_job_idx ON tool_runs (job_id);

CREATE TABLE IF NOT EXISTS audio_reports (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL UNIQUE REFERENCES audio_jobs(id) ON DELETE CASCADE,
    input_asset_id uuid NOT NULL REFERENCES audio_assets(id),
    processing_chain_id text NOT NULL,
    report_uri text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
