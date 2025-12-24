-- Migration: Add evaluation tables for Market-1501 dataset evaluation
-- Run this after the existing pipeline_jobs and pipeline_configs tables are created

-- Dataset management table
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL DEFAULT 'market1501',
    upload_path TEXT NOT NULL,
    num_query INT,
    num_gallery INT,
    num_identities INT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'uploaded'
);

-- Evaluation jobs table
CREATE TABLE IF NOT EXISTS evaluation_jobs (
    eval_job_id UUID PRIMARY KEY,
    dataset_id INT REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    progress FLOAT DEFAULT 0.0,

    -- Standard ReID metrics
    map_score FLOAT,
    rank1_accuracy FLOAT,
    rank5_accuracy FLOAT,
    rank10_accuracy FLOAT,

    -- Gallery statistics (stored as JSONB)
    gallery_stats JSONB,

    -- Performance metrics (stored as JSONB)
    performance_stats JSONB,

    error_message TEXT
);

-- Per-query detailed results (optional, for detailed analysis)
CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id SERIAL PRIMARY KEY,
    eval_job_id UUID REFERENCES evaluation_jobs(eval_job_id) ON DELETE CASCADE,
    query_id VARCHAR(255) NOT NULL,
    query_person_id INT NOT NULL,
    query_camera_id INT NOT NULL,

    -- Gallery matching results
    matched_person_id INT,
    gallery_decision VARCHAR(20),
    similarity_score FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_eval_jobs_status ON evaluation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_eval_jobs_dataset ON evaluation_jobs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_job ON evaluation_results(eval_job_id);

-- Comments for documentation
COMMENT ON TABLE datasets IS 'Uploaded ReID datasets (Market-1501, etc.)';
COMMENT ON TABLE evaluation_jobs IS 'Evaluation jobs with metrics and statistics';
COMMENT ON TABLE evaluation_results IS 'Per-query detailed results (optional)';
