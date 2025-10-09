CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(255) PRIMARY KEY,
    age_range VARCHAR(20),
    gender VARCHAR(20),
    smoking_status VARCHAR(50),
    has_hypertension BOOLEAN DEFAULT FALSE,
    has_diabetes BOOLEAN DEFAULT FALSE,
    has_asthma_history BOOLEAN DEFAULT FALSE,
    previous_respiratory_infections INTEGER DEFAULT 0,
    current_medications TEXT,
    allergies TEXT,
    last_consultation_date TIMESTAMP,
    avatar TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patient medical history
CREATE TABLE IF NOT EXISTS medical_history (
    history_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255) REFERENCES patients(patient_id) ON DELETE CASCADE,
    visit_date TIMESTAMP,
    diagnosis TEXT,
    symptoms TEXT,
    treatment_prescribed TEXT,
    follow_up_required BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit trail for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255),
    query_timestamp TIMESTAMP,
    query_text TEXT,
    response_text TEXT,
    audio_analysis_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_patient_history ON medical_history(patient_id);
CREATE INDEX IF NOT EXISTS idx_visit_date ON medical_history(visit_date);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(query_timestamp);