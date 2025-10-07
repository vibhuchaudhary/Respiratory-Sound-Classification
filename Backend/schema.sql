CREATE TABLE patients (
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medical_history (
    history_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255) REFERENCES patients(patient_id),
    visit_date TIMESTAMP,
    diagnosis TEXT,
    symptoms TEXT,
    treatment_prescribed TEXT,
    follow_up_required BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_log (
    log_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255),
    query_timestamp TIMESTAMP,
    query_text TEXT,
    response_text TEXT,
    audio_analysis_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patient_id ON medical_history(patient_id);
CREATE INDEX idx_visit_date ON medical_history(visit_date);
CREATE INDEX idx_audit_patient ON audit_log(patient_id);
CREATE INDEX idx_audit_timestamp ON audit_log(query_timestamp);
