import psycopg2
import bcrypt
import json
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "medical_db")
DB_USER = os.getenv("DB_USER", "medical_admin")
DB_PASS = os.getenv("DB_PASS", "default_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def create_connection():
    """Creates and returns a new connection to the database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ DATABASE CONNECTION FAILED: {e}")
        return None

def add_patient(conn, name, username, password, history=None):
    """Adds a new patient to the 'patients' table."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    sql = """
        INSERT INTO patients (name, username, password_hash, patient_history)
        VALUES (%s, %s, %s, %s) 
        RETURNING id;
    """
    patient_id = None
    try:
        with conn.cursor() as cur:
            history_json = json.dumps(history) if history else None
            cur.execute(sql, (name, username, hashed_password.decode('utf-8'), history_json))
            patient_id = cur.fetchone()[0]
            conn.commit()
            print(f"✅ Patient '{name}' added successfully to DB with ID: {patient_id}")
    except psycopg2.Error as e:
        print(f"❌ Error in add_patient: {e}")
        conn.rollback()
        raise e
    return patient_id

def initialize_database():
    """Connects to the DB and creates the 'patients' table if it doesn't exist."""
    conn = create_connection()
    if not conn:
        print("Aborting setup due to connection failure.")
        return

    try:
        with conn.cursor() as cur:
            create_table_command = """
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                patient_history JSONB
            );
            """
            cur.execute(create_table_command)
            conn.commit()
            print("✅ 'patients' table created or already exists.")

    except Exception as e:
        print(f"❌ An error occurred during initialization: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("--- Running Database Initial Setup ---")
    initialize_database()
    print("--- Setup Complete ---")