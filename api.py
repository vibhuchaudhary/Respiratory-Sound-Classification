from flask import Flask, request, jsonify
from flask_cors import CORS
import database
import psycopg2
import sys

app = Flask(__name__)
CORS(app)

def check_database_readiness():
    """Checks if the DB is connected and if the patients table exists."""
    print("--- Checking database readiness ---")
    conn = database.create_connection()
    if not conn:
        print("\nFATAL: Could not connect to the database.")
        print("Please check your credentials in database.py and ensure the PostgreSQL server is running.")
        sys.exit(1)
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM patients LIMIT 1;")
        print("Database is ready.")
    except psycopg2.errors.UndefinedTable:
        print("\nFATAL: The 'patients' table was not found in the database.")
        print("SOLUTION: Please run the setup script once by typing this in your terminal:")
        print("          python database.py")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

@app.route('/register', methods=['POST'])
def register_patient():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')
    history = data.get('history', '')

    if not all([name, username, password]):
        return jsonify({"error": "Name, username, and password are required"}), 400

    conn = database.create_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        patient_id = database.add_patient(conn, name, username, password, {"details": history})
        return jsonify({"success": True, "message": "Patient registered successfully", "id": patient_id}), 201
    except psycopg2.Error as e:
        if e.pgcode == '23505':
            return jsonify({"error": f"Username '{username}' already exists."}), 409
        print(f"Database Error: {e}")
        return jsonify({"error": "A database error occurred."}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    check_database_readiness()
    print("--- Starting Flask API server ---")
    app.run(debug=True, port=5000)