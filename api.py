from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import database
import psycopg2
import sys
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app)

# --- Configuration for file uploads ---
UPLOAD_FOLDER = 'uploads/avatars'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/avatars/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- NEW: Login Endpoint ---
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"error": "Username and password are required"}), 400

    conn = database.create_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # Step 1: Verify the password
        if database.verify_password(conn, username, password):
            # Step 2: If valid, fetch the user's profile data
            patient_data = database.get_patient_by_username(conn, username)
            if patient_data:
                # Construct the full avatar URL
                if patient_data.get('avatar'):
                    patient_data['avatar'] = f"http://127.0.0.1:5000/uploads/avatars/{patient_data['avatar']}"
                
                return jsonify({"success": True, "user": patient_data}), 200
            else:
                # Should not happen if password verified, but good practice
                return jsonify({"error": "User not found after verification"}), 404
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        print(f"Server Error during login: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/register', methods=['POST'])
def register_patient():
    name = request.form.get('name')
    username = request.form.get('username')
    password = request.form.get('password')
    history = request.form.get('history', '')
    
    avatar_filename = None
    if 'avatar' in request.files:
        file = request.files['avatar']
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{ext}"
            filename = secure_filename(unique_filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            avatar_filename = filename

    if not all([name, username, password]):
        return jsonify({"error": "Name, username, and password are required"}), 400

    conn = database.create_connection()
    try:
        patient_id = database.add_patient(conn, name, username, password, {"details": history}, avatar_filename)
        return jsonify({"success": True, "message": "Patient registered successfully", "id": patient_id}), 201
    except psycopg2.Error as e:
        if e.pgcode == '23505':
            return jsonify({"error": f"Username '{username}' already exists."}), 409
        return jsonify({"error": "A database error occurred."}), 500
    finally:
        if conn:
            conn.close()

@app.route('/profile/<string:username>', methods=['GET', 'PUT'])
def profile(username):
    conn = database.create_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        if request.method == 'GET':
            patient_data = database.get_patient_by_username(conn, username)
            if patient_data:
                if patient_data.get('avatar'):
                    patient_data['avatar'] = f"http://127.0.0.1:5000/uploads/avatars/{patient_data['avatar']}"
                return jsonify(patient_data), 200
            else:
                return jsonify({"error": "Patient not found"}), 404
        
        if request.method == 'PUT':
            name = request.form.get('name')
            avatar_filename = None
            if 'avatar' in request.files:
                file = request.files['avatar']
                if file and allowed_file(file.filename):
                    ext = file.filename.rsplit('.', 1)[1].lower()
                    unique_filename = f"{uuid.uuid4()}.{ext}"
                    filename = secure_filename(unique_filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    avatar_filename = filename

            updated_user = database.update_patient_profile(conn, username, name=name, avatar=avatar_filename)
            if updated_user:
                full_user_data = database.get_patient_by_username(conn, username)
                if full_user_data.get('avatar'):
                     full_user_data['avatar'] = f"http://127.0.0.1:5000/uploads/avatars/{full_user_data['avatar']}"
                return jsonify({"success": True, "user": full_user_data}), 200
            else:
                return jsonify({"error": "Update failed or user not found"}), 404
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    print("--- Starting Flask API server ---")
    app.run(debug=True, port=5000)