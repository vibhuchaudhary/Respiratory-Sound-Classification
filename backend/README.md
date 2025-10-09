# A.I.R.A. - Backend

This is the FastAPI server that powers the A.I.R.A. application. It handles user authentication, patient data management, audio classification, and communication with the OpenAI API.

---

## Prerequisites

Before you begin, make sure you have the following installed:
* Python 3.9+ and Pip
* A running PostgreSQL database instance

---

## Setup Instructions

1.  **Navigate to the Backend Directory**
    From the project root, move into this directory:
    ```bash
    cd backend
    ```

2.  **Create and Activate a Virtual Environment**
    It's highly recommended to use a virtual environment.
    ```bash
    # Create the environment
    python3 -m venv venv

    # Activate it (macOS/Linux)
    source venv/bin/activate

    # Activate it (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in this directory by copying the example file.
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file and fill in your specific credentials. It will look like this:

    ```env
    # --- PostgreSQL Database ---
    POSTGRES_USER=your_db_user
    POSTGRES_PASSWORD=your_db_password
    POSTGRES_DB=aira_db
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432

    # --- API Keys ---
    OPENAI_API_KEY="sk-..."

    # --- Authentication ---
    JWT_SECRET_KEY="a-very-strong-and-secret-key-for-jwt"
    ```

5.  **Database Setup**
    Make sure your PostgreSQL server is running and that the database specified in your `.env` file exists. You will need to manually run the SQL scripts to create the `patients`, `medical_history`, and `audit_log` tables.

6.  **Run the Server**
    You're all set! Fire up the development server.
    ```bash
    uvicorn main:app --reload
    ```
    The API will be live at `http://localhost:8000`.

---

## API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, you can access it at:

* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`