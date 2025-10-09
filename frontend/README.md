# A.I.R.A. - Frontend

This is the React user interface for the A.I.R.A. application, built with Vite. It allows patients to register, log in, manage their profile, and interact with the AI chat companion.

---

## Prerequisites

Make sure you have the following installed:
* Node.js (v18 or newer)
* npm (or your preferred package manager like yarn/pnpm)

---

## Setup Instructions

1.  **Navigate to the Frontend Directory**
    From the project root, move into this directory:
    ```bash
    cd frontend
    ```

2.  **Install Dependencies**
    This will install React and all other necessary packages.
    ```bash
    npm install
    ```

3.  **Set Up Environment Variables**
    The frontend needs to know where the backend API is running. Create a `.env` file in this directory.
    ```bash
    touch .env
    ```
    Open the newly created `.env` file and add the following line. This points the app to the default address for the FastAPI backend.

    ```env
    VITE_API_BASE_URL=http://localhost:8000
    ```

4.  **Run the Development Server**
    You're ready to go. This command starts the Vite development server.
    ```bash
    npm run dev
    ```
    The application will be available in your browser, usually at `http://localhost:5173`.

