import React, { useState } from 'react';
import '../css/AuthPage.css';

const AuthPage = ({ onLoginSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [name, setName] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [history, setHistory] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (isLogin) {
            // --- Mock Login ---
            // In a real app, you would call a /login API endpoint here.
            // For now, we'll just log in successfully to show the UI flow.
            if (username && password) {
                onLoginSuccess();
            } else {
                setError("Please enter username and password.");
            }
        } else {
            // --- Registration ---
            try {
                const response = await fetch('http://127.0.0.1:5000/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, username, password, history }),
                });

                const data = await response.json();

                if (response.ok) {
                    setSuccess('Registration successful! Please switch to login.');
                    setIsLogin(true); // Switch to login form
                } else {
                    setError(data.error || 'Registration failed.');
                }
            } catch (err) {
                setError('Could not connect to the server. Is the API running?');
            }
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-form-wrapper">
                <h2>{isLogin ? 'Patient Login' : 'Patient Registration'}</h2>
                {error && <p className="message error">{error}</p>}
                {success && <p className="message success">{success}</p>}
                <form onSubmit={handleSubmit}>
                    {!isLogin && (
                        <div className="input-group">
                            <label>Patient Name</label>
                            <input type="text" value={name} onChange={(e) => setName(e.target.value)} required />
                        </div>
                    )}
                    <div className="input-group">
                        <label>Username</label>
                        <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
                    </div>
                    <div className="input-group">
                        <label>Password</label>
                        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
                    </div>
                    {!isLogin && (
                        <div className="input-group">
                            <label>Brief Medical History</label>
                            <textarea value={history} onChange={(e) => setHistory(e.target.value)} />
                        </div>
                    )}
                    <button type="submit" className="auth-button">
                        {isLogin ? 'Login' : 'Register'}
                    </button>
                </form>
                <p className="toggle-text">
                    {isLogin ? "Don't have an account?" : 'Already have an account?'}
                    <button onClick={() => setIsLogin(!isLogin)} className="toggle-button">
                        {isLogin ? 'Register' : 'Login'}
                    </button>
                </p>
            </div>
        </div>
    );
};

export default AuthPage;