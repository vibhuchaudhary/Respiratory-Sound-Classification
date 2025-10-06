import React, { useState } from 'react';
import '../css/AuthPage.css';
import { FiUser } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const AuthPage = ({ onLoginSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [name, setName] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [history, setHistory] = useState('');
    const [avatarFile, setAvatarFile] = useState(null);
    const [avatarPreview, setAvatarPreview] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const navigate = useNavigate();

    const handleAvatarChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setAvatarFile(file);
            setAvatarPreview(URL.createObjectURL(file));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (isLogin) {
            // --- UPDATED: Real login logic ---
            try {
                const response = await fetch('http://127.0.0.1:5000/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password }),
                });

                const data = await response.json();

                if (response.ok) {
                    onLoginSuccess(data.user); // Pass the real user data
                    navigate('/chat');
                } else {
                    setError(data.error || 'Login failed.');
                }
            } catch (err) {
                setError('Could not connect to the server. Is the API running?');
            }
        } else {
            // Registration logic remains the same
            const formData = new FormData();
            formData.append('name', name);
            formData.append('username', username);
            formData.append('password', password);
            formData.append('history', history);
            if (avatarFile) {
                formData.append('avatar', avatarFile);
            }

            try {
                const response = await fetch('http://127.0.0.1:5000/register', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    setSuccess('Registration successful! Please switch to login.');
                    setIsLogin(true);
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
                
                {!isLogin && (
                    <div className="avatar-upload-section">
                        <label htmlFor="avatar-input" className="avatar-uploader">
                            {avatarPreview ? (
                                <img src={avatarPreview} alt="Avatar Preview" className="avatar-preview" />
                            ) : (
                                <div className="avatar-placeholder">
                                    <FiUser />
                                    <span>Add Photo</span>
                                </div>
                            )}
                        </label>
                        <input id="avatar-input" type="file" accept="image/*" onChange={handleAvatarChange} />
                    </div>
                )}

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