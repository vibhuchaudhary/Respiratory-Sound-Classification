import React, { useState } from 'react';
import './css/App.css';
import Chatbot from './components/Chatbot';
import Sidebar from './components/Sidebar';
import AuthPage from './components/AuthPage';

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    const handleLoginSuccess = () => {
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        setIsAuthenticated(false);
    };

    return (
        <>
            {isAuthenticated ? (
                <div className="app-container">
                    <Sidebar onLogout={handleLogout} />
                    <Chatbot />
                </div>
            ) : (
                <AuthPage onLoginSuccess={handleLoginSuccess} />
            )}
        </>
    );
}

export default App;