import React, { useState, Component } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './css/App.css';

import AuthPage from './components/AuthPage';
import Sidebar from './components/Sidebar';
import Chatbot from './components/Chatbot';
import HealthProfile from './components/HealthProfile';
import EditProfile from './components/EditProfile';

const ProtectedLayout = ({ user, onLogout }) => (
    <MainLayout user={user} onLogout={onLogout}>
        {/* This Outlet component will render the matched child route */}
        <Outlet /> 
    </MainLayout>
);

// Error Boundary Component
class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '20px', color: 'red' }}>
                    <h1>Something went wrong.</h1>
                    <pre>{this.state.error?.toString()}</pre>
                </div>
            );
        }
        return this.props.children;
    }
}

const ProtectedRoute = ({ isAuthenticated, children }) => {
    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }
    return children;
};

const MainLayout = ({ user, onLogout, children }) => (
    <div className="app-container">
        <Sidebar user={user} onLogout={onLogout} />
        {children}
    </div>
);

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);

    const handleLoginSuccess = (userData) => {
        console.log('Login success:', userData);
        setCurrentUser(userData);
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        console.log('Logging out');
        setCurrentUser(null);
        setIsAuthenticated(false);
    };

    return (
        <ErrorBoundary>
            <BrowserRouter>
                <Routes>
                    <Route path="/login" element={<AuthPage onLoginSuccess={handleLoginSuccess} />} />

                    <Route path="/chat" element={
                        <ProtectedRoute isAuthenticated={isAuthenticated}>
                            <MainLayout user={currentUser} onLogout={handleLogout}>
                                <Chatbot />
                            </MainLayout>
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/health-profile" element={
                        <ProtectedRoute isAuthenticated={isAuthenticated}>
                            <MainLayout user={currentUser} onLogout={handleLogout}>
                                <HealthProfile user={currentUser} />
                            </MainLayout>
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/edit-profile" element={
                        <ProtectedRoute isAuthenticated={isAuthenticated}>
                            <MainLayout user={currentUser} onLogout={handleLogout}>
                                <EditProfile user={currentUser} />
                            </MainLayout>
                        </ProtectedRoute>
                    } />
                    <Route path="*" element={<Navigate to={isAuthenticated ? "/chat" : "/login"} replace />} />
                </Routes>
            </BrowserRouter>
        </ErrorBoundary>
    );
}

export default App;