import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AuthPage from "./components/AuthPage";
import Sidebar from "./components/Sidebar";
import Chatbot from "./components/Chatbot";
import HealthProfile from "./components/HealthProfile";
import EditProfile from "./components/EditProfile";
import "./css/App.css";

// Wrapper to protect routes that need authentication
const ProtectedRoute = ({ isAuthenticated, children }) => {
    if (!isAuthenticated) {
        return <Navigate to="/login" />;
    }
    return children;
};

// Layout with sidebar for authenticated pages
const MainLayout = ({ user, onLogout, onUserUpdate, children }) => (
    <div className="app-container">
        <Sidebar user={user} onLogout={onLogout} />
        {React.cloneElement(children, { onUserUpdate })}
    </div>
);

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);

    const handleLoginSuccess = (userData) => {
        setCurrentUser(userData);
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        setCurrentUser(null);
        setIsAuthenticated(false);
    };

    // Update user data after profile changes
    const handleUserUpdate = (updatedUser) => {
        setCurrentUser(updatedUser);
    };

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/login" element={<AuthPage onLoginSuccess={handleLoginSuccess} />} />

                <Route path="/chat" element={
                    <ProtectedRoute isAuthenticated={isAuthenticated}>
                        <MainLayout user={currentUser} onLogout={handleLogout} onUserUpdate={handleUserUpdate}>
                            <Chatbot user={currentUser} />
                        </MainLayout>
                    </ProtectedRoute>
                } />
                
                <Route path="/health-profile" element={
                    <ProtectedRoute isAuthenticated={isAuthenticated}>
                        <MainLayout user={currentUser} onLogout={handleLogout} onUserUpdate={handleUserUpdate}>
                            <HealthProfile user={currentUser} />
                        </MainLayout>
                    </ProtectedRoute>
                } />
                
                <Route path="/edit-profile" element={
                    <ProtectedRoute isAuthenticated={isAuthenticated}>
                        <MainLayout user={currentUser} onLogout={handleLogout} onUserUpdate={handleUserUpdate}>
                            <EditProfile user={currentUser} />
                        </MainLayout>
                    </ProtectedRoute>
                } />
                
                <Route path="*" element={<Navigate to={isAuthenticated ? "/chat" : "/login"} />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;