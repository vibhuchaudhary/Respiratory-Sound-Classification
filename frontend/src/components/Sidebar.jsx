import React from 'react';
import { NavLink } from 'react-router-dom';
import '../css/Sidebar.css';
import { 
    FiMessageSquare, 
    FiUser, 
    FiLogOut, 
    FiEdit2
} from 'react-icons/fi';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const Sidebar = ({ user, onLogout }) => {
    if (!user) {
        return null; 
    }

    // Construct full avatar URL
    const getAvatarUrl = () => {
        if (!user.avatar) {
            return "https://via.placeholder.com/150/4A90E2/FFFFFF?text=Patient";
        }
        
        // If avatar starts with http, it's already a full URL (Google OAuth)
        if (user.avatar.startsWith('http')) {
            return user.avatar;
        }
        
        // Otherwise, prepend the API base URL (local uploads)
        return `${API_BASE_URL}${user.avatar}`;
    };

    return (
        <div className="sidebar-container">
            <div className="profile-section">
                <img 
                    src={getAvatarUrl()} 
                    alt="Profile" 
                    className="profile-picture"
                    onError={(e) => {
                        e.target.src = "https://via.placeholder.com/150/4A90E2/FFFFFF?text=Patient";
                    }}
                />
                <h4 className="profile-username">@{user.username}</h4>
                {user.name && <p className="profile-name">{user.name}</p>}
            </div>
            
            <nav className="navigation-menu">
                <NavLink to="/chat" className="nav-item">
                    <FiMessageSquare />
                    <span>Chat</span>
                </NavLink>
                <NavLink to="/health-profile" className="nav-item">
                    <FiUser />
                    <span>Health Profile</span>
                </NavLink>
                <NavLink to="/edit-profile" className="nav-item">
                    <FiEdit2 />
                    <span>Edit Profile</span>
                </NavLink>
            </nav>

            <div className="sidebar-footer">
                <button className="logout-button" onClick={onLogout}>
                    <FiLogOut />
                    <span>Logout</span>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;