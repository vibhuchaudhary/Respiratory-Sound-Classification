import React from 'react';
import { NavLink } from 'react-router-dom';
import '../css/Sidebar.css';
import { FiMessageSquare, FiUser, FiLogOut, FiEdit2 } from 'react-icons/fi';
import { getAvatarUrl } from '../config/api';

const Sidebar = ({ user, onLogout }) => {
    if (!user) {
        return null; 
    }

    return (
        <div className="sidebar-container">
            <div className="profile-section">
                <img 
                    src={getAvatarUrl(user.avatar)} 
                    alt="Profile" 
                    className="profile-picture"
                    onError={(e) => {
                        e.target.src = "https://ui-avatars.com/api/?name=Patient&size=150&background=4A90E2&color=ffffff";
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