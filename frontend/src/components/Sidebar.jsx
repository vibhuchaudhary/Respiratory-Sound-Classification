import React from 'react';
import { NavLink } from 'react-router-dom';
import '../css/Sidebar.css';
import { 
    FiMessageSquare, 
    FiUser, 
    FiLogOut, 
    FiUpload,
    FiEdit2
} from 'react-icons/fi';

const Sidebar = ({ user, onLogout }) => {
    if (!user) {
        return null; 
    }

    return (
        <div className="sidebar-container">
            <div className="profile-section">
                <img src={user.avatar} alt="Profile" className="profile-picture" />
                <h3 className="profile-name">{user.name}</h3>
                <span className="profile-username">@{user.username}</span>
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

            <div className="upload-section">
                <button className="upload-button"><FiUpload /></button>
            </div>

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