import React from 'react';
import '../css/Sidebar.css';
import { FiMessageSquare, FiClock, FiSettings, FiLogOut } from 'react-icons/fi';

const Sidebar = ({ onLogout }) => {
    return (
        <div className="sidebar-container">
            <div className="profile-section">
                <img 
                    src="https://via.placeholder.com/60"
                    alt="Profile" 
                    className="profile-picture" 
                />
                <h3 className="profile-name">User Name</h3>
            </div>
            
            <nav className="navigation-menu">
                <a href="#" className="nav-item active">
                    <FiMessageSquare />
                    <span>Chat</span>
                </a>
                <a href="#" className="nav-item">
                    <FiClock />
                    <span>History</span>
                </a>
                <a href="#" className="nav-item">
                    <FiSettings />
                    <span>Settings</span>
                </a>
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