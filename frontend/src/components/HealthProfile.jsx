import React, { useState, useEffect } from 'react';
import '../css/HealthProfile.css';

const HealthProfile = ({ user }) => {
    const [healthData, setHealthData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (user && user.username) {
            const fetchHealthData = async () => {
                try {
                    const response = await fetch(`http://127.0.0.1:5000/profile/${user.username}`);
                    if (!response.ok) {
                        throw new Error('Could not fetch health data.');
                    }
                    const data = await response.json();
                    setHealthData(data);
                } catch (err) {
                    setError(err.message);
                } finally {
                    setIsLoading(false);
                }
            };
            fetchHealthData();
        } else {
            setIsLoading(false);
        }
    }, [user]);

    if (!user) {
        return <div className="profile-page-container"><p>No user data available.</p></div>;
    }

    if (isLoading) {
        return <div className="profile-page-container"><p>Loading profile...</p></div>;
    }

    if (error) {
        return <div className="profile-page-container"><p>Error: {error}</p></div>;
    }

    if (!healthData) {
        return <div className="profile-page-container"><p>No health data found.</p></div>;
    }
    
    const historyDetails = healthData.patient_history?.details || 'No detailed history available.';

    return (
        <div className="profile-page-container">
            <div className="profile-header">
                <h1>{healthData.name}'s Health Profile</h1>
                <p>@{healthData.username}</p>
            </div>
            <div className="profile-content">
                <div className="profile-card full-width">
                    <h3>Medical History Summary</h3>
                    <p>{historyDetails}</p>
                </div>
            </div>
        </div>
    );
};

export default HealthProfile;