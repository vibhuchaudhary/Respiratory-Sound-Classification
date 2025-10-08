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
                    const response = await fetch(`http://localhost:8000/api/patient/${user.username}`);
                    
                    if (!response.ok) {
                        if (response.status === 404) {
                            throw new Error('Patient profile not found.');
                        }
                        throw new Error('Could not fetch health data.');
                    }
                    
                    const data = await response.json();
                    setHealthData(data);
                } catch (err) {
                    setError(err.message);
                    console.error('Health profile fetch error:', err);
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
        return (
            <div className="profile-page-container">
                <p>No user data available. Please log in.</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="profile-page-container">
                <div className="loading-spinner">
                    <p>Loading profile...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="profile-page-container">
                <div className="error-message">
                    <h3>⚠️ Error</h3>
                    <p>{error}</p>
                    <button onClick={() => window.location.reload()}>Retry</button>
                </div>
            </div>
        );
    }

    if (!healthData) {
        return (
            <div className="profile-page-container">
                <p>No health data found.</p>
            </div>
        );
    }

    return (
        <div className="profile-page-container">
            <div className="profile-header">
                <h1> <span> 🩺 </span> Health Profile</h1>
                <p className="patient-id">Patient ID: {healthData.patient_id}</p>
            </div>

            <div className="profile-content">
                {/* Basic Information Card */}
                <div className="profile-card">
                    <h3>📋 Basic Information</h3>
                    <div className="info-grid">
                        <div className="info-item">
                            <span className="info-label">Age Range:</span>
                            <span className="info-value">{healthData.age_range || 'Not specified'}</span>
                        </div>
                        <div className="info-item">
                            <span className="info-label">Gender:</span>
                            <span className="info-value">{healthData.gender || 'Not specified'}</span>
                        </div>
                        <div className="info-item">
                            <span className="info-label">Smoking Status:</span>
                            <span className="info-value">{healthData.smoking_status || 'Not specified'}</span>
                        </div>
                        <div className="info-item">
                            <span className="info-label">Previous Respiratory Infections:</span>
                            <span className="info-value">
                                {healthData.previous_respiratory_infections !== null 
                                    ? healthData.previous_respiratory_infections 
                                    : 'Not specified'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Comorbidities Card */}
                <div className="profile-card">
                    <h3>🩺 Comorbidities</h3>
                    {healthData.comorbidities && healthData.comorbidities.length > 0 ? (
                        <ul className="comorbidities-list">
                            {healthData.comorbidities.map((condition, index) => (
                                <li key={index} className="comorbidity-item">
                                    <span className="bullet">•</span> {condition}
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="no-data">No comorbidities reported</p>
                    )}
                </div>

                {/* Medications Card */}
                <div className="profile-card">
                    <h3>💊 Current Medications</h3>
                    {healthData.current_medications && healthData.current_medications.trim() ? (
                        <p className="medications-text">{healthData.current_medications}</p>
                    ) : (
                        <p className="no-data">No current medications reported</p>
                    )}
                </div>

                {/* Allergies Card */}
                <div className="profile-card">
                    <h3>⚠️ Allergies</h3>
                    {healthData.allergies && healthData.allergies.trim() ? (
                        <p className="allergies-text">{healthData.allergies}</p>
                    ) : (
                        <p className="no-data">No known allergies</p>
                    )}
                </div>

                {/* Information Note */}
                <div className="profile-card info-note">
                    <p>
                        ℹ️ This information is used to provide personalized health recommendations. 
                        Keep your profile updated for the best experience.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default HealthProfile;