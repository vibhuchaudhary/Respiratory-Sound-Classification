import React, { useState } from 'react';
import '../css/AuthPage.css';
import { useNavigate } from 'react-router-dom';

const AuthPage = ({ onLoginSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [ageRange, setAgeRange] = useState('');
    const [gender, setGender] = useState('');
    const [smokingStatus, setSmokingStatus] = useState('');
    const [hasHypertension, setHasHypertension] = useState(false);
    const [hasDiabetes, setHasDiabetes] = useState(false);
    const [hasAsthma, setHasAsthma] = useState(false);
    const [previousInfections, setPreviousInfections] = useState(0);
    const [medications, setMedications] = useState('');
    const [allergies, setAllergies] = useState('');
    const [lastConsultationDate, setLastConsultationDate] = useState('');
    const [avatar, setAvatar] = useState(null);
    const [avatarPreview, setAvatarPreview] = useState(null);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const navigate = useNavigate();

    const handleAvatarChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setAvatar(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setAvatarPreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const resetForm = () => {
        setUsername('');
        setAgeRange('');
        setGender('');
        setSmokingStatus('');
        setHasHypertension(false);
        setHasDiabetes(false);
        setHasAsthma(false);
        setPreviousInfections(0);
        setMedications('');
        setAllergies('');
        setLastConsultationDate('');
        setAvatar(null);
        setAvatarPreview(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (isLogin) {
            try {
                const response = await fetch('http://localhost:8000/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username }),
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    onLoginSuccess(data.user_info); 
                    navigate('/chat');
                } else {
                    setError(data.detail || 'Login failed.');
                }
            } catch (err) {
                setError('Cannot connect to server. Is the API running?');
                console.error('Login error:', err);
            }
        } else {
            // Registration
            const formData = new FormData();
            formData.append('patient_id', username);
            formData.append('age_range', ageRange);
            formData.append('gender', gender);
            formData.append('smoking_status', smokingStatus);
            formData.append('has_hypertension', hasHypertension);
            formData.append('has_diabetes', hasDiabetes);
            formData.append('has_asthma_history', hasAsthma);
            formData.append('previous_respiratory_infections', previousInfections);
            formData.append('current_medications', medications);
            formData.append('allergies', allergies);
            formData.append('last_consultation_date', lastConsultationDate);
            
            if (avatar) {
                formData.append('avatar', avatar);
            }

            try {
                const response = await fetch('http://localhost:8000/register', {
                    method: 'POST',
                    body: formData,
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    setSuccess('Registration successful! Please login.');
                    setIsLogin(true);
                    resetForm();
                } else {
                    setError(data.detail || 'Registration failed.');
                }
            } catch (err) {
                setError('Cannot connect to server. Is the API running?');
                console.error('Registration error:', err);
            }
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-form-wrapper">
                <h2>{isLogin ? 'Patient Login' : 'Patient Registration'}</h2>
                
                {error && <p className="message error">{error}</p>}
                {success && <p className="message success">{success}</p>}
                
                <form onSubmit={handleSubmit}>
                    {!isLogin && (
                        <>
                            <div className="input-group">
                                <label>Profile Picture (Optional)</label>
                                <input 
                                    type="file" 
                                    accept="image/*"
                                    onChange={handleAvatarChange}
                                />
                                {avatarPreview && (
                                    <div style={{ marginTop: '10px', textAlign: 'center' }}>
                                        <img 
                                            src={avatarPreview} 
                                            alt="Avatar preview" 
                                            style={{ 
                                                width: '100px', 
                                                height: '100px', 
                                                borderRadius: '50%', 
                                                objectFit: 'cover',
                                                border: '2px solid #be8456'
                                            }} 
                                        />
                                    </div>
                                )}
                            </div>
                            
                            <div className="input-group">
                                <label>Username (Patient ID)</label>
                                <input 
                                    type="text" 
                                    value={username} 
                                    onChange={(e) => setUsername(e.target.value)} 
                                    required 
                                />
                            </div>
                            
                            <div className="input-group">
                                <label>Age Range</label>
                                <select 
                                    value={ageRange} 
                                    onChange={(e) => setAgeRange(e.target.value)} 
                                    required
                                >
                                    <option value="">Select...</option>
                                    <option value="0-9">0-9</option>
                                    <option value="10-19">10-19</option>
                                    <option value="20-29">20-29</option>
                                    <option value="30-39">30-39</option>
                                    <option value="40-49">40-49</option>
                                    <option value="50+">50+</option>
                                </select>
                            </div>
                            
                            <div className="input-group">
                                <label>Gender</label>
                                <select 
                                    value={gender} 
                                    onChange={(e) => setGender(e.target.value)} 
                                    required
                                >
                                    <option value="">Select...</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                            
                            <div className="input-group">
                                <label>Smoking Status</label>
                                <select 
                                    value={smokingStatus} 
                                    onChange={(e) => setSmokingStatus(e.target.value)} 
                                    required
                                >
                                    <option value="">Select...</option>
                                    <option value="Never Smoked">Never Smoked</option>
                                    <option value="Former Smoker">Former Smoker</option>
                                    <option value="Current Smoker">Current Smoker</option>
                                </select>
                            </div>
                            
                            <div className="input-group">
                                <label>Last Consultation Date (Optional)</label>
                                <input 
                                    type="date" 
                                    value={lastConsultationDate} 
                                    onChange={(e) => setLastConsultationDate(e.target.value)} 
                                />
                            </div>
                            
                            <div className="input-group">
                                <label>Current Medications (Optional)</label>
                                <textarea 
                                    placeholder="List any current medications" 
                                    value={medications} 
                                    onChange={(e) => setMedications(e.target.value)} 
                                />
                            </div>
                            
                            <div className="input-group">
                                <label>Allergies (Optional)</label>
                                <textarea 
                                    placeholder="List any known allergies" 
                                    value={allergies} 
                                    onChange={(e) => setAllergies(e.target.value)} 
                                />
                            </div>
                            
                            <div className="input-group">
                                <label>Previous Respiratory Infections</label>
                                <input 
                                    type="number" 
                                    min="0" 
                                    value={previousInfections} 
                                    onChange={(e) => setPreviousInfections(parseInt(e.target.value, 10) || 0)} 
                                />
                            </div>
                            
                            <div className="checkbox-group">
                                <label>
                                    <input 
                                        type="checkbox" 
                                        checked={hasHypertension} 
                                        onChange={(e) => setHasHypertension(e.target.checked)} 
                                    />
                                    History of Hypertension
                                </label>
                                <label>
                                    <input 
                                        type="checkbox" 
                                        checked={hasDiabetes} 
                                        onChange={(e) => setHasDiabetes(e.target.checked)} 
                                    />
                                    History of Diabetes
                                </label>
                                <label>
                                    <input 
                                        type="checkbox" 
                                        checked={hasAsthma} 
                                        onChange={(e) => setHasAsthma(e.target.checked)} 
                                    />
                                    History of Asthma
                                </label>
                            </div>
                        </>
                    )}

                    {isLogin && (
                        <div className="input-group">
                            <label>Username (Patient ID)</label>
                            <input 
                                type="text" 
                                value={username} 
                                onChange={(e) => setUsername(e.target.value)} 
                                required 
                            />
                        </div>
                    )}
                    
                    <button type="submit" className="auth-button">
                        {isLogin ? 'Login' : 'Register'}
                    </button>
                </form>
                
                <p className="toggle-text">
                    {isLogin ? "Don't have an account?" : 'Already have an account?'}
                    <button 
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError('');
                            setSuccess('');
                        }} 
                        className="toggle-button"
                    >
                        {isLogin ? 'Register' : 'Login'}
                    </button>
                </p>
            </div>
        </div>
    );
};

export default AuthPage;