import React, { useState, useEffect } from 'react';
import '../css/EditProfile.css';

const EditProfile = ({ user, onProfileUpdate }) => {
    const [name, setName] = useState('');
    const [username, setUsername] = useState('');
    const [avatarPreview, setAvatarPreview] = useState(null);
    const [avatarFile, setAvatarFile] = useState(null); // State for the file object

    useEffect(() => {
        if (user) {
            setName(user.name);
            setUsername(user.username);
            setAvatarPreview(user.avatar);
        }
    }, [user]);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setAvatarFile(file); // Store the file object
            setAvatarPreview(URL.createObjectURL(file)); // Create a temporary URL for preview
        }
    };

    const handleSave = async () => {
        const formData = new FormData();
        formData.append('name', name);
        if (avatarFile) {
            formData.append('avatar', avatarFile);
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/profile/${username}`, {
                method: 'PUT',
                // Again, do not set the Content-Type header
                body: formData,
            });

            const data = await response.json();
            
            if (response.ok) {
                alert('Profile updated successfully!');
                if (onProfileUpdate) {
                    onProfileUpdate(data.user);
                }
            } else {
                alert('Update failed: ' + data.error);
            }
        } catch (error) {
            alert('Network error. Make sure the backend is running.');
        }
    };
    
    if (!user) {
        return <div className="edit-profile-container"><h1>Loading...</h1></div>;
    }

    return (
        <div className="edit-profile-container">
            <h1>Edit Profile</h1>
            <div className="edit-form">
                {avatarPreview && (
                    <div className="avatar-preview">
                        <img src={avatarPreview} alt="Avatar preview" />
                    </div>
                )}
                
                <div className="form-group">
                    <label>Full Name</label>
                    <input type="text" value={name} onChange={e => setName(e.target.value)} />
                </div>
                
                <div className="form-group">
                    <label>Username</label>
                    <input type="text" value={username} disabled />
                </div>
                
                <div className="form-group">
                    <label>Update Profile Picture</label>
                    <input type="file" accept="image/*" onChange={handleFileChange} />
                </div>
                
                <button onClick={handleSave} className="save-button">Save Changes</button>
            </div>
        </div>
    );
};

export default EditProfile;