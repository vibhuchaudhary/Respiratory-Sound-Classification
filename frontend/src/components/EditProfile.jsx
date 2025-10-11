import React, { useEffect, useState } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import "../css/EditProfile.css";
import { API_BASE_URL, getAvatarUrl } from "../config/api";

const GENDER_OPTIONS = ["Male", "Female", "Other", "Prefer not to say"];
const SMOKING_STATUS_OPTIONS = ["Never", "Former", "Current"];

export default function EditProfile({ user, onUserUpdate }) {
  const [formData, setFormData] = useState({
    email: "",
    full_name: "",
    age_range: "",
    gender: "",
    smoking_status: "",
    has_hypertension: false,
    has_diabetes: false,
    has_asthma_history: false,
    previous_respiratory_infections: 0,
    current_medications: "",
    allergies: "",
    last_consultation_date: "",
    password: "",
  });
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPasswordField, setShowPasswordField] = useState(false);

  // Load current patient data when component mounts
  useEffect(() => {
    if (!user?.patient_id) return;
    
    axios
      .get(`${API_BASE_URL}/api/patient/${user.patient_id}`)
      .then((res) => {
        setFormData((prev) => ({ 
          ...prev, 
          email: res.data.email || "",
          full_name: res.data.full_name || "",
          age_range: res.data.age_range || "",
          gender: res.data.gender || "",
          smoking_status: res.data.smoking_status || "",
          has_hypertension: res.data.has_hypertension || false,
          has_diabetes: res.data.has_diabetes || false,
          has_asthma_history: res.data.has_asthma_history || false,
          previous_respiratory_infections: res.data.previous_respiratory_infections || 0,
          current_medications: res.data.current_medications || "",
          allergies: res.data.allergies || "",
          last_consultation_date: res.data.last_consultation_date || "",
        }));
        
        if (res.data.avatar) {
          setAvatarPreview(getAvatarUrl(res.data.avatar));
        }
      })
      .catch((err) => {
        console.error(err);
        toast.error("Failed to load profile");
      });
  }, [user]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleAvatarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        toast.error("Image size must be less than 5MB");
        return;
      }
      
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
      if (!validTypes.includes(file.type)) {
        toast.error("Invalid image format. Please use JPG, PNG, GIF, or WEBP");
        return;
      }
      
      setAvatarFile(file);
      // Show preview of selected image
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const submitData = new FormData();

      // Include all fields that have values
      if (formData.email) submitData.append('email', formData.email);
      if (formData.full_name) submitData.append('full_name', formData.full_name);
      if (formData.age_range) submitData.append('age_range', formData.age_range);
      if (formData.gender) submitData.append('gender', formData.gender);
      if (formData.smoking_status) submitData.append('smoking_status', formData.smoking_status);
      if (formData.current_medications) submitData.append('current_medications', formData.current_medications);
      if (formData.allergies) submitData.append('allergies', formData.allergies);
      if (formData.last_consultation_date) submitData.append('last_consultation_date', formData.last_consultation_date);
      
      // Only include password if user wants to change it
      if (formData.password && formData.password.trim()) {
        if (formData.password.length < 6) {
          toast.error("Password must be at least 6 characters");
          setLoading(false);
          return;
        }
        submitData.append('password', formData.password);
      }
      
      // Always include boolean fields
      submitData.append('has_hypertension', formData.has_hypertension);
      submitData.append('has_diabetes', formData.has_diabetes);
      submitData.append('has_asthma_history', formData.has_asthma_history);
      submitData.append('previous_respiratory_infections', formData.previous_respiratory_infections);

      if (avatarFile) {
        submitData.append('avatar', avatarFile);
      }

      await axios.put(
        `${API_BASE_URL}/api/patient/${user.username}/update`,
        submitData,
        { 
          headers: { 
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          } 
        }
      );
      
      toast.success("Profile updated successfully!");
      
      // Clear password field after successful update
      setFormData(prev => ({ ...prev, password: "" }));
      setShowPasswordField(false);
      
      // Fetch updated profile data
      const updatedProfile = await axios.get(`${API_BASE_URL}/api/patient/${user.patient_id}`);
      
      // Update parent component with new data
      if (onUserUpdate) {
        onUserUpdate({
          ...user,
          name: updatedProfile.data.full_name || user.name,
          email: updatedProfile.data.email || user.email,
          avatar: updatedProfile.data.avatar || user.avatar
        });
      }
      
    } catch (err) {
      console.error(err);
      toast.error(err.response?.data?.detail || "Update failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="edit-profile-container">
      <h1>Edit Profile 🩺</h1>
      
      <form onSubmit={handleSubmit} className="edit-form">
        {/* Avatar Upload */}
        <div className="form-grid-span-2 avatar-upload-section">
          <label>Profile Picture</label>
          {avatarPreview && (
            <img 
              src={avatarPreview} 
              alt="Avatar preview" 
              className="avatar-preview"
              onError={(e) => {
                e.target.src = "https://ui-avatars.com/api/?name=Patient&size=150&background=4A90E2&color=ffffff";
              }}
            />
          )}
          <div className="file-input-wrapper">
            <label htmlFor="avatar" className="file-upload-button">
              Choose Image
            </label>
            <input 
              id="avatar"
              type="file" 
              accept="image/jpeg,image/jpg,image/png,image/gif,image/webp"
              onChange={handleAvatarChange}
              className="file-input-hidden"
            />
            <span className="file-name-display">
              {avatarFile ? avatarFile.name : "No file selected"}
            </span>
          </div>
          <small className="help-text">Allowed formats: JPG, PNG, GIF, WEBP (Max 5MB)</small>
        </div>

        {/* Account Info */}
        <div className="form-grid-span-2">
          <h3>Account Information</h3>
        </div>

        <div>
          <label htmlFor="email">Email Address</label>
          <input 
            id="email" 
            type="email" 
            name="email" 
            value={formData.email} 
            onChange={handleChange} 
            placeholder="your.email@example.com"
            disabled={user?.auth_provider === 'google'}
          />
          {user?.auth_provider === 'google' && (
            <small className="help-text">Email cannot be changed for Google accounts</small>
          )}
        </div>

        <div>
          <label htmlFor="full_name">Full Name</label>
          <input 
            id="full_name" 
            type="text" 
            name="full_name" 
            value={formData.full_name} 
            onChange={handleChange} 
            placeholder="John Doe"
          />
        </div>

        {/* Password Change (only for local auth users) */}
        {user?.auth_provider !== 'google' && (
          <div className="form-grid-span-2">
            {!showPasswordField ? (
              <button 
                type="button" 
                onClick={() => setShowPasswordField(true)}
                className="secondary-button"
              >
                Change Password
              </button>
            ) : (
              <>
                <label htmlFor="password">New Password (leave blank to keep current)</label>
                <input 
                  id="password" 
                  type="password" 
                  name="password" 
                  value={formData.password} 
                  onChange={handleChange} 
                  placeholder="Enter new password"
                  minLength="6"
                />
                <small className="help-text">Minimum 6 characters</small>
              </>
            )}
          </div>
        )}

        {/* Basic Info */}
        <div className="form-grid-span-2">
          <h3>Health Information</h3>
        </div>

        <div>
          <label htmlFor="age_range">Age Range</label>
          <input 
            id="age_range" 
            type="text" 
            name="age_range" 
            value={formData.age_range} 
            onChange={handleChange} 
            placeholder="e.g. 30-40" 
          />
        </div>
        
        <div>
          <label htmlFor="gender">Gender</label>
          <select id="gender" name="gender" value={formData.gender} onChange={handleChange}>
            <option value="">Select...</option>
            {GENDER_OPTIONS.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label htmlFor="smoking_status">Smoking Status</label>
          <select id="smoking_status" name="smoking_status" value={formData.smoking_status} onChange={handleChange}>
            <option value="">Select...</option>
            {SMOKING_STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label htmlFor="previous_respiratory_infections">Previous Respiratory Infections</label>
          <input 
            id="previous_respiratory_infections" 
            type="number" 
            name="previous_respiratory_infections" 
            value={formData.previous_respiratory_infections} 
            onChange={handleChange} 
            min="0" 
          />
        </div>
        
        {/* Medical Conditions */}
        <div className="form-grid-span-2 checkbox-group">
          <h4>Medical Conditions</h4>
          <label htmlFor="has_hypertension">
            <input 
              type="checkbox" 
              id="has_hypertension" 
              name="has_hypertension" 
              checked={formData.has_hypertension} 
              onChange={handleChange} 
            />
            Hypertension
          </label>
          <label htmlFor="has_diabetes">
            <input 
              type="checkbox" 
              id="has_diabetes" 
              name="has_diabetes" 
              checked={formData.has_diabetes} 
              onChange={handleChange} 
            />
            Diabetes
          </label>
          <label htmlFor="has_asthma_history">
            <input 
              type="checkbox" 
              id="has_asthma_history" 
              name="has_asthma_history" 
              checked={formData.has_asthma_history} 
              onChange={handleChange} 
            />
            Asthma History
          </label>
        </div>
        
        {/* Medications */}
        <div className="form-grid-span-2">
          <label htmlFor="current_medications">Current Medications</label>
          <textarea 
            id="current_medications" 
            name="current_medications" 
            value={formData.current_medications} 
            onChange={handleChange} 
            rows="3"
            placeholder="List any medications you are currently taking..."
          />
        </div>
        
        {/* Allergies */}
        <div className="form-grid-span-2">
          <label htmlFor="allergies">Allergies</label>
          <textarea 
            id="allergies" 
            name="allergies" 
            value={formData.allergies} 
            onChange={handleChange} 
            rows="3"
            placeholder="List any known allergies..."
          />
        </div>
        
        {/* Last Consultation */}
        <div className="form-grid-span-2">
          <label htmlFor="last_consultation_date">Last Consultation Date</label>
          <input 
            id="last_consultation_date" 
            type="date" 
            name="last_consultation_date" 
            value={formData.last_consultation_date ? formData.last_consultation_date.split("T")[0] : ""} 
            onChange={handleChange} 
          />
        </div>
        
        <button type="submit" disabled={loading} className="form-grid-span-2">
          {loading ? "Saving..." : "💾 Save Changes"}
        </button>
      </form>
    </div>
  );
}