import React, { useEffect, useState } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import "../css/EditProfile.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const GENDER_OPTIONS = ["Male", "Female", "Other"];
const SMOKING_STATUS_OPTIONS = ["Never", "Former", "Current"];

export default function EditProfile({ user, onUserUpdate }) {
  const [formData, setFormData] = useState({
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
    avatar: "",
  });
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load current patient data when component mounts
  useEffect(() => {
    if (!user?.username) return;
    
    axios
      .get(`${API_BASE_URL}/api/patient/${user.username}`)
      .then((res) => {
        setFormData((prev) => ({ ...prev, ...res.data }));
        if (res.data.avatar) {
          const avatarUrl = res.data.avatar.startsWith('http') 
            ? res.data.avatar 
            : `${API_BASE_URL}${res.data.avatar}`;
          setAvatarPreview(avatarUrl);
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

      // Only include fields that have values
      if (formData.age_range) submitData.append('age_range', formData.age_range);
      if (formData.gender) submitData.append('gender', formData.gender);
      if (formData.smoking_status) submitData.append('smoking_status', formData.smoking_status);
      if (formData.current_medications) submitData.append('current_medications', formData.current_medications);
      if (formData.allergies) submitData.append('allergies', formData.allergies);
      if (formData.last_consultation_date) submitData.append('last_consultation_date', formData.last_consultation_date);
      
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
      
      // Fetch updated profile data
      const updatedProfile = await axios.get(`${API_BASE_URL}/api/patient/${user.username}`);
      
      // Update parent component with new avatar
      if (onUserUpdate) {
        onUserUpdate({
          ...user,
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
            />
          )}
          <div className="file-input-wrapper">
            <label htmlFor="avatar" className="file-upload-button">
              Choose Image
            </label>
            <input 
              id="avatar"
              type="file" 
              accept="image/*"
              onChange={handleAvatarChange}
              className="file-input-hidden"
            />
            <span className="file-name-display">
              {avatarFile ? avatarFile.name : "No file selected"}
            </span>
          </div>
        </div>

        {/* Basic Info */}
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