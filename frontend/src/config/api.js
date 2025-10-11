const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '');
  }
  
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }

  return '';  
}

export const API_BASE_URL = getApiBaseUrl();

export const getApiUrl = (endpoint) => {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${cleanEndpoint}`;
};

export const getAvatarUrl = (avatarPath) => {
  if (!avatarPath) {
    return 'https://ui-avatars.com/api/?name=Patient&size=150&background=4A90E2&color=ffffff';
  }
  
  if (avatarPath.startsWith('http://') || avatarPath.startsWith('https://')) {
    return avatarPath;
  }
  
  const cleanPath = avatarPath.startsWith('/') ? avatarPath : `/${avatarPath}`;
  return `${API_BASE_URL}${cleanPath}`;
};

export default API_BASE_URL;