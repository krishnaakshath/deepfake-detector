// Deployed backend URL (Render)
const DEPLOYED_API_URL = 'https://deepguard-api-d568.onrender.com';

// Local development URL
const LOCAL_API_URL = 'http://localhost:8000';

// Automatically detect environment
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// Export the active API URL
window.DEEPGUARD_API_URL = isLocalhost ? LOCAL_API_URL : DEPLOYED_API_URL;

console.log(`🌐 Environment: ${isLocalhost ? 'Local' : 'Production'}`);
console.log(`🔗 API URL: ${window.DEEPGUARD_API_URL}`);

console.log(`🌐 API URL: ${window.DEEPGUARD_API_URL}`);
