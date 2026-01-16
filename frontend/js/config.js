/**
 * DeepGuard Configuration
 * Edit this file to switch between local development and deployed backend
 */

// Set to true when deploying, false for local development
const IS_DEPLOYED = true;

// Your deployed backend URL (update this after deploying to Render)
const DEPLOYED_API_URL = 'https://deepguard-api-d568.onrender.com';

// Local development URL
const LOCAL_API_URL = 'http://localhost:8000';

// Export the active API URL
window.DEEPGUARD_API_URL = IS_DEPLOYED ? DEPLOYED_API_URL : LOCAL_API_URL;

console.log(`🌐 API URL: ${window.DEEPGUARD_API_URL}`);
