/*
============================================================
TrafficIQ

Constants

Author : Vamsi Krishna
============================================================
*/

// ==========================================================
// Backend
// ==========================================================

const API_BASE = (function(){
    if (window.location.protocol.startsWith('http')) {
        return window.location.origin;
    }
    return 'http://127.0.0.1:8000';
})();

// ==========================================================
// API Endpoints
// ==========================================================

const API = {

    dashboard: `${API_BASE}/dashboard`,

    statistics: `${API_BASE}/statistics`,

    analytics: `${API_BASE}/analytics`,

    density: `${API_BASE}/density`,

    lanes: `${API_BASE}/lanes`,

    signals: `${API_BASE}/signals`,

    emergency: `${API_BASE}/emergency`,

    health: `${API_BASE}/health`,

    camera: `${API_BASE}/camera`,

    startVideo: `${API_BASE}/start-video`,

    stopVideo: `${API_BASE}/stop-video`,

    videoFeed: `${API_BASE}/video-feed`

};

// ==========================================================
// Dashboard Refresh
// ==========================================================

const REFRESH_INTERVAL = 500;

// ==========================================================
// Lane Names
// ==========================================================

const LANES = [

    "Lane_A",

    "Lane_B",

    "Lane_C",

    "Lane_D"

];

// ==========================================================
// Density Levels
// ==========================================================

const DENSITY = {

    LOW: "LOW",

    MEDIUM: "MEDIUM",

    HIGH: "HIGH",

    VERY_HIGH: "VERY HIGH"

};

// ==========================================================
// Colors
// ==========================================================

const COLORS = {

    blue: "#3B82F6",

    green: "#22C55E",

    yellow: "#F59E0B",

    red: "#EF4444",

    cyan: "#06B6D4",

    purple: "#8B5CF6"

};