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

const API_BASE = "http://127.0.0.1:8000";

// ==========================================================
// API Endpoints
// ==========================================================

const API = {

    dashboard: `${API_BASE}/dashboard`,

    analytics: `${API_BASE}/analytics`,

    density: `${API_BASE}/density`,

    traffic: `${API_BASE}/traffic`,

    signals: `${API_BASE}/signals`,

    startVideo: `${API_BASE}/start-video`,

    stopVideo: `${API_BASE}/stop-video`,

    videoFeed: `${API_BASE}/video-feed`

};

// ==========================================================
// Dashboard Refresh
// ==========================================================

const REFRESH_INTERVAL = 1000;

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