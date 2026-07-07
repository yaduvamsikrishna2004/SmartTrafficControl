/* ==========================================================
   TrafficIQ API Service
========================================================== */

const API_BASE = "http://127.0.0.1:8000";

// ======================================
// Backend Status
// ======================================

async function getStatus(){

    const response = await fetch(`${API_BASE}/status`);

    return await response.json();

}

// ======================================
// Traffic Counts
// ======================================

async function getTraffic(){

    const response = await fetch(`${API_BASE}/traffic`);

    return await response.json();

}

// ======================================
// Density
// ======================================

async function getDensity(){

    const response = await fetch(`${API_BASE}/density`);

    return await response.json();

}

// ======================================
// Signal Plan
// ======================================

async function getSignals(){

    const response = await fetch(`${API_BASE}/signals`);

    return await response.json();

}

// ======================================
// Analytics
// ======================================

async function getAnalytics(){

    const response = await fetch(`${API_BASE}/analytics`);

    return await response.json();

}

// ======================================
// Start Processing
// ======================================

async function startVideo(){

    return await fetch(

        `${API_BASE}/start-video`,

        {

            method:"POST"

        }

    );

}

// ======================================
// Stop Processing
// ======================================

async function stopVideo(){

    return await fetch(

        `${API_BASE}/stop-video`,

        {

            method:"POST"

        }

    );

}