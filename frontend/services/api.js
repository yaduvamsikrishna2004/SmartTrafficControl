/*
============================================================
TrafficIQ

API Service

Author : Vamsi Krishna

Description:
Handles all communication with the FastAPI backend.
============================================================
*/

// ==========================================================
// Generic GET Request
// ==========================================================

async function get(endpoint){

    try{

        const response = await fetch(endpoint);

        if(!response.ok){

            throw new Error(`HTTP ${response.status}`);

        }

        const json = await response.json();
        console.debug("GET", endpoint, json);
        return json;

    }

    catch(error){

        console.error("GET Error:", error);

        return null;

    }

}

// ==========================================================
// Generic POST Request
// ==========================================================

async function post(endpoint, body = {}){

    try{

        const response = await fetch(endpoint,{

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify(body)

        });

        if(!response.ok){

            throw new Error(`HTTP ${response.status}`);

        }

        return await response.json();

    }

    catch(error){

        console.error("POST Error:", error);

        return null;

    }

}

// ==========================================================
// Dashboard
// ==========================================================

async function getDashboard(){

    const resp = await get(API.dashboard);

    // backend wraps data with { system:..., dashboard: {...} }
    if(resp && resp.dashboard){
        console.debug("Unwrapping dashboard payload");
        return {
            ...resp.dashboard,
            system: resp.system || {}
        };
    }

    return resp;

}

// ==========================================================
// Analytics
// ==========================================================

async function getAnalytics(){

    return await get(API.analytics);

}

// ==========================================================
// Density
// ==========================================================

async function getDensity(){

    return await get(API.density);

}

// ==========================================================
// Traffic Counter
// ==========================================================

async function getTraffic(){

    return await get(API.traffic);

}

// ==========================================================
// Signals
// ==========================================================

async function getSignals(){

    return await get(API.signals);

}

// ==========================================================
// Lanes
// ==========================================================

async function getLanes(){

    return await get(API.lanes);

}

// ==========================================================
// Emergency
// ==========================================================

async function getEmergency(){

    return await get(API.emergency);

}

// ==========================================================
// Health
// ==========================================================

async function getHealth(){

    return await get(API.health);

}

// ==========================================================
// Camera
// ==========================================================

async function getCamera(){

    return await get(API.camera);

}

// ==========================================================
// Start Video
// ==========================================================

async function startVideo(){

    return await post(API.startVideo);

}

// ==========================================================
// Stop Video
// ==========================================================

async function stopVideo(){

    return await post(API.stopVideo);

}

// ==========================================================
// Health Check
// ==========================================================

async function checkBackend(){

    try{

        const response = await fetch(API.health);

        return response.ok;

    }

    catch{

        return false;

    }

}

// ==========================================================
// Dashboard Refresh
// ==========================================================

async function fetchDashboard(){

    const data = await getDashboard();

    if(!data){

        console.warn("Dashboard data unavailable.");

        return null;

    }

    console.debug("fetchDashboard ->", data);
    return data;

}