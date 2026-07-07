/* ==========================================================
   TrafficIQ API
========================================================== */

const API = "http://127.0.0.1:8000";

async function getDashboard(){

    const response = await fetch(`${API}/dashboard`);

    return await response.json();

}

async function startVideo(){

    return await fetch(`${API}/start-video`,{

        method:"POST"

    });

}

async function stopVideo(){

    return await fetch(`${API}/stop-video`,{

        method:"POST"

    });

}