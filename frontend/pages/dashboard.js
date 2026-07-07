/* ==========================================================
   TrafficIQ Dashboard
========================================================== */

function updateClock(){

    const now = new Date();

    document.getElementById("clock").innerHTML =
        now.toLocaleTimeString();

}

setInterval(updateClock,1000);

updateClock();

// ======================================

async function loadDashboard(){

    try{

        const status = await getStatus();

        console.log(status);

    }

    catch(error){

        console.error(error);

    }

}

loadDashboard();

/* ==========================================================
   Dashboard Data
========================================================== */

async function loadDashboard(){

    try{

        const status = await getStatus();

        console.log(status);

        // Backend Status
        document.querySelector(".live-badge").innerHTML =
            status.processing
            ? "🟢 LIVE"
            : "⚪ IDLE";

        // FPS
        document.getElementById("fps").innerHTML =
            status.fps;

        document.getElementById("fpsCard").innerHTML =
            status.fps;

    }

    catch(error){

        console.error(error);

    }

}
setInterval(loadDashboard,1000);

loadDashboard();