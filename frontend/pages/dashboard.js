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

// ==========================================

async function loadDashboard(){

    try{

        const data = await getDashboard();

        console.log(data);

        document.querySelector(".live-badge").innerHTML =
            data.system.processing
            ? "🟢 LIVE"
            : "⚪ IDLE";

        document.getElementById("fps").innerHTML =
            data.system.fps;

        document.getElementById("fpsCard").innerHTML =
            data.system.fps;

        document.getElementById("vehicleCount").innerHTML =
            data.dashboard.statistics.total_vehicles;

    }

    catch(err){

        console.log(err);

    }

}

setInterval(loadDashboard,1000);

loadDashboard();