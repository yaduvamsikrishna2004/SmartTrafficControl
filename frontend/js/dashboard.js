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