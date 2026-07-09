/*
============================================================
TrafficIQ

Dashboard Controller

Author : Vamsi Krishna

Description:
Main controller for the TrafficIQ Dashboard.
============================================================
*/

// ==========================================================
// Dashboard State
// ==========================================================

let dashboardData = null;

// ==========================================================
// Initialize Dashboard
// ==========================================================

async function initializeDashboard() {

    console.log("TrafficIQ Dashboard Started");

    initializeCharts();

    initializeButtons();

    try {

        await refreshDashboard();

    } catch (error) {

        console.warn("Initial dashboard load failed.", error);

    }

    setInterval(() => {

        refreshDashboard().catch(() => {});

    }, REFRESH_INTERVAL);
}

// ==========================================================
// Refresh Dashboard
// ==========================================================

async function refreshDashboard() {

    const data = await fetchDashboard();

    if (!data) {

        console.warn("Dashboard refresh failed.");

        const fallback = {
            statistics: {},
            density: { class_density: {} },
            signals: {},
            counter: {},
            emergency: { active: false }
        };

        dashboardData = fallback;
        updateKPICards(fallback);
        updateLaneCards(fallback);
        updateSignalPanel(fallback);
        updateCharts(fallback);
        updateSystemStatus(fallback);
        return;

    }

    dashboardData = data;

    updateKPICards(data);
    updateLaneCards(data);
    updateSignalPanel(data);
    updateCharts(data);
    updateSystemStatus(data);

}

// ==========================================================
// KPI Cards
// ==========================================================

function updateKPICards(data) {

    const stats = data.statistics || {};

    setText(
        "vehicleCount",
        formatNumber(stats.total_vehicles || 0)
    );

    setText(
        "confidence",
        stats.confidence || "95%"
    );

    setText(
        "fpsCard",
        data.processing_time
            ? Math.round(1000 / data.processing_time)
            : 0
    );

    // Density

    const density = data.density?.class_density || {};

    let highest = 0;

    for (const lane in density) {

        highest = Math.max(

            highest,

            density[lane].total

        );

    }

    setText(

        "density",

        densityLevel(highest)

    );

    // Green Lane

    const signals = data.signals || {};

    let lane = "-";

    for (const key in signals) {

        lane = key.replace("_", " ");

        break;

    }

    setText(

        "greenLane",

        lane

    );

    // Emergency Count

    const emergency = data.emergency;

    if (emergency && emergency.active) {

        setText(

            "emergencyCount",

            1

        );

    }

    else {

        setText(

            "emergencyCount",

            0

        );

    }

}

// ==========================================================
// System Status
// ==========================================================

function updateSystemStatus(data) {

    // Sidebar FPS

    const fps = data.processing_time

        ? Math.round(1000 / data.processing_time)

        : 0;

    setText("fps", fps);

    setText("overlayFPS", fps);

    // Camera Overlay

    const total =

        data.statistics?.total_vehicles || 0;

    setText(

        "overlayVehicles",

        total

    );

}

// ==========================================================
// Buttons
// ==========================================================

function initializeButtons() {

    const startButton =

        document.querySelector(".btn-primary");

    if (startButton) {

        startButton.addEventListener(

            "click",

            async () => {

                const result =

                    await startVideo();

                console.log(result);

            }

        );

    }

    const stopButton =

        document.querySelector(".btn-secondary");

    if (stopButton) {

        stopButton.addEventListener(

            "click",

            async () => {

                const result =

                    await stopVideo();

                console.log(result);

            }

        );

    }

}

// ==========================================================
// Page Loaded
// ==========================================================

window.addEventListener(

    "DOMContentLoaded",

    initializeDashboard

);