/*
============================================================
TrafficIQ

Signal Panel Component

Author : Vamsi Krishna

Description:
Updates AI Decision Engine and Emergency Panel.
============================================================
*/

// ==========================================================
// Update AI Decision Panel
// ==========================================================

function updateSignalPanel(dashboard){

    if(!dashboard) return;

    const signals = dashboard.signals || {};

    const emergency = dashboard.emergency || {};
    const emergencySummary = dashboard.emergency_summary || {};

    // ------------------------------------------
    // Find Active Lane
    // ------------------------------------------

    let selectedLane = "-";
    let greenTime = 0;
    let decisionReason = "-";
    let aiMode = "Adaptive";

    for(const lane in signals){

        const signal = signals[lane];

        selectedLane = lane;

        greenTime = signal.green_time || 0;

        decisionReason =
            signal.reason ||
            "Adaptive AI Density Control";

        aiMode =
            signal.mode ||
            "Adaptive";

        break;

    }

    // ------------------------------------------
    // Update Decision Panel
    // ------------------------------------------

    setText(
        "decisionLane",
        selectedLane.replace("_"," ")
    );

    setText(
        "greenTime",
        greenTime + " sec"
    );

    setText(
        "decisionMode",
        aiMode
    );

    setText(
        "decisionReason",
        decisionReason
    );

    // ------------------------------------------
    // AI Confidence
    // ------------------------------------------

    const confidence =
        dashboard.statistics?.confidence ??
        "95%";

    setText(
        "aiConfidence",
        confidence
    );

    // ------------------------------------------
    // Density Score
    // ------------------------------------------

    let score = "-";

    if(signals[selectedLane]){

        score =
            signals[selectedLane].score ?? "-";

    }

    setText(
        "densityScore",
        score
    );

    // ======================================================
    // Emergency Panel
    // ======================================================

    const emergencyStatus =
        $("emergencyStatus");

    const emergencyVehicle =
        $("emergencyVehicle");

    const priorityLane =
        $("priorityLane");

    const emergencyConfidence =
        $("emergencyConfidence");

    const emergencyTrackId =
        $("emergencyTrackId");

    const overrideStatus =
        $("overrideStatus");

    // ------------------------------------------

    if(emergency.active){

        emergencyStatus.textContent =
            "ACTIVE";

        emergencyStatus.style.color =
            "#EF4444";

        emergencyVehicle.textContent =
            emergency.vehicle;

        priorityLane.textContent =
            emergency.lane;

        emergencyConfidence.textContent =
            emergency.confidence
                ? `${Math.round(emergency.confidence * 100)}%`
                : "-";

        emergencyTrackId.textContent =
            emergency.track_id || "-";

        overrideStatus.textContent =
            "Enabled";

        overrideStatus.style.color =
            "#EF4444";

        // Emergency Banner

        const banner =
            $("emergencyBanner");

        if(banner){

            banner.classList.remove(
                "hidden"
            );

            banner.innerHTML =
                "🚨 " +
                emergency.vehicle.toUpperCase() +
                " DETECTED - " +
                emergency.lane +
                " (" +
                (emergencySummary.current_count || 0) +
                " active)";

        }

    }

    else{

        emergencyStatus.textContent =
            "No Emergency";

        emergencyStatus.style.color =
            "#22C55E";

        emergencyVehicle.textContent =
            "-";

        priorityLane.textContent =
            "-";

        overrideStatus.textContent =
            "Disabled";

        overrideStatus.style.color =
            "#22C55E";

        const banner =
            $("emergencyBanner");

        if(banner){

            banner.classList.add(
                "hidden"
            );

        }

    }

}