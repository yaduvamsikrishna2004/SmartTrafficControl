/*
============================================================
TrafficIQ

Lane Cards Component

Author : Vamsi Krishna

Description:
Updates all lane cards from dashboard data.
============================================================
*/

// ==========================================================
// Update Lane Cards
// ==========================================================

function updateLaneCards(dashboard){

    if(!dashboard) return;

    const counter = dashboard.counter || {};

    const density = dashboard.density?.class_density || {};

    const signals = dashboard.signals || {};

    LANES.forEach(lane => {

        // ----------------------------------------
        // Vehicle Count
        // ----------------------------------------

        const vehicles =
            counter[lane]?.total ?? 0;

        // ----------------------------------------
        // Density
        // ----------------------------------------

        const total =
            density[lane]?.total ?? vehicles;

        const level =
            densityLevel(total);

        // ----------------------------------------
        // Green Time
        // ----------------------------------------

        const green =
            signals[lane]?.green_time ?? 15;

        // ----------------------------------------
        // Element IDs
        // ----------------------------------------

        const laneId =
            lane.replace("_","");

        setText(`${laneId}Vehicles`, vehicles);

        setText(`${laneId}Status`, level);

        setText(`${laneId}Green`, `${green} sec`);

        // ----------------------------------------
        // Progress Bar
        // ----------------------------------------

        const progress =
            $(`${laneId}Progress`);

        if(progress){

            progress.style.width =
                progressWidth(total);

            progress.className =
                "progress-bar";

            switch(level){

                case "LOW":

                    progress.style.background =
                        "#22C55E";

                    break;

                case "MEDIUM":

                    progress.style.background =
                        "#FACC15";

                    break;

                case "HIGH":

                    progress.style.background =
                        "#F97316";

                    break;

                default:

                    progress.style.background =
                        "#EF4444";

            }

        }

    });

}