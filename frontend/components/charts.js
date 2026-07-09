/*
============================================================
TrafficIQ

Charts Component

Author : Vamsi Krishna

Description:
Creates and updates all dashboard charts.
============================================================
*/

let vehicleChart;
let densityChart;
let distributionChart;
let timelineChart;

// ==========================================================
// Initialize Charts
// ==========================================================

function initializeCharts(){

    createVehicleChart();

    createDensityChart();

    createDistributionChart();

    createTimelineChart();

}

// ==========================================================
// Vehicle Trend
// ==========================================================

function createVehicleChart(){

    const canvas = document.getElementById("vehicleChart");

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    vehicleChart = new Chart(ctx,{

        type:"line",

        data:{

            labels:[],

            datasets:[{

                label:"Vehicles",

                data:[],

                borderColor:"#3B82F6",

                backgroundColor:"rgba(59,130,246,.15)",

                borderWidth:3,

                tension:.4,

                fill:true

            }]

        },

        options:{

            responsive:true,

            maintainAspectRatio:false,

            animation:true,

            plugins:{

                legend:{

                    display:false

                }

            }

        }

    });

}

// ==========================================================
// Lane Density
// ==========================================================

function createDensityChart(){

    const canvas = document.getElementById("densityChart");

    if (!canvas) return;

    const ctx=canvas.getContext("2d");

    densityChart=new Chart(ctx,{

        type:"bar",

        data:{

            labels:[
                "Lane A",
                "Lane B",
                "Lane C",
                "Lane D"
            ],

            datasets:[{

                data:[0,0,0,0],

                backgroundColor:[

                    "#3B82F6",

                    "#22C55E",

                    "#F59E0B",

                    "#EF4444"

                ]

            }]

        },

        options:{

            responsive:true,

            plugins:{

                legend:{

                    display:false

                }

            }

        }

    });

}

// ==========================================================
// Vehicle Distribution
// ==========================================================

function createDistributionChart(){

    const canvas = document.getElementById("distributionChart");

    if (!canvas) return;

    const ctx=canvas.getContext("2d");

    distributionChart=new Chart(ctx,{

        type:"doughnut",

        data:{

            labels:[

                "Car",

                "Bus",

                "Van",

                "Others"

            ],

            datasets:[{

                data:[0,0,0,0],

                backgroundColor:[

                    "#3B82F6",

                    "#22C55E",

                    "#F59E0B",

                    "#8B5CF6"

                ]

            }]

        },

        options:{

            responsive:true,

            cutout:"70%"

        }

    });

}

// ==========================================================
// Signal Timeline
// ==========================================================

function createTimelineChart(){

    const canvas = document.getElementById("timelineChart");

    if (!canvas) return;

    const ctx=canvas.getContext("2d");

    timelineChart=new Chart(ctx,{

        type:"line",

        data:{

            labels:[],

            datasets:[{

                label:"Green Time",

                data:[],

                borderColor:"#22C55E",

                backgroundColor:"rgba(34,197,94,.15)",

                fill:true,

                tension:.35

            }]

        },

        options:{

            responsive:true,

            plugins:{

                legend:{

                    display:false

                }

            }

        }

    });

}

// ==========================================================
// Update Charts
// ==========================================================

function updateCharts(dashboard){

    if(!dashboard) return;

    // --------------------------
    // Vehicle Total
    // --------------------------

    const total =
        dashboard.statistics?.total_vehicles || 0;

    const time =
        new Date().toLocaleTimeString();

    // Vehicle Trend

    vehicleChart.data.labels.push(time);

    vehicleChart.data.datasets[0].data.push(total);

    if(vehicleChart.data.labels.length>15){

        vehicleChart.data.labels.shift();

        vehicleChart.data.datasets[0].data.shift();

    }

    vehicleChart.update();

    // --------------------------
    // Lane Density
    // --------------------------

    const density =
        dashboard.density?.class_density || {};

    densityChart.data.datasets[0].data=[

        density.Lane_A?.total||0,

        density.Lane_B?.total||0,

        density.Lane_C?.total||0,

        density.Lane_D?.total||0

    ];

    densityChart.update();

    // --------------------------
    // Vehicle Distribution
    // --------------------------

    distributionChart.data.datasets[0].data=[

        dashboard.statistics?.cars||0,

        dashboard.statistics?.bus||0,

        dashboard.statistics?.van||0,

        dashboard.statistics?.others||0

    ];

    distributionChart.update();

    // --------------------------
    // Signal Timeline
    // --------------------------

    let green=0;

    for(const lane in dashboard.signals){

        green=
            dashboard.signals[lane].green_time;

        break;

    }

    timelineChart.data.labels.push(time);

    timelineChart.data.datasets[0].data.push(green);

    if(timelineChart.data.labels.length>15){

        timelineChart.data.labels.shift();

        timelineChart.data.datasets[0].data.shift();

    }

    timelineChart.update();

}