/*
============================================================
TrafficIQ

Helper Functions

Author : Vamsi Krishna
============================================================
*/

// ==========================================================
// Get Element
// ==========================================================

function $(id){

    return document.getElementById(id);

}

// ==========================================================
// Set Text
// ==========================================================

function setText(id,value){

    const element=$(id);

    if(element){

        element.textContent=value;

    }

}

// ==========================================================
// Format Number
// ==========================================================

function formatNumber(value){

    return Number(value).toLocaleString();

}

// ==========================================================
// Clock
// ==========================================================

function updateClock(){

    const now=new Date();

    $("clock").textContent=

        now.toLocaleTimeString();

}

// ==========================================================
// Start Clock
// ==========================================================

setInterval(updateClock,1000);

updateClock();

// ==========================================================
// Percentage
// ==========================================================

function percentage(value,max=20){

    return Math.min(

        (value/max)*100,

        100

    );

}

// ==========================================================
// Density Level
// ==========================================================

function densityLevel(total){

    if(total<=5)

        return "LOW";

    if(total<=10)

        return "MEDIUM";

    if(total<=20)

        return "HIGH";

    return "VERY HIGH";

}

// ==========================================================
// Progress Width
// ==========================================================

function progressWidth(total){

    return percentage(total)+"%";

}