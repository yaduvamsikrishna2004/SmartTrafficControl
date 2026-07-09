/*
==========================================================
TrafficIQ Sidebar

Author : Vamsi Krishna
==========================================================
*/

console.log("Sidebar Component Loaded");

// ======================================================
// Initialize Sidebar
// ======================================================

document.addEventListener("DOMContentLoaded", () => {

    initializeSidebar();

});

// ======================================================

function initializeSidebar() {

    const menuItems = document.querySelectorAll(".sidebar-menu a");

    menuItems.forEach(item => {

        item.addEventListener("click", () => {

            menuItems.forEach(link => {

                link.classList.remove("active");

            });

            item.classList.add("active");

        });

    });

}

// ======================================================
// Sidebar Collapse (Future Feature)
// ======================================================

function toggleSidebar() {

    const sidebar = document.querySelector(".sidebar");

    if (!sidebar) return;

    sidebar.classList.toggle("collapsed");

}

// ======================================================
// Mobile Sidebar
// ======================================================

function openSidebar() {

    const sidebar = document.querySelector(".sidebar");

    if (!sidebar) return;

    sidebar.classList.add("show");

}

function closeSidebar() {

    const sidebar = document.querySelector(".sidebar");

    if (!sidebar) return;

    sidebar.classList.remove("show");

}

// ======================================================
// Resize Handling
// ======================================================

window.addEventListener("resize", () => {

    if (window.innerWidth > 992) {

        closeSidebar();

    }

});

// ======================================================
// Export (Optional)
// ======================================================

window.Sidebar = {

    toggleSidebar,

    openSidebar,

    closeSidebar

};