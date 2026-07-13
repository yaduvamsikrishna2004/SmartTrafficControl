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

    // Set up click handlers for smooth scrolling
    menuItems.forEach(item => {

        item.addEventListener("click", event => {

            event.preventDefault();

            menuItems.forEach(link => link.classList.remove("active"));

            item.classList.add("active");

            const href = item.getAttribute("href") || "";
            const sectionId = href.startsWith("#") ? href.substring(1) : href;
            const section = document.getElementById(sectionId);

            if (section) {
                section.scrollIntoView({ behavior: "smooth", block: "start" });
            }

        });

    });

    // Set up intersection observer to highlight active section on scroll
    const sections = [];
    menuItems.forEach(item => {
        const href = item.getAttribute("href") || "";
        const sectionId = href.startsWith("#") ? href.substring(1) : href;
        const section = document.getElementById(sectionId);
        if (section) {
            sections.push({
                element: section,
                link: item,
                id: sectionId
            });
        }
    });

    if (sections.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    menuItems.forEach(link => link.classList.remove("active"));
                    const matched = sections.find(s => s.element === entry.target);
                    if (matched) {
                        matched.link.classList.add("active");
                    }
                }
            });
        }, {
            rootMargin: "-50% 0px -50% 0px",
            threshold: 0
        });

        sections.forEach(s => observer.observe(s.element));
    }

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