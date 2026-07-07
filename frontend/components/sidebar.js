document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('sidebar-root');
  const target = container || document.querySelector('.sidebar');

  if (!target) return;

  if (!container && target.querySelector('nav')) {
    return;
  }

  target.innerHTML = `
    <div class="logo">
      <h2>TrafficIQ</h2>
      <p>AI Traffic Intelligence</p>
    </div>

    <nav class="sidebar-menu">
      <a href="#" class="active"><i data-lucide="layout-dashboard"></i>Dashboard</a>
      <a href="#"><i data-lucide="video"></i>Monitoring</a>
      <a href="#"><i data-lucide="traffic-cone"></i>Signal Control</a>
      <a href="#"><i data-lucide="bar-chart-3"></i>Analytics</a>
      <a href="#"><i data-lucide="file-text"></i>Reports</a>
      <a href="#"><i data-lucide="settings"></i>Settings</a>
    </nav>
  `;
});
