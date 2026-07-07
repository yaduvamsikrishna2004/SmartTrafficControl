document.addEventListener('DOMContentLoaded', () => {
  const header = document.getElementById('header-root');
  if (!header) return;

  header.innerHTML = `
    <header class="page-header">
      <h1>Smart Traffic Control</h1>
      <nav>
        <a href="index.html">Dashboard</a>
        <a href="monitor.html">Monitor</a>
        <a href="analytics.html">Analytics</a>
        <a href="reports.html">Reports</a>
        <a href="settings.html">Settings</a>
      </nav>
    </header>
  `;
});
