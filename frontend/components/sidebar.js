document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('sidebar-root');
  if (!container) return;

  container.innerHTML = `
    <aside class="sidebar">
      <h3>Navigation</h3>
      <ul>
        <li><a href="index.html">Dashboard</a></li>
        <li><a href="monitor.html">Monitor</a></li>
        <li><a href="analytics.html">Analytics</a></li>
        <li><a href="reports.html">Reports</a></li>
        <li><a href="settings.html">Settings</a></li>
      </ul>
    </aside>
  `;
});
