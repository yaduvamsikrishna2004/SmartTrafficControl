document.addEventListener('DOMContentLoaded', () => {
  const charts = document.querySelectorAll('[data-chart]');
  charts.forEach((chart) => {
    chart.textContent = 'Chart ready';
  });
});
