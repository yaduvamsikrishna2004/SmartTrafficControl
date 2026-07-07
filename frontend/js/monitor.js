document.addEventListener('DOMContentLoaded', () => {
  const panel = document.getElementById('monitor-map');
  if (panel) {
    panel.innerHTML = '<p>Live camera feed and AI detections will be rendered here.</p>';
  }
});
