document.addEventListener('DOMContentLoaded', async () => {
  const status = document.querySelector('[data-status]');
  const density = document.querySelector('[data-density]');
  const signals = document.querySelector('[data-signals]');
  const detections = document.querySelector('[data-detections]');

  if (status) {
    const data = await fetchTrafficData();
    status.textContent = `System status: ${data.status}`;
  }

  if (density) density.textContent = 'Medium';
  if (signals) signals.textContent = '8';
  if (detections) detections.textContent = '24';
});
