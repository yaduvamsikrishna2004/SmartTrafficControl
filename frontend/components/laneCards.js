document.addEventListener('DOMContentLoaded', () => {
  const laneCards = document.querySelectorAll('[data-lane-card]');
  laneCards.forEach((card) => {
    card.classList.add('lane-card');
  });
});
