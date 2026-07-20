/**
 * TrafficIQ — Enterprise AI Traffic Management System
 * Dashboard Controller
 */

// ==========================================================
// CONSTANTS
// ==========================================================

const API_BASE = (() => {
  if (window.location.protocol.startsWith('http')) {
    const host = window.location.hostname;
    if (host === '127.0.0.1' || host === 'localhost') {
      return `http://${host}:8000`;
    }
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
})();

const API = {
  dashboard: `${API_BASE}/dashboard`,
  startVideo: `${API_BASE}/start-video`,
  stopVideo: `${API_BASE}/stop-video`,
  videoFeed: `${API_BASE}/video-feed`,
  health: `${API_BASE}/health`,
  analytics: `${API_BASE}/analytics`,
  density: `${API_BASE}/density`,
  signals: `${API_BASE}/signals`,
  emergency: `${API_BASE}/emergency`,
  restart: `${API_BASE}/restart-video`,
  perf: `${API_BASE}/perf`
};

const REFRESH_INTERVAL = 500;
const LANES = ['Lane_A', 'Lane_B', 'Lane_C', 'Lane_D'];
const MAX_TIMELINE_EVENTS = 50;

// ==========================================================
// STATE
// ==========================================================

let dashboardData = null;
let monitoringActive = false;
let eventTimeline = [];
let emergencyActive = false;
let chartInstances = {};

// ==========================================================
// DOM HELPERS
// ==========================================================

const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);
const qs = (sel) => document.querySelector(sel);

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function toggleClass(id, cls, force) {
  const el = $(id);
  if (el) el.classList.toggle(cls, force);
}

// ==========================================================
// CLOCK
// ==========================================================

function updateClock() {
  const now = new Date();
  const time = now.toLocaleTimeString('en-US', { hour12: false });
  const el = $('headerClock');
  if (el) el.textContent = time;
}
updateClock();
setInterval(updateClock, 1000);

// ==========================================================
// TOAST NOTIFICATIONS
// ==========================================================

function showToast(message, type = 'info') {
  const container = $('toastContainer');
  if (!container) return;

  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <span class="toast-message">${message}</span>
    <span class="toast-close" onclick="this.parentElement.remove()">×</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ==========================================================
// API
// ==========================================================

async function apiGet(endpoint) {
  try {
    const res = await fetch(endpoint);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('GET error:', err);
    return null;
  }
}

async function apiPost(endpoint) {
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('POST error:', err);
    return null;
  }
}

async function fetchDashboard() {
  const resp = await apiGet(API.dashboard);
  if (resp && resp.dashboard) {
    return { ...resp.dashboard, system: resp.system || {} };
  }
  return resp;
}

// ==========================================================
// ANIMATED COUNTER
// ==========================================================

function animateCounter(el, target, duration = 600) {
  if (!el) return;
  const start = parseInt(el.textContent.replace(/,/g, '')) || 0;
  const diff = target - start;
  if (diff === 0) return;
  
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + diff * eased);
    el.textContent = current.toLocaleString();
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      el.textContent = target.toLocaleString();
    }
  }
  requestAnimationFrame(update);
}

// ==========================================================
// SYSTEM STATUS
// ==========================================================

const STATUS_ITEMS = [
  { id: 'statusBackend', label: 'Backend', key: 'backend' },
  { id: 'statusAI', label: 'AI Engine', key: 'ai_engine' },
  { id: 'statusVideo', label: 'Video Feed', key: 'video_feed' },
  { id: 'statusCamera', label: 'Camera Status', key: 'camera' },
  { id: 'statusDetection', label: 'Detection', key: 'detection' },
  { id: 'statusTracker', label: 'Tracker Status', key: 'tracker' },
  { id: 'statusSignal', label: 'Signal Ctrl', key: 'signal_controller' },
  { id: 'statusEmergency', label: 'Emergency Mode', key: 'emergency_mode' },
  { id: 'statusGPU', label: 'GPU Usage', key: 'gpu' },
  { id: 'statusFPS', label: 'Inference FPS', key: 'inference_fps' }
];

function getStatusColor(value) {
  if (!value || value === 'Offline' || value === 'Inactive' || value === 'Stopped' || value === 'Disabled') return 'red';
  if (value === 'Active' || value === 'Running' || value === 'Online' || value === 'Loaded' || value === 'Enabled') return 'green';
  if (value === 'Warning' || value === 'Degraded') return 'yellow';
  return 'green';
}

function updateSystemStatus(data) {
  const system = data.system || {};
  const stats = data.statistics || {};
  
  // Determine statuses
  const backend = system.backend ? 'Online' : 'Offline';
  const processing = system.processing || monitoringActive;
  const emergencyMode = data.emergency?.active ? 'Active' : 'Standby';
  
  const statusMap = {
    backend,
    ai_engine: backend === 'Online' ? 'Running' : 'Offline',
    video_feed: processing ? 'Active' : 'Stopped',
    camera: processing ? 'Online' : 'Offline',
    detection: backend === 'Online' ? 'Active' : 'Inactive',
    tracker: backend === 'Online' ? 'Active' : 'Inactive',
    signal_controller: backend === 'Online' ? 'Active' : 'Inactive',
    emergency_mode: emergencyMode,
    gpu: data.fps ? `${data.fps} FPS` : 'Standby',
    inference_fps: `${data.fps || 0}`
  };

  STATUS_ITEMS.forEach(item => {
    const el = $(item.id);
    if (!el) return;
    const value = statusMap[item.key] || 'Offline';
    const color = getStatusColor(value);
    
    // Keep label
    const labelSpan = el.querySelector('.status-card-label');
    if (labelSpan) labelSpan.textContent = item.label;
    
    // Update value and dot
    const valueSpan = el.querySelector('.status-card-value');
    if (valueSpan) {
      const dot = valueSpan.querySelector('.dot');
      const textNode = valueSpan.childNodes[2] || valueSpan.lastChild;
      if (dot) {
        dot.className = `dot ${color}`;
        if (color === 'green') dot.classList.add('pulse');
      }
      // Set text after the dot
      const textEl = valueSpan.querySelector('.status-text');
      if (textEl) textEl.textContent = value;
    }
    
    el.className = `status-card status-${color}`;
  });

  // Sidebar status updates
  const sidebarStatuses = [
    { id: 'sidebarBackend', label: 'Backend', value: backend, color: getStatusColor(backend) },
    { id: 'sidebarAI', label: 'AI Engine', value: statusMap.ai_engine, color: getStatusColor(statusMap.ai_engine) },
    { id: 'sidebarCamera', label: 'Camera', value: statusMap.camera, color: getStatusColor(statusMap.camera) },
    { id: 'sidebarDetection', label: 'Detection', value: statusMap.detection, color: getStatusColor(statusMap.detection) }
  ];

  sidebarStatuses.forEach(s => {
    const el = $(s.id);
    if (!el) return;
    const dot = el.querySelector('.status-dot');
    const val = el.querySelector('.status-text');
    if (dot) { dot.className = `status-dot ${s.color}`; if (s.color === 'green') dot.classList.add('pulse'); }
    if (val) val.textContent = s.value;
  });
}

// ==========================================================
// STATISTICS
// ==========================================================

function updateStatistics(data) {
  const stats = data.statistics || {};
  const emergencySummary = data.emergency_summary || {};
  const signals = data.signals || {};
  const system = data.system || {};

  const totalVehicles = stats.total_vehicles || 0;
  const currentVehicles = Object.values(data.counter || {}).reduce((sum, lane) => sum + (lane.total || 0), 0);
  const emergencyCount = emergencySummary.current_count || 0;
  const activeLanes = Object.keys(signals).length || 0;
  const confidence = stats.confidence || '0%';
  const uptime = system.uptime || '00:00:00';

  // Animate counters
  animateCounter($('statVehiclesToday'), totalVehicles);
  animateCounter($('statCurrentVehicles'), currentVehicles);
  animateCounter($('statEmergency'), emergencyCount);
  
  setText('statSignalChanges', Object.keys(signals).length || '0');
  setText('statActiveLanes', activeLanes);
  
  // Average wait time
  let totalWait = 0;
  let waitCount = 0;
  for (const lane in signals) {
    if (signals[lane].wait_time) {
      totalWait += signals[lane].wait_time;
      waitCount++;
    }
  }
  const avgWait = waitCount > 0 ? Math.round(totalWait / waitCount) : 0;
  setText('statAvgWait', `${avgWait}s`);
  
  // Accuracy
  setText('statAccuracy', confidence);
  
  // Uptime
  setText('statUptime', uptime);
}

// ==========================================================
// CAMERA / VIDEO FEED
// ==========================================================

// Track whether we've already set the MJPEG stream URL
// to avoid reconnecting every refresh cycle
let _mjpegStreamSet = false;

function updateCameraFeed(data) {
  const img = $('cameraFeed');
  if (!img) return;
  
  // ============================================================
  // FIX: Only set img.src ONCE when monitoring starts.
  // Do NOT reset it every 500ms — that kills the MJPEG connection
  // and causes the browser to show a black screen.
  // ============================================================
  if (monitoringActive && !_mjpegStreamSet) {
    const feedUrl = `${API.videoFeed}?t=${Date.now()}`;
    console.log('[CameraFeed] Setting MJPEG stream URL (one-time):', feedUrl);
    img.src = feedUrl;
    img.style.display = 'block';
    _mjpegStreamSet = true;
  } else if (!monitoringActive) {
    // Reset the flag when monitoring stops
    _mjpegStreamSet = false;
    img.src = 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22 viewBox=%220 0 1280 720%22%3E%3Crect width=%221280%22 height=%22720%22 fill=%22070B14%22/%3E%3Crect x=%2230%22 y=%2230%22 width=%221220%22 height=%22660%22 rx=%2220%22 fill=%22101827%22/%3E%3Cpath d=%22M530 220h220c70 0 126 56 126 126v128c0 70-56 126-126 126H530c-70 0-126-56-126-126V346c0-70 56-126 126-126z%22 fill=%223B82F6%22 opacity=%220.5%22/%3E%3Ccircle cx=%22590%22 cy=%22410%22 r=%2270%22 fill=%22white%22 opacity=%220.1%22/%3E%3C/svg%3E';
  }

  // Update HUD overlay
  setText('hudFPS', data.fps || '0');
  setText('hudVehicles', data.statistics?.total_vehicles || '0');
  setText('hudEmergency', (data.emergency_summary?.current_count || '0'));
  
  const currentGreen = data.current_green || {};
  setText('hudSignal', currentGreen.lane ? currentGreen.lane.replace('_', ' ') : '--');
  
  setText('cameraName', 'Intersection A - Main');
  setText('hudInference', `${data.statistics?.confidence || '0%'}`);
  
  // Status bar
  setText('camFPS', data.fps || '0');
  setText('camFrame', data.frame_count || '0');
  setText('camDetections', data.statistics?.total_vehicles || '0');
  setText('camEmergency', data.emergency_summary?.current_count || '0');
  
  const greenLane = currentGreen.lane ? currentGreen.lane.replace('_', ' ') : '--';
  setText('camSignal', greenLane);
  setText('camInference', `${data.statistics?.confidence || '0%'}`);
}

// ==========================================================
// LANE STATUS CARDS
// ==========================================================

function updateLaneCards(data) {
  const counter = data.counter || {};
  const density = data.density?.class_density || {};
  const signals = data.signals || {};

  LANES.forEach(lane => {
    const laneId = lane.replace('_', '');
    const vehicles = counter[lane]?.total ?? 0;
    const laneDensity = density[lane]?.total ?? vehicles;
    const level = densityLevel(laneDensity);
    const green = signals[lane]?.green_time ?? 15;

    // Card
    const card = $(`laneCard${laneId}`);
    if (!card) return;

    // Name
    const nameEl = card.querySelector('.lane-name');
    if (nameEl) nameEl.textContent = lane.replace('_', ' ');

    // Density badge
    const densityEl = card.querySelector('.lane-density');
    if (densityEl) {
      densityEl.textContent = level;
      densityEl.className = `lane-density ${level.toLowerCase().replace(' ', '-')}`;
    }

    // Progress bar
    const progressBar = card.querySelector('.lane-progress-bar');
    if (progressBar) {
      const width = Math.min((laneDensity / 20) * 100, 100);
      progressBar.style.width = `${width}%`;
      const colorMap = { 'LOW': 'green', 'MEDIUM': 'blue', 'HIGH': 'amber', 'VERY HIGH': 'red' };
      progressBar.className = `lane-progress-bar ${colorMap[level] || 'blue'}`;
    }

    // Stats
    const vehiclesSpan = card.querySelector('.lane-vehicles');
    if (vehiclesSpan) vehiclesSpan.textContent = vehicles;
    
    const greenSpan = card.querySelector('.lane-green');
    if (greenSpan) greenSpan.textContent = `${green}s`;
  });
}

function densityLevel(total) {
  if (total <= 5) return 'LOW';
  if (total <= 10) return 'MEDIUM';
  if (total <= 20) return 'HIGH';
  return 'VERY HIGH';
}

// ==========================================================
// EVENT TIMELINE
// ==========================================================

function addTimelineEvent(icon, title, description, type = 'green') {
  const list = $('timelineList');
  if (!list) return;

  const time = new Date().toLocaleTimeString('en-US', { hour12: false });
  
  // Remove empty state if present
  const empty = list.querySelector('.timeline-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'timeline-item';
  item.innerHTML = `
    <div class="timeline-icon ${type}">${icon}</div>
    <div class="timeline-content">
      <h5>${title}</h5>
      <p>${description}</p>
    </div>
    <span class="timeline-time">${time}</span>
  `;
  
  list.insertBefore(item, list.firstChild);
  
  eventTimeline.push({ icon, title, description, type, time });
  if (eventTimeline.length > MAX_TIMELINE_EVENTS) {
    eventTimeline.shift();
    if (list.lastChild) list.removeChild(list.lastChild);
  }

  // Update count
  const countEl = $('timelineCount');
  if (countEl) countEl.textContent = `${Math.min(eventTimeline.length, MAX_TIMELINE_EVENTS)} Events`;
}

function addInitialTimelineEvents() {
  addTimelineEvent('⚡', 'System Initialized', 'TrafficIQ AI Engine loaded successfully', 'green');
  addTimelineEvent('📷', 'Camera Connected', 'Primary intersection feed active', 'blue');
  addTimelineEvent('🔄', 'Signal Controller', 'Adaptive mode ready for traffic management', 'cyan');
}

// ==========================================================
// EMERGENCY PANEL
// ==========================================================

function updateEmergencyPanel(data) {
  const emergency = data.emergency || {};
  const emergencySummary = data.emergency_summary || {};
  const panel = $('emergencyPanel');

  if (emergency.active && emergencySummary.current_count > 0) {
    panel.classList.add('active');
    emergencyActive = true;

    setText('emergencyVehicle', emergency.vehicle?.toUpperCase() || 'UNKNOWN');
    setText('emergencyConfidence', emergency.confidence ? `${Math.round(emergency.confidence * 100)}%` : '-');
    setText('emergencyLane', emergency.lane?.replace('_', ' ') || '-');
    
    const now = new Date();
    setText('emergencyTimestamp', now.toLocaleTimeString('en-US', { hour12: false }));
    
    const currentGreen = data.current_green || {};
    setText('emergencySignal', currentGreen.lane ? currentGreen.lane.replace('_', ' ') : '--');
    setText('emergencyEstimate', '~15 sec');
    setText('emergencyAction', emergency.override ? 'Signal Override — Green Wave Active' : 'Monitoring');

    if (!emergencyLogged) {
      addTimelineEvent('🚨', `${emergency.vehicle?.toUpperCase() || 'EMERGENCY'} Detected`, `Priority override on ${emergency.lane?.replace('_', ' ') || 'unknown lane'}`, 'red');
      showToast(`🚨 ${emergency.vehicle?.toUpperCase() || 'Emergency'} Detected on ${emergency.lane?.replace('_', ' ') || 'unknown lane'}`, 'error');
      emergencyLogged = true;
    }
  } else {
    panel.classList.remove('active');
    emergencyActive = false;
    emergencyLogged = false;
  }
}

let emergencyLogged = false;

// ==========================================================
// CHARTS
// ==========================================================

function initializeCharts() {
  createVehicleChart();
  createDensityChart();
  createDistributionChart();
  createTimelineChart();
}

function createVehicleChart() {
  const canvas = $('vehicleChart');
  if (!canvas) return;

  chartInstances.vehicle = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Vehicles',
        data: [],
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { 
          display: true,
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748B', font: { size: 10 } }
        }
      },
      animation: { duration: 300 }
    }
  });
}

function createDensityChart() {
  const canvas = $('densityChart');
  if (!canvas) return;

  chartInstances.density = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Lane A', 'Lane B', 'Lane C', 'Lane D'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#3B82F6', '#22C55E', '#F59E0B', '#EF4444'],
        borderRadius: 4,
        borderSkipped: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { 
          grid: { display: false },
          ticks: { color: '#64748B', font: { size: 10 } }
        },
        y: { 
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748B', font: { size: 10 } }
        }
      },
      animation: { duration: 300 }
    }
  });
}

function createDistributionChart() {
  const canvas = $('distributionChart');
  if (!canvas) return;

  chartInstances.distribution = new Chart(canvas.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Car', 'Bus', 'Van', 'Others'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#3B82F6', '#22C55E', '#F59E0B', '#8B5CF6'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '75%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94A3B8', padding: 12, font: { size: 11 } }
        }
      },
      animation: { duration: 300 }
    }
  });
}

function createTimelineChart() {
  const canvas = $('signalTimelineChart');
  if (!canvas) return;

  chartInstances.signalTimeline = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Green Time',
        data: [],
        borderColor: '#22C55E',
        backgroundColor: 'rgba(34,197,94,0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { 
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748B', font: { size: 10 } }
        }
      },
      animation: { duration: 300 }
    }
  });
}

function updateCharts(data) {
  if (!data) return;
  const time = new Date().toLocaleTimeString('en-US', { hour12: false });
  const stats = data.statistics || {};
  
  // Vehicle Trend
  if (chartInstances.vehicle) {
    const c = chartInstances.vehicle;
    c.data.labels.push(time);
    c.data.datasets[0].data.push(stats.total_vehicles || 0);
    if (c.data.labels.length > 20) {
      c.data.labels.shift();
      c.data.datasets[0].data.shift();
    }
    c.update('none');
  }

  // Lane Density
  if (chartInstances.density) {
    const density = data.density?.class_density || {};
    chartInstances.density.data.datasets[0].data = [
      density.Lane_A?.total || 0,
      density.Lane_B?.total || 0,
      density.Lane_C?.total || 0,
      density.Lane_D?.total || 0
    ];
    chartInstances.density.update('none');
  }

  // Distribution
  if (chartInstances.distribution) {
    chartInstances.distribution.data.datasets[0].data = [
      stats.cars || 0,
      stats.bus || 0,
      stats.van || 0,
      stats.others || 0
    ];
    chartInstances.distribution.update('none');
  }

  // Signal Timeline
  if (chartInstances.signalTimeline) {
    const signals = data.signals || {};
    let green = 0;
    for (const lane in signals) {
      green = signals[lane].green_time || 0;
      break;
    }
    const c = chartInstances.signalTimeline;
    c.data.labels.push(time);
    c.data.datasets[0].data.push(green);
    if (c.data.labels.length > 20) {
      c.data.labels.shift();
      c.data.datasets[0].data.shift();
    }
    c.update('none');
  }
}

// ==========================================================
// CONTROL BUTTONS
// ==========================================================

function setButtonLoading(btn, loading) {
  if (!btn) return;
  if (loading) {
    btn.classList.add('loading');
    btn.disabled = true;
    const icon = btn.querySelector('svg');
    const text = btn.querySelector('.btn-text');
    if (icon) icon.style.display = 'none';
    if (!btn.querySelector('.btn-spinner')) {
      const spinner = document.createElement('span');
      spinner.className = 'btn-spinner';
      btn.insertBefore(spinner, btn.firstChild);
    }
  } else {
    btn.classList.remove('loading');
    btn.disabled = false;
    const icon = btn.querySelector('svg');
    const spinner = btn.querySelector('.btn-spinner');
    if (icon) icon.style.display = '';
    if (spinner) spinner.remove();
  }
}

async function handleStartMonitoring() {
  const btn = $('btnStart');
  setButtonLoading(btn, true);
  
  try {
    const result = await apiPost(API.startVideo);
    if (result) {
      monitoringActive = true;
      // Reset MJPEG flag so the stream URL gets set on next refreshDashboard call
      _mjpegStreamSet = false;
      showToast('Monitoring started successfully', 'success');
      addTimelineEvent('▶️', 'Monitoring Started', 'Camera, detection, and tracking systems activated', 'green');
      updateButtonStates();
    } else {
      showToast('Failed to start monitoring', 'error');
    }
  } catch (err) {
    showToast('Error starting monitoring', 'error');
  }
  
  setButtonLoading(btn, false);
}

async function handleStopMonitoring() {
  const btn = $('btnStop');
  setButtonLoading(btn, true);
  
  try {
    const result = await apiPost(API.stopVideo);
    if (result) {
      monitoringActive = false;
      _mjpegStreamSet = false;
      showToast('Monitoring stopped', 'warning');
      addTimelineEvent('⏹️', 'Monitoring Stopped', 'All systems paused', 'yellow');
      updateButtonStates();
    } else {
      showToast('Failed to stop monitoring', 'error');
    }
  } catch (err) {
    showToast('Error stopping monitoring', 'error');
  }
  
  setButtonLoading(btn, false);
}

async function handleRestart() {
  const btn = $('btnRestart');
  setButtonLoading(btn, true);
  
  try {
    // Stop first
    await apiPost(API.stopVideo);
    await new Promise(r => setTimeout(r, 500));
    // Start
    const result = await apiPost(API.startVideo);
    if (result) {
      monitoringActive = true;
      _mjpegStreamSet = false;
      showToast('System restarted successfully', 'success');
      addTimelineEvent('🔄', 'System Restarted', 'Full system restart completed', 'cyan');
      updateButtonStates();
    }
  } catch (err) {
    showToast('Error restarting system', 'error');
  }
  
  setButtonLoading(btn, false);
}

function handleSwitchCamera() {
  showToast('Switching camera feed...', 'info');
  addTimelineEvent('📷', 'Camera Switched', 'Alternate camera feed activated', 'blue');
}

function handleSettings() {
  showToast('Settings panel opening...', 'info');
}

function updateButtonStates() {
  const startBtn = $('btnStart');
  const stopBtn = $('btnStop');
  const restartBtn = $('btnRestart');
  
  if (startBtn) startBtn.disabled = monitoringActive;
  if (stopBtn) stopBtn.disabled = !monitoringActive;
  if (restartBtn) restartBtn.disabled = false;
  
  // Update status text
  const statusEl = $('controlStatus');
  if (statusEl) {
    statusEl.textContent = monitoringActive ? 'System Running' : 'System Idle';
    statusEl.style.color = monitoringActive ? 'var(--green-400)' : 'var(--text-tertiary)';
  }
}

// ==========================================================
// SIDEBAR
// ==========================================================

function initializeSidebar() {
  const items = $$('.sidebar-nav-item');
  const sections = [];
  
  items.forEach(item => {
    const sectionId = item.dataset.section;
    if (sectionId) {
      const section = $(sectionId);
      if (section) {
        sections.push({ element: section, link: item, id: sectionId });
      }
    }
    
    item.addEventListener('click', () => {
      items.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      
      const sectionId = item.dataset.section;
      if (sectionId) {
        const target = $(sectionId);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
      
      // Close sidebar on mobile
      if (window.innerWidth <= 992) {
        toggleSidebar(false);
      }
    });
  });

  // Intersection Observer for active section
  if (sections.length > 0 && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          items.forEach(i => i.classList.remove('active'));
          const matched = sections.find(s => s.element === entry.target);
          if (matched) matched.link.classList.add('active');
        }
      });
    }, { rootMargin: '-50% 0px -50% 0px', threshold: 0 });

    sections.forEach(s => observer.observe(s.element));
  }
}

function toggleSidebar(force) {
  const sidebar = qs('.sidebar');
  const overlay = qs('.sidebar-overlay');
  if (!sidebar) return;
  
  if (typeof force === 'boolean') {
    sidebar.classList.toggle('open', force);
    if (overlay) overlay.classList.toggle('active', force);
  } else {
    sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('active');
  }
}

// ==========================================================
// DASHBOARD REFRESH
// ==========================================================

async function refreshDashboard() {
  const data = await fetchDashboard();
  
  if (!data) {
    const fallback = {
      statistics: {},
      density: { class_density: {} },
      signals: {},
      counter: {},
      emergency: { active: false },
      emergency_summary: { current_count: 0 },
      current_green: { lane: '-', green_time: 0 },
      system: { backend: false, processing: false },
      fps: 0,
      frame_count: 0
    };
    dashboardData = fallback;
    updateAll(fallback);
    return;
  }

  dashboardData = data;
  updateAll(data);
}

function updateAll(data) {
  updateSystemStatus(data);
  updateStatistics(data);
  updateCameraFeed(data);
  updateLaneCards(data);
  updateEmergencyPanel(data);
  updateCharts(data);
  updateButtonStates();
}

// ==========================================================
// INITIALIZATION
// ==========================================================

async function initializeDashboard() {
  console.log('🚦 TrafficIQ Dashboard Initializing...');
  
  initializeSidebar();
  initializeCharts();
  addInitialTimelineEvents();
  
  // Button handlers
  $('btnStart')?.addEventListener('click', handleStartMonitoring);
  $('btnStop')?.addEventListener('click', handleStopMonitoring);
  $('btnRestart')?.addEventListener('click', handleRestart);
  $('btnSwitchCamera')?.addEventListener('click', handleSwitchCamera);
  $('btnSettings')?.addEventListener('click', handleSettings);
  
  // Hamburger menu for mobile
  $('menuToggle')?.addEventListener('click', () => toggleSidebar());
  qs('.sidebar-overlay')?.addEventListener('click', () => toggleSidebar(false));
  
  // Initial data load
  try {
    await refreshDashboard();
  } catch (err) {
    console.warn('Initial dashboard load failed:', err);
  }
  
  // Periodic refresh
  setInterval(() => {
    refreshDashboard().catch(() => {});
  }, REFRESH_INTERVAL);
  
  console.log('✅ TrafficIQ Dashboard Ready');
}

// Start on DOM ready
document.addEventListener('DOMContentLoaded', initializeDashboard);