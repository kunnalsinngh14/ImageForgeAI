// Gradio Tab Navigation
function switchTab(tabId) {
  ['generator', 'benchmarks', 'diagnostics'].forEach(t => {
    const pane = document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1));
    const nav = document.getElementById('tabNav' + t.charAt(0).toUpperCase() + t.slice(1));
    if (t === tabId) {
      pane.style.display = 'block';
      nav.classList.add('active');
    } else {
      pane.style.display = 'none';
      nav.classList.remove('active');
    }
  });
}

// Accordion Toggles
function toggleAccordion(id) {
  const body = document.getElementById(id);
  const icon = document.getElementById(id + 'Icon');
  if (body.style.display === 'none' || body.style.display === '') {
    body.style.display = 'flex';
    if (icon) icon.textContent = '▲';
  } else {
    body.style.display = 'none';
    if (icon) icon.textContent = '▼';
  }
}

// Mode selection
let currentMode = 'adaptive';
let currentResolution = '512x512';

function setMode(mode) {
  currentMode = mode;
  ['Adaptive', 'Optimized', 'Baseline'].forEach(m => {
    const pill = document.getElementById('pill' + m);
    if (m.toLowerCase() === mode) {
      pill.classList.add('active');
    } else {
      pill.classList.remove('active');
    }
  });
  
  const info = document.getElementById('modeInfo');
  if (mode === 'adaptive') {
    info.textContent = 'Adaptive: Dynamic resource budgeting • Optimized: DPM-Solver++ 20 steps with guided filtering • Baseline: Unoptimized reference';
  } else if (mode === 'optimized') {
    info.textContent = 'Optimized: 20 steps DPM-Solver++ with guided filtering and fp16 weights';
  } else {
    info.textContent = 'Baseline: Standard 50-step diffusion reference baseline without step pruning';
  }
}

function setResolution(res) {
  currentResolution = res;
}

// Generation simulation with step updates matching Gradio generator callback
function generateImage() {
  const btn = document.getElementById('generateBtn');
  const status = document.getElementById('statusInput');
  const steps = parseInt(document.getElementById('stepsInput').value) || 20;
  
  btn.disabled = true;
  btn.style.opacity = '0.5';
  btn.style.cursor = 'not-allowed';
  
  status.value = `Initializing ${currentMode} pipeline (${currentResolution}, ${steps} steps)...`;
  status.style.color = '#eab308';
  
  let currentStep = 1;
  const interval = setInterval(() => {
    if (currentStep <= steps) {
      const pct = Math.round((currentStep / steps) * 100);
      const remaining = ((steps - currentStep) * 0.38).toFixed(1);
      status.value = `Step ${currentStep}/${steps} (${pct}%) • 0.38s/step (~${remaining}s remaining)`;
      currentStep += Math.max(1, Math.floor(steps / 8));
    } else {
      clearInterval(interval);
      status.value = "Applying classical CV post-processing (LAB-space CLAHE and guided edge filtering)...";
      
      setTimeout(() => {
        status.value = "Complete";
        status.style.color = "#10b981";
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        
        // Update metrics
        let lat = currentMode === 'adaptive' ? 2.31 : (currentMode === 'optimized' ? 3.12 : 7.20);
        let denoise = (lat * 0.84).toFixed(2);
        let vram = currentResolution === '512x512' ? '1,365 MB' : (currentResolution === '384x384' ? '980 MB' : '720 MB');
        
        document.getElementById('mLatency').textContent = lat.toFixed(2) + 's';
        document.getElementById('mDenoise').textContent = denoise + 's';
        document.getElementById('mRes').textContent = currentResolution;
        document.getElementById('mVram').textContent = vram;
        document.getElementById('mMode').textContent = currentMode;
        
        // Trigger reveal animation
        const img = document.getElementById('outputImage');
        img.classList.remove('image-reveal');
        void img.offsetWidth;
        img.classList.add('image-reveal');
      }, 500);
    }
  }, 150);
}

function downloadImage(imgId, filename) {
  const src = document.getElementById(imgId).src;
  const a = document.createElement('a');
  a.href = src;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function toggleFullscreen(frameId) {
  const frame = document.getElementById(frameId);
  if (!document.fullscreenElement) {
    frame.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}
