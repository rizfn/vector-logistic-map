// Parameters
const controlsWidth = 220; // px, fixed width for controls column
const size = 512;
let b = 1; // Amplitude parameter
const alpha_fixed = 10.0; // Fixed alpha for Gaussian
let epsilon = 0.2;
let stepsPerFrame = 8;

// Layout
function getCanvasDim() {
  // Always fit canvas inside viewport minus controls
  return Math.min(window.innerHeight, window.innerWidth - controlsWidth);
}
let canvasDim = getCanvasDim();

// Main flexbox container
const container = d3.select("body")
  .append("div")
  .attr("id", "flex-container")
  .style("display", "flex")
  .style("flex-direction", "row")
  .style("height", `${window.innerHeight}px`)
  .style("width", `${window.innerWidth}px`)
  .style("margin", "0")
  .style("padding", "0");

// Controls column
const controls = container
  .append("div")
  .attr("id", "controls-col")
  .style("width", `${controlsWidth}px`)
  .style("height", `${window.innerHeight}px`)
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("justify-content", "center")
  .style("gap", "24px");

// Equation display at top
const equationDisplay = controls.append("div")
  .style("font-size", "0.9em")
  .style("text-align", "center")
  .style("padding", "8px")
  .style("margin-bottom", "16px")
  .html("f(x) = b e<sup>-α(x-0.5)²</sup><br>α = 10");

// Button group (above sliders, Unicode icons, only pause/play and reset)
const buttonGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "row")
  .style("gap", "32px")
  .style("margin-bottom", "32px");

// Pause/Play button (Unicode)
const pausePlayBtn = buttonGroup.append("button")
  .attr("title", "Pause/Play")
  .style("background", "none")
  .style("border", "none")
  .style("cursor", "pointer")
  .style("padding", "8px")
  .style("font-size", "2.5em")
  .text("⏸");

// Reset button (Unicode, clockwise)
const resetBtn = buttonGroup.append("button")
  .attr("title", "Reset")
  .style("background", "none")
  .style("border", "none")
  .style("cursor", "pointer")
  .style("padding", "8px")
  .style("font-size", "2.5em")
  .text("↻");

// D3 UI (sliders below buttons)
const alphaGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px");
alphaGroup.append("label").text("b");
alphaGroup.append("input")
  .attr("type", "range")
  .attr("min", 0.5)
  .attr("max", 1.4)
  .attr("step", 0.01)
  .attr("value", b)
  .attr("id", "alphaSlider")
  .on("input", function() {
    b = +this.value;
    d3.select("#alphaVal").text(b.toFixed(2));
  });
alphaGroup.append("span").attr("id", "alphaVal").text(b.toFixed(2));

const epsilonGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px");
// Change label to Unicode ε, no colon
epsilonGroup.append("label").text("ε");
epsilonGroup.append("input")
  .attr("type", "range")
  .attr("min", 0)
  .attr("max", 1.1)
  .attr("step", 0.01)
  .attr("value", epsilon)
  .attr("id", "epsilonSlider")
  .on("input", function() {
    epsilon = +this.value;
    d3.select("#epsilonVal").text(epsilon);
  });
epsilonGroup.append("span").attr("id", "epsilonVal").text(epsilon);

// Brush controls
const brushBtn = controls.append("button")
  .attr("title", "Brush")
  .style("background", "none")
  .style("border", "none")
  .style("cursor", "pointer")
  .style("padding", "8px")
  .style("font-size", "2.5em")
  .text("🖌");

const brushSizeGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px")
  .style("visibility", "hidden");
brushSizeGroup.append("label").text("Brush Size");
brushSizeGroup.append("input")
  .attr("type", "range")
  .attr("min", 1)
  .attr("max", 50)
  .attr("step", 1)
  .attr("value", 10)
  .attr("id", "brushSizeSlider")
  .on("input", function() {
    brushSize = +this.value;
    d3.select("#brushSizeVal").text(brushSize);
  });
brushSizeGroup.append("span").attr("id", "brushSizeVal").text(10);

const brushColorGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px")
  .style("visibility", "hidden");
brushColorGroup.append("label").text("Brush Value");
brushColorGroup.append("input")
  .attr("type", "range")
  .attr("min", 0)
  .attr("max", 1)
  .attr("step", 0.01)
  .attr("value", 0.5)
  .attr("id", "brushColorSlider")
  .on("input", function() {
    brushValue = +this.value;
    d3.select("#brushColorVal").text(brushValue.toFixed(2));
    const color = d3.interpolateViridis(brushValue);
    d3.select("#brushColorPreview").style("background-color", color);
  });
brushColorGroup.append("span").attr("id", "brushColorVal").text("0.50");
brushColorGroup.append("div")
  .attr("id", "brushColorPreview")
  .style("width", "40px")
  .style("height", "20px")
  .style("background-color", d3.interpolateViridis(0.5))
  .style("border", "1px solid #000");

// Brush state
let brushSize = 10;
let brushValue = 0.5;
let isDrawing = false;
let brushEnabled = false;

// Canvas column
const canvasCol = container
  .append("div")
  .attr("id", "canvas-col")
  .style("flex", "none")
  .style("width", `${canvasDim}px`)
  .style("height", `${canvasDim}px`)
  .style("display", "flex")
  .style("align-items", "center")
  .style("justify-content", "center");

const canvas = canvasCol
  .append("canvas")
  .attr("width", size)
  .attr("height", size)
  .style("width", `${canvasDim}px`)
  .style("height", `${canvasDim}px`)
  .node();
const ctx = canvas.getContext('2d');

// Overlay canvas for brush preview
const overlayCanvas = canvasCol
  .append("canvas")
  .attr("width", size)
  .attr("height", size)
  .style("width", `${canvasDim}px`)
  .style("height", `${canvasDim}px`)
  .style("position", "absolute")
  .style("pointer-events", "none")
  .style("display", "none")
  .node();
const overlayCtx = overlayCanvas.getContext('2d');

// GPU.js setup
const gpu = new GPU.GPU({ mode: 'gpu' });
const updateKernel = gpu.createKernel(function (lattice, b_param, epsilon) {
  const N = this.constants.size;
  const alpha_fixed = this.constants.alpha_fixed;
  const i = this.thread.y;
  const j = this.thread.x;

  function idx(x, N) {
    return (x + N) % N;
  }

  const self = lattice[i][j];
  const up = lattice[idx(i - 1, N)][j];
  const down = lattice[idx(i + 1, N)][j];
  const left = lattice[i][idx(j - 1, N)];
  const right = lattice[i][idx(j + 1, N)];

  // Gaussian map: f(x) = b * exp(-alpha * (x - 0.5)^2)
  const f_self = b_param * Math.exp(-alpha_fixed * (self - 0.5) * (self - 0.5));
  const f_up = b_param * Math.exp(-alpha_fixed * (up - 0.5) * (up - 0.5));
  const f_down = b_param * Math.exp(-alpha_fixed * (down - 0.5) * (down - 0.5));
  const f_left = b_param * Math.exp(-alpha_fixed * (left - 0.5) * (left - 0.5));
  const f_right = b_param * Math.exp(-alpha_fixed * (right - 0.5) * (right - 0.5));

  // Each neighbor gets epsilon/4, self gets 1-epsilon
  return (1 - epsilon) * f_self +
         (epsilon / 4.0) * (f_up + f_down + f_left + f_right);
})
.setOutput([size, size])
.setConstants({ size, alpha_fixed: alpha_fixed });

// Initialize lattice
let lattice = [];
for (let i = 0; i < size; i++) {
  lattice[i] = [];
  for (let j = 0; j < size; j++) {
    lattice[i][j] = Math.random() * 0.6 + 0.2; // Range [0.2, 0.8] around 0.5
  }
}

// Visualization
function draw(lattice) {
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const v = lattice[i][j];
      // Normalize for visualization (values centered around 0.5, range ~[0, 1.5])
      const normalized = Math.min(Math.max(v, 0), 1);
      const color = d3.interpolateMagma(normalized);
      const rgb = d3.color(color);
      const idx = 4 * (i * size + j);
      img.data[idx] = rgb.r;
      img.data[idx + 1] = rgb.g;
      img.data[idx + 2] = rgb.b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Animation control
let running = true;
let rafId = null;

function updatePausePlayIcon() {
  pausePlayBtn.text(running ? "⏸" : "▶");
}

function step() {
  if (!running) return;
  for (let s = 0; s < stepsPerFrame; s++) {
    lattice = updateKernel(lattice, b, epsilon);
  }
  draw(lattice);
  rafId = requestAnimationFrame(step);
}

// Pause/Play button event handler
pausePlayBtn.on("click", () => {
  running = !running;
  updatePausePlayIcon();
  if (running) {
    step();
  } else {
    if (rafId) cancelAnimationFrame(rafId);
  }
});

// Reset button event handler
resetBtn.on("click", () => {
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      lattice[i][j] = Math.random() * 0.6 + 0.2;
    }
  }
  draw(lattice);
});

// Brush button toggle
brushBtn.on("click", () => {
  brushEnabled = !brushEnabled;
  d3.select(overlayCanvas).style("display", brushEnabled ? "block" : "none");
  brushBtn.style("opacity", brushEnabled ? "1" : "0.5");
  brushSizeGroup.style("visibility", brushEnabled ? "visible" : "hidden");
  brushColorGroup.style("visibility", brushEnabled ? "visible" : "hidden");
});

// Initial draw
draw(lattice);
step();

// Mouse interaction
d3.select(canvas)
  .on("mousedown", (event) => {
    if (!brushEnabled) return;
    isDrawing = true;
    paintAt(event);
  })
  .on("mousemove", (event) => {
    if (!brushEnabled) return;
    drawBrushPreview(event);
    if (isDrawing) paintAt(event);
  })
  .on("mouseup", () => {
    if (!brushEnabled) return;
    isDrawing = false;
  })
  .on("mouseleave", () => {
    if (!brushEnabled) return;
    isDrawing = false;
    overlayCtx.clearRect(0, 0, size, size);
  });

function paintAt(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) * size / canvasDim);
  const y = Math.floor((event.clientY - rect.top) * size / canvasDim);
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const dx = i - y;
      const dy = j - x;
      if (dx * dx + dy * dy <= brushSize * brushSize) {
        lattice[i][j] = brushValue;
      }
    }
  }
  draw(lattice);
}

function drawBrushPreview(event) {
  overlayCtx.clearRect(0, 0, size, size);
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * size / canvasDim;
  const y = (event.clientY - rect.top) * size / canvasDim;
  
  overlayCtx.strokeStyle = 'white';
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.arc(x, y, brushSize, 0, 2 * Math.PI);
  overlayCtx.stroke();
}

// Handle window resize
window.addEventListener("resize", () => {
  canvasDim = getCanvasDim();
  canvasCol
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
  d3.select(canvas)
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
  d3.select(overlayCanvas)
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
});

// Initialize pause/play icon
updatePausePlayIcon();
