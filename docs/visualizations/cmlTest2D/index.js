// Parameters
const controlsWidth = 220; // px, fixed width for controls column
const size = 512;
let alpha = 3.9;
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
// Change label to Unicode α, no colon
alphaGroup.append("label").text("α");
alphaGroup.append("input")
  .attr("type", "range")
  .attr("min", 2.5)
  .attr("max", 4.0)
  .attr("step", 0.01)
  .attr("value", alpha)
  .attr("id", "alphaSlider")
  .on("input", function() {
    alpha = +this.value;
    d3.select("#alphaVal").text(alpha);
  });
alphaGroup.append("span").attr("id", "alphaVal").text(alpha);

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
  .attr("max", 0.25)
  .attr("step", 0.01)
  .attr("value", epsilon)
  .attr("id", "epsilonSlider")
  .on("input", function() {
    epsilon = +this.value;
    d3.select("#epsilonVal").text(epsilon);
  });
epsilonGroup.append("span").attr("id", "epsilonVal").text(epsilon);

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

// GPU.js setup
const gpu = new GPU.GPU({ mode: 'gpu' });
const updateKernel = gpu.createKernel(function (lattice, alpha, epsilon) {
  const N = this.constants.size;
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

  const f_self = alpha * self * (1 - self);
  const f_up = alpha * up * (1 - up);
  const f_down = alpha * down * (1 - down);
  const f_left = alpha * left * (1 - left);
  const f_right = alpha * right * (1 - right);

  return (1 - 4 * epsilon) * f_self +
         epsilon * (f_up + f_down + f_left + f_right);
})
.setOutput([size, size])
.setConstants({ size });

// Initialize lattice
let lattice = [];
for (let i = 0; i < size; i++) {
  lattice[i] = [];
  for (let j = 0; j < size; j++) {
    lattice[i][j] = Math.random() * 0.5 + 0.25;
  }
}

// Visualization
function draw(lattice) {
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const v = lattice[i][j];
      // Use d3.interpolateMagma for color
      const color = d3.interpolateMagma(v);
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
    lattice = updateKernel(lattice, alpha, epsilon);
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
      lattice[i][j] = Math.random() * 0.5 + 0.25;
    }
    }
    draw(lattice);
});

// Initial draw
draw(lattice);
step();

// Handle window resize
window.addEventListener("resize", () => {
  canvasDim = getCanvasDim();
    canvasCol.style("width", `${canvasDim}px`).style("height", `${canvasDim}px`);
    canvas.style("width", `${canvasDim}px`).style("height", `${canvasDim}px`);
    }
);

// Initialize pause/play icon
updatePausePlayIcon();
