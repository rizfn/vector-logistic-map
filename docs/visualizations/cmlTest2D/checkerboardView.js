// Parameters
const controlsWidth = 220;
const size = 512;
const displaySize = size / 2; // Subsample: only i,j both even → 256×256
let alpha = 3.9;
let epsilon = 0.2;
let stepsPerFrame = 16; // Draw every 16 steps instead of 8

// Layout
function getCanvasDim() {
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
  .style("font-size", "0.8em")
  .style("text-align", "center")
  .style("padding", "8px")
  .style("margin-bottom", "16px")
  .html("Subsampled view<br>Every 2nd site (i,j both even)");

// Button group
const buttonGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "row")
  .style("gap", "32px")
  .style("margin-bottom", "32px");

const pausePlayBtn = buttonGroup.append("button")
  .attr("title", "Pause/Play")
  .style("background", "none")
  .style("border", "none")
  .style("cursor", "pointer")
  .style("padding", "8px")
  .style("font-size", "2.5em")
  .text("⏸");

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
    d3.select("#alphaVal").text(alpha.toFixed(2));
  });
alphaGroup.append("span").attr("id", "alphaVal").text(alpha.toFixed(2));

const epsilonGroup = controls.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px");
epsilonGroup.append("label").text("ε");
epsilonGroup.append("input")
  .attr("type", "range")
  .attr("min", 0)
  .attr("max", 1.2)
  .attr("step", 0.01)
  .attr("value", epsilon)
  .attr("id", "epsilonSlider")
  .on("input", function() {
    epsilon = +this.value;
    d3.select("#epsilonVal").text(epsilon.toFixed(2));
  });
epsilonGroup.append("span").attr("id", "epsilonVal").text(epsilon.toFixed(2));

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
    const color = d3.interpolateMagma(brushValue);
    d3.select("#brushColorPreview").style("background-color", color);
  });
brushColorGroup.append("span").attr("id", "brushColorVal").text("0.50");
brushColorGroup.append("div")
  .attr("id", "brushColorPreview")
  .style("width", "40px")
  .style("height", "20px")
  .style("background-color", d3.interpolateMagma(0.5))
  .style("border", "1px solid #000");

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
  .attr("width", displaySize)
  .attr("height", displaySize)
  .style("width", `${canvasDim}px`)
  .style("height", `${canvasDim}px`)
  .node();
const ctx = canvas.getContext('2d');

const overlayCanvas = canvasCol
  .append("canvas")
  .attr("width", displaySize)
  .attr("height", displaySize)
  .style("width", `${canvasDim}px`)
  .style("height", `${canvasDim}px`)
  .style("position", "absolute")
  .style("pointer-events", "none")
  .style("display", "none")
  .node();
const overlayCtx = overlayCanvas.getContext('2d');

// GPU.js setup - same update kernel
const gpu = (() => {
  try {
    return new GPU.GPU();
  } catch (e) {
    console.warn("GPU.GPU constructor failed, falling back to GPU. This may be a workaround for Safari on macOS.");
    return new GPU();
  }
})();

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

  return (1 - epsilon) * f_self +
         (epsilon / 4.0) * (f_up + f_down + f_left + f_right);
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

// Visualization - subsample only sites where BOTH i and j are even
function draw(lattice) {
  const img = ctx.createImageData(displaySize, displaySize);
  for (let i = 0; i < displaySize; i++) {
    for (let j = 0; j < displaySize; j++) {
      // Map display coordinates to lattice coordinates
      // displaySize=256 maps to size=512 with step of 2
      const latticeI = i * 2;
      const latticeJ = j * 2;
      
      const v = lattice[latticeI][latticeJ];
      const color = d3.interpolateMagma(v);
      const rgb = d3.color(color);
      const idx = 4 * (i * displaySize + j);
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

pausePlayBtn.on("click", () => {
  running = !running;
  updatePausePlayIcon();
  if (running) {
    step();
  } else {
    if (rafId) cancelAnimationFrame(rafId);
  }
});

resetBtn.on("click", () => {
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      lattice[i][j] = Math.random() * 0.5 + 0.25;
    }
  }
  draw(lattice);
});

brushBtn.on("click", () => {
  brushEnabled = !brushEnabled;
  d3.select(overlayCanvas).style("display", brushEnabled ? "block" : "none");
  brushBtn.style("opacity", brushEnabled ? "1" : "0.5");
  brushSizeGroup.style("visibility", brushEnabled ? "visible" : "hidden");
  brushColorGroup.style("visibility", brushEnabled ? "visible" : "hidden");
});

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
  // Map from display coordinates to lattice coordinates
  const displayX = Math.floor((event.clientX - rect.left) * displaySize / canvasDim);
  const displayY = Math.floor((event.clientY - rect.top) * displaySize / canvasDim);
  
  // Convert display to lattice: multiply by 2 since we're subsampling
  const latticeX = displayX * 2;
  const latticeY = displayY * 2;
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const dx = i - latticeY;
      const dy = j - latticeX;
      if (dx * dx + dy * dy <= brushSize * brushSize) {
        lattice[i][j] = brushValue;
      }
    }
  }
  draw(lattice);
}

function drawBrushPreview(event) {
  overlayCtx.clearRect(0, 0, displaySize, displaySize);
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * displaySize / canvasDim;
  const y = (event.clientY - rect.top) * displaySize / canvasDim;
  
  overlayCtx.strokeStyle = 'white';
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  // Brush size is in lattice coordinates, scale down by 2 for display
  overlayCtx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
  overlayCtx.stroke();
}

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

updatePausePlayIcon();
