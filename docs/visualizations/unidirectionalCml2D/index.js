// Parameters
const minControlsWidth = 240; // px, keep enough room for controls while maximizing square canvas
const size = 512;
let alpha = 3.9;
let epsilon = 0.2;
let stepsPerFrame = 16;

let boundaryConfig = {
  mode: "random",
  constant: 0.5,
  checkerA: 0.2,
  checkerB: 0.8,
  frequency: 0.05,
  amplitude: 0.25,
  offset: 0.5
};

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function boundaryValueFromConfig(i, config) {
  if (config.mode === "constant") {
    return clamp01(config.constant);
  }

  if (config.mode === "checkerboard") {
    return (i % 2 === 0) ? clamp01(config.checkerA) : clamp01(config.checkerB);
  }

  if (config.mode === "sinusoid") {
    const phase = 2 * Math.PI * config.frequency * i;
    return clamp01(config.offset + config.amplitude * Math.sin(phase));
  }

  return Math.random();
}

// Directional boundary conditions for incoming coupling.
// Edit these to customize the driving values at the left and bottom edges.
function leftBoundaryIC(i) {
  return boundaryValueFromConfig(i, boundaryConfig);
}

function bottomBoundaryIC(i) {
  return boundaryValueFromConfig(i, boundaryConfig);
}

let leftBoundary = new Array(size);
let bottomBoundary = new Array(size);
let boundaryPlot = null;

function activeBoundaryArray() {
  return leftBoundary;
}

function renderBoundaryPlot() {
  if (!boundaryPlot) return;

  const values = activeBoundaryArray();
  const data = Array.from({ length: size }, (_, index) => ({
    index,
    value: clamp01(values[index] ?? 0)
  }));

  const t = boundaryPlot.zoomTransform;
  const zx = t.rescaleX(boundaryPlot.xScale);
  const zy = t.rescaleY(boundaryPlot.yScale);

  boundaryPlot.pointLayer
    .selectAll("circle")
    .data(data, (d) => d.index)
    .join("circle")
    .attr("cx", (d) => zx(d.index))
    .attr("cy", (d) => zy(d.value))
    .attr("r", 2)
    .attr("fill", "#111")
    .attr("cursor", "ns-resize")
    .call(boundaryPlot.dragBehavior);

  boundaryPlot.linePath.attr(
    "d",
    d3.line()
      .x((d) => zx(d.index))
      .y((d) => zy(d.value))(data)
  );

  boundaryPlot.xAxisGroup.call(
    d3.axisBottom(zx)
      .ticks(4)
      .tickFormat(d3.format("d"))
  );

  boundaryPlot.yAxisGroup.call(
    d3.axisLeft(zy)
      .ticks(4)
      .tickFormat(d3.format(".2f"))
  );

}

function refreshBoundaryValues() {
  for (let i = 0; i < size; i++) {
    const v = leftBoundaryIC(i);
    leftBoundary[i] = v;
    bottomBoundary[i] = v;
  }
  renderBoundaryPlot();
}

refreshBoundaryValues();

// Layout
function getCanvasDim() {
  // Keep canvas square and let controls use all remaining width.
  return Math.max(100, Math.min(window.innerHeight, window.innerWidth - minControlsWidth));
}
let canvasDim = getCanvasDim();

// Main flexbox container
const container = d3.select("body")
  .append("div")
  .attr("id", "flex-container")
  .style("display", "flex")
  .style("flex-direction", "row")
  .style("justify-content", "space-evenly")
  .style("height", `${window.innerHeight}px`)
  .style("width", `${window.innerWidth}px`)
  .style("margin", "0")
  .style("padding", "0");

// Controls column
const controls = container
  .append("div")
  .attr("id", "controls-col")
  .style("flex", "1")
  .style("min-width", `${minControlsWidth}px`)
  .style("height", `${window.innerHeight}px`)
  .style("overflow-y", "auto")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("justify-content", "flex-start")
  .style("margin", "0")
  .style("padding", "0")
  .style("gap", "16px");

const controlsGrid = controls.append("div")
  .style("display", "grid")
  .style("grid-template-columns", "1fr 1fr")
  .style("grid-template-areas", '"panel-a panel-b" "panel-c panel-c"')
  .style("column-gap", "12px")
  .style("row-gap", "12px")
  .style("width", "100%")
  .style("box-sizing", "border-box")
  .style("padding", "0 8px 8px 8px");

const panelA = controlsGrid.append("div")
  .style("grid-area", "panel-a")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("justify-content", "flex-start")
  .style("gap", "12px")
  .style("min-height", "220px");

const panelB = controlsGrid.append("div")
  .style("grid-area", "panel-b")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("justify-content", "flex-start")
  .style("gap", "12px")
  .style("min-height", "220px");

const panelC = controlsGrid.append("div")
  .style("grid-area", "panel-c")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("justify-content", "flex-start")
  .style("gap", "10px")
  .style("width", "100%");

// Button group (above sliders, Unicode icons, only pause/play and reset)
const buttonGroup = panelA.append("div")
  .style("display", "flex")
  .style("flex-direction", "row")
  .style("gap", "32px")
  .style("margin", "0");

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
const alphaGroup = panelA.append("div")
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

const epsilonGroup = panelA.append("div")
  .style("display", "flex")
  .style("flex-direction", "column")
  .style("align-items", "center")
  .style("gap", "8px");
// Change label to Unicode ε, no colon
epsilonGroup.append("label").text("ε");
epsilonGroup.append("input")
  .attr("type", "range")
  .attr("min", -0.2)
  .attr("max", 1.1)
  .attr("step", 0.01)
  .attr("value", epsilon)
  .attr("id", "epsilonSlider")
  .on("input", function() {
    epsilon = +this.value;
    d3.select("#epsilonVal").text(epsilon);
  });
epsilonGroup.append("span").attr("id", "epsilonVal").text(epsilon);

function buildSharedBoundaryControl() {
  const section = panelC.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("align-items", "center")
    .style("gap", "8px")
    .style("width", "100%");

  section.append("div")
    .style("font-size", "1em")
    .text("BC");

  const modeSelect = section.append("select")
    .style("width", "170px")
    .style("font-size", "0.95em")
    .on("click", () => {
      refreshBoundaryValues();
      draw(lattice);
    })
    .on("change", function() {
      boundaryConfig.mode = this.value;
      updateParamRow();
      refreshBoundaryValues();
      draw(lattice);
    });

  modeSelect.selectAll("option")
    .data([
      { value: "random", label: "Random" },
      { value: "constant", label: "Constant" },
      { value: "checkerboard", label: "Checkerboard" },
      { value: "sinusoid", label: "Sinusoid" }
    ])
    .enter()
    .append("option")
    .attr("value", (d) => d.value)
    .text((d) => d.label);

  modeSelect.property("value", boundaryConfig.mode);

  const paramRow = section.append("div")
    .style("display", "grid")
    .style("grid-template-columns", "auto 56px auto 56px auto 56px")
    .style("column-gap", "6px")
    .style("row-gap", "4px")
    .style("align-items", "center")
    .style("justify-content", "center")
    .style("width", "170px");

  const p1Label = paramRow.append("span")
    .style("font-size", "0.8em")
    .style("text-align", "right");
  const p1Input = paramRow.append("input")
    .attr("type", "number")
    .style("width", "56px")
    .style("font-size", "0.8em")
    .on("input", function() {
      const v = +this.value;
      if (boundaryConfig.mode === "constant") {
        boundaryConfig.constant = clamp01(v);
      } else if (boundaryConfig.mode === "checkerboard") {
        boundaryConfig.checkerA = clamp01(v);
      } else if (boundaryConfig.mode === "sinusoid") {
        boundaryConfig.frequency = Math.max(0, v);
      }
      refreshBoundaryValues();
      draw(lattice);
    });

  const p2Label = paramRow.append("span")
    .style("font-size", "0.8em")
    .style("text-align", "right");
  const p2Input = paramRow.append("input")
    .attr("type", "number")
    .style("width", "56px")
    .style("font-size", "0.8em")
    .on("input", function() {
      const v = +this.value;
      if (boundaryConfig.mode === "checkerboard") {
        boundaryConfig.checkerB = clamp01(v);
      } else if (boundaryConfig.mode === "sinusoid") {
        boundaryConfig.amplitude = clamp01(v);
      }
      refreshBoundaryValues();
      draw(lattice);
    });

  const p3Label = paramRow.append("span")
    .style("font-size", "0.8em")
    .style("text-align", "right");
  const p3Input = paramRow.append("input")
    .attr("type", "number")
    .style("width", "56px")
    .style("font-size", "0.8em")
    .on("input", function() {
      if (boundaryConfig.mode !== "sinusoid") return;
      boundaryConfig.offset = clamp01(+this.value);
      refreshBoundaryValues();
      draw(lattice);
    });

  function hide(input, label) {
    input.style("display", "none");
    label.style("display", "none");
  }

  function show(input, label, labelText, value, min, max, step) {
    label.style("display", "inline").text(labelText);
    input
      .style("display", "inline-block")
      .attr("min", min)
      .attr("max", max)
      .attr("step", step)
      .property("value", value);
  }

  function updateParamRow() {
    hide(p1Input, p1Label);
    hide(p2Input, p2Label);
    hide(p3Input, p3Label);

    if (boundaryConfig.mode === "constant") {
      show(p1Input, p1Label, "c", boundaryConfig.constant, 0, 1, 0.01);
      return;
    }

    if (boundaryConfig.mode === "checkerboard") {
      show(p1Input, p1Label, "a", boundaryConfig.checkerA, 0, 1, 0.01);
      show(p2Input, p2Label, "b", boundaryConfig.checkerB, 0, 1, 0.01);
      return;
    }

    if (boundaryConfig.mode === "sinusoid") {
      show(p1Input, p1Label, "f", boundaryConfig.frequency, 0, 2, 0.01);
      show(p2Input, p2Label, "A", boundaryConfig.amplitude, 0, 1, 0.01);
      show(p3Input, p3Label, "o", boundaryConfig.offset, 0, 1, 0.01);
    }
  }

  updateParamRow();
}

buildSharedBoundaryControl();

function buildBoundaryPlotControl() {
  const section = panelC.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("align-items", "center")
    .style("gap", "8px")
    .style("width", "100%");

  const margin = { top: 8, right: 12, bottom: 22, left: 34 };
  const clipId = `boundary-clip-${Math.random().toString(36).slice(2)}`;

  const svg = section.append("svg")
    .style("border", "1px solid #aaa")
    .style("background", "#fff");

  const defs = svg.append("defs");
  const clipRect = defs.append("clipPath")
    .attr("id", clipId)
    .append("rect")
    .attr("x", 0)
    .attr("y", 0);

  const root = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear()
    .domain([0, size - 1])
    .range([0, 1]);

  const yScale = d3.scaleLinear()
    .domain([0, 1])
    .range([1, 0]);

  const xAxisGroup = root.append("g")
    .attr("transform", "translate(0,0)");

  const yAxisGroup = root.append("g");

  const plotViewport = root.append("g")
    .attr("clip-path", `url(#${clipId})`);

  const plotLayer = plotViewport.append("g");

  const lineGenerator = d3.line()
    .x((d) => xScale(d.index))
    .y((d) => yScale(d.value));

  const linePath = plotLayer.append("path")
    .attr("fill", "none")
    .attr("stroke", "#444")
    .attr("stroke-width", 1);

  const pointLayer = plotLayer.append("g");

  const dragBehavior = d3.drag()
    .on("start drag", (event, d) => {
      const pointer = d3.pointer(event, svg.node());
      const localY = pointer[1] - margin.top;
      const zy = boundaryPlot.zoomTransform.rescaleY(yScale);
      const newValue = clamp01(zy.invert(localY));
      leftBoundary[d.index] = newValue;
      bottomBoundary[d.index] = newValue;
      renderBoundaryPlot();
      draw(lattice);
    });

  const zoomBehavior = d3.zoom()
    .scaleExtent([1, 20])
    .on("zoom", (event) => {
      boundaryPlot.zoomTransform = event.transform;
      renderBoundaryPlot();
    });

  svg.call(zoomBehavior);

  boundaryPlot = {
    svg,
    xScale,
    yScale,
    xAxisGroup,
    yAxisGroup,
    lineGenerator,
    linePath,
    pointLayer,
    clipRect,
    root,
    margin,
    dragBehavior,
    zoomTransform: d3.zoomIdentity
  };

  function resizeBoundaryPlot() {
    const panelWidth = Math.floor(panelC.node().getBoundingClientRect().width);
    // panel-c spans two columns; size the plot to a single 1fr column.
    const oneFrWidth = Math.max(180, Math.floor((panelWidth - 12) / 2));
    const width = Math.max(180, oneFrWidth - 50);
    const height = Math.max(220, Math.floor(window.innerHeight * 0.34));
    const innerWidth = Math.max(10, width - margin.left - margin.right);
    const innerHeight = Math.max(10, height - margin.top - margin.bottom);

    svg
      .attr("width", width)
      .attr("height", height);

    root.attr("transform", `translate(${margin.left},${margin.top})`);

    clipRect
      .attr("width", innerWidth)
      .attr("height", innerHeight);

    xScale.range([0, innerWidth]);
    yScale.range([innerHeight, 0]);
    xAxisGroup.attr("transform", `translate(0,${innerHeight})`);

    renderBoundaryPlot();
  }

  boundaryPlot.resize = resizeBoundaryPlot;
  resizeBoundaryPlot();

  renderBoundaryPlot();
}

buildBoundaryPlotControl();

// Brush controls
const brushBtn = panelB.append("button")
  .attr("title", "Brush")
  .style("background", "none")
  .style("border", "none")
  .style("cursor", "pointer")
  .style("padding", "8px")
  .style("font-size", "2.5em")
  .text("🖌");

const brushSizeGroup = panelB.append("div")
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

const brushColorGroup = panelB.append("div")
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
const updateKernel = gpu.createKernel(function (lattice, alpha, epsilon, leftBoundary, bottomBoundary) {
  const N = this.constants.size;
  const i = this.thread.y;
  const j = this.thread.x;

  const self = lattice[i][j];
  const fromBottom = (i === N - 1) ? bottomBoundary[j] : lattice[i + 1][j];
  const fromLeft = (j === 0) ? leftBoundary[i] : lattice[i][j - 1];

  const f_self = alpha * self * (1 - self);
  const f_bottom = alpha * fromBottom * (1 - fromBottom);
  const f_left = alpha * fromLeft * (1 - fromLeft);

  // Directional coupling: each site receives from bottom and left only.
  return (1 - epsilon) * f_self +
         (epsilon / 2.0) * (f_bottom + f_left);
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
refreshBoundaryValues();

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
    lattice = updateKernel(lattice, alpha, epsilon, leftBoundary, bottomBoundary);
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
  refreshBoundaryValues();
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      lattice[i][j] = Math.random() * 0.5 + 0.25;
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
  container
    .style("height", `${window.innerHeight}px`)
    .style("width", `${window.innerWidth}px`);
  controls.style("height", `${window.innerHeight}px`);
  canvasCol
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
  d3.select(canvas)
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
  d3.select(overlayCanvas)
    .style("width", `${canvasDim}px`)
    .style("height", `${canvasDim}px`);
  if (boundaryPlot && boundaryPlot.resize) {
    boundaryPlot.resize();
  }
});

// Initialize pause/play icon
updatePausePlayIcon();
