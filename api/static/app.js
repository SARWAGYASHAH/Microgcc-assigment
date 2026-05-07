const state = {
  states: [],
  health: null,
  comparison: null,
  selectedState: null,
  forecast: null,
  historical: null,
};

const el = {
  apiDot: document.querySelector("#apiDot"),
  apiStatus: document.querySelector("#apiStatus"),
  stateSelect: document.querySelector("#stateSelect"),
  refreshButton: document.querySelector("#refreshButton"),
  pageTitle: document.querySelector("#pageTitle"),
  statesCount: document.querySelector("#statesCount"),
  bestModel: document.querySelector("#bestModel"),
  bestMape: document.querySelector("#bestMape"),
  forecastHorizon: document.querySelector("#forecastHorizon"),
  forecastList: document.querySelector("#forecastList"),
  modelRows: document.querySelector("#modelRows"),
  salesChart: document.querySelector("#salesChart"),
  modelChart: document.querySelector("#modelChart"),
};

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${path} returned ${response.status}`);
  }
  return response.json();
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function setStatus(ok, text) {
  el.apiDot.classList.toggle("ok", ok);
  el.apiDot.classList.toggle("bad", !ok);
  el.apiStatus.textContent = text;
}

function getStateMetrics() {
  const details = state.comparison?.detailed_results?.[state.selectedState] || {};
  const best = state.comparison?.best_models?.[state.selectedState] || state.forecast?.best_model;
  return {
    best,
    metrics: best ? details[best]?.metrics : null,
    all: details,
  };
}

async function loadBaseData() {
  const [health, statesResponse, comparison] = await Promise.all([
    getJson("/health"),
    getJson("/states"),
    getJson("/models/comparison"),
  ]);

  state.health = health;
  state.states = statesResponse.states || [];
  state.comparison = comparison;
  state.selectedState = state.selectedState || state.states[0] || null;

  el.stateSelect.innerHTML = state.states
    .map((name) => `<option value="${name}">${name}</option>`)
    .join("");
  el.stateSelect.value = state.selectedState;
}

async function loadStateData() {
  if (!state.selectedState) return;
  const encoded = encodeURIComponent(state.selectedState);
  const [forecast, historical] = await Promise.all([
    getJson(`/forecast/${encoded}`),
    getJson(`/historical/${encoded}?limit=72`),
  ]);
  state.forecast = forecast;
  state.historical = historical;
}

function renderSummary() {
  const metrics = getStateMetrics();
  el.pageTitle.textContent = `${state.selectedState} sales forecast`;
  el.statesCount.textContent = state.health?.states_count ?? state.states.length;
  el.bestModel.textContent = metrics.best || "--";
  el.bestMape.textContent = metrics.metrics ? `${number(metrics.metrics.mape)}%` : "--";
  el.forecastHorizon.textContent = state.forecast
    ? `${state.forecast.forecast_horizon_weeks} weeks`
    : "--";
}

function renderForecastList() {
  const predictions = state.forecast?.predictions || [];
  el.forecastList.innerHTML = predictions
    .map((item) => `
      <div class="forecast-item">
        <span>${item.date}</span>
        <strong>${money(item.predicted_sales)}</strong>
      </div>
    `)
    .join("");
}

function renderModelRows() {
  const { best, all } = getStateMetrics();
  const rows = Object.entries(all)
    .sort((a, b) => (a[1].metrics?.mape || Infinity) - (b[1].metrics?.mape || Infinity))
    .map(([model, data]) => {
      const metrics = data.metrics || {};
      const rowClass = model === best ? "best-row" : "";
      return `
        <tr class="${rowClass}">
          <td>${model}</td>
          <td>${number(metrics.mape)}%</td>
          <td>${money(metrics.mae)}</td>
          <td>${money(metrics.rmse)}</td>
        </tr>
      `;
    })
    .join("");
  el.modelRows.innerHTML = rows || `<tr><td colspan="4">No comparison data available</td></tr>`;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(220, Math.floor(Number(canvas.getAttribute("height")) * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: canvas.width / ratio, height: canvas.height / ratio };
}

function drawGrid(ctx, width, height, padding) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#e5ebf3";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#647084";
  ctx.font = "12px system-ui, sans-serif";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + ((height - padding.top - padding.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
}

function drawLine(ctx, points, color) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  ctx.fillStyle = color;
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderSalesChart() {
  const { ctx, width, height } = setupCanvas(el.salesChart);
  const padding = { top: 18, right: 18, bottom: 36, left: 62 };
  drawGrid(ctx, width, height, padding);

  const historical = (state.historical?.data || []).map((item) => ({
    date: item.date,
    value: Number(item.sales),
    kind: "historical",
  }));
  const forecasts = (state.forecast?.predictions || []).map((item) => ({
    date: item.date,
    value: Number(item.predicted_sales),
    kind: "forecast",
  }));
  const combined = [...historical, ...forecasts];
  if (!combined.length) return;

  const values = combined.map((item) => item.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * 0.12, 1);
  const low = min - pad;
  const high = max + pad;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const toPoint = (item, index) => ({
    x: padding.left + (plotWidth * index) / Math.max(combined.length - 1, 1),
    y: padding.top + plotHeight - ((item.value - low) / (high - low)) * plotHeight,
    date: item.date,
  });

  const points = combined.map(toPoint);
  const splitIndex = historical.length;
  drawLine(ctx, points.slice(0, splitIndex), "#2f67d8");
  drawLine(ctx, points.slice(Math.max(0, splitIndex - 1)), "#d85f45");

  ctx.fillStyle = "#647084";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(money(high), 10, padding.top + 4);
  ctx.fillText(money(low), 10, height - padding.bottom);
  ctx.fillText(combined[0].date, padding.left, height - 12);
  ctx.textAlign = "right";
  ctx.fillText(combined[combined.length - 1].date, width - padding.right, height - 12);
  ctx.textAlign = "left";
}

function renderModelChart() {
  const { ctx, width, height } = setupCanvas(el.modelChart);
  const padding = { top: 20, right: 18, bottom: 46, left: 42 };
  drawGrid(ctx, width, height, padding);

  const entries = Object.entries(getStateMetrics().all)
    .map(([model, data]) => ({ model, mape: Number(data.metrics?.mape) }))
    .filter((item) => !Number.isNaN(item.mape));
  if (!entries.length) return;

  const max = Math.max(...entries.map((item) => item.mape), 1);
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const barWidth = Math.min(58, plotWidth / entries.length - 18);

  entries.forEach((item, index) => {
    const x = padding.left + (plotWidth * (index + 0.5)) / entries.length - barWidth / 2;
    const barHeight = (item.mape / max) * plotHeight;
    const y = padding.top + plotHeight - barHeight;
    ctx.fillStyle = item.model === getStateMetrics().best ? "#0f8b8d" : "#d85f45";
    ctx.fillRect(x, y, barWidth, barHeight);

    ctx.fillStyle = "#18212f";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${number(item.mape, 1)}%`, x + barWidth / 2, y - 6);
    ctx.fillStyle = "#647084";
    ctx.fillText(item.model, x + barWidth / 2, height - 16);
  });
  ctx.textAlign = "left";
}

function renderAll() {
  renderSummary();
  renderForecastList();
  renderModelRows();
  renderSalesChart();
  renderModelChart();
}

async function refresh() {
  try {
    setStatus(true, "Loading data");
    await loadBaseData();
    await loadStateData();
    renderAll();
    setStatus(true, "API healthy");
  } catch (error) {
    console.error(error);
    setStatus(false, error.message);
  }
}

el.stateSelect.addEventListener("change", async (event) => {
  state.selectedState = event.target.value;
  try {
    setStatus(true, "Loading state");
    await loadStateData();
    renderAll();
    setStatus(true, "API healthy");
  } catch (error) {
    console.error(error);
    setStatus(false, error.message);
  }
});

el.refreshButton.addEventListener("click", refresh);
window.addEventListener("resize", () => {
  renderSalesChart();
  renderModelChart();
});

refresh();
