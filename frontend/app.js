const hasDocument = typeof document !== "undefined";
const byId = (id) => (hasDocument ? document.getElementById(id) : null);

const healthStatus = byId("healthStatus");
const queryInput = byId("queryInput");
const sendBtn = byId("sendBtn");
const chatStatus = byId("chatStatus");
const progressStatusBadge = byId("progressStatusBadge");
const progressMeterFill = byId("progressMeterFill");
const progressCurrentTitle = byId("progressCurrentTitle");
const progressElapsed = byId("progressElapsed");
const progressNarrative = byId("progressNarrative");
const progressStageList = byId("progressStageList");
const progressActivityLog = byId("progressActivityLog");
const outputView = byId("outputView");
const answerText = byId("answerText");
const taskType = byId("taskType");
const usedTools = byId("usedTools");
const traceIdText = byId("traceIdText");
const numericEvidenceList = byId("numericEvidenceList");
const textEvidenceList = byId("textEvidenceList");
const limitationsList = byId("limitationsList");
const citationsList = byId("citationsList");
const traceInput = byId("traceInput");
const traceBtn = byId("traceBtn");
const debugBundleBtn = byId("debugBundleBtn");
const copyTraceStatus = byId("copyTraceStatus");
const traceAudit = byId("traceAudit");
const traceTabContent = byId("traceTabContent");
const traceSummary = byId("traceSummary");
const traceJson = byId("traceJson");
let traceLoadTimer = null;
let lastLoadedTraceId = "";
let latestAnswerTraceId = "";
let currentTraceUi = null;
let currentTraceTab = "workflow";
let progressTimer = null;
let progressStartedAt = 0;
let progressLogItems = [];
let progressPollTimer = null;
let progressPollingTraceId = "";
let progressPollStartedAt = 0;
let progressUsingRealEvents = false;
let progressLastRealEvent = "";

const runProgressStages = [
  {
    key: "intent",
    label: "Intent",
    start: 0,
    title: "正在识别研究意图",
    text: "系统正在判断公司、问题类型、时间范围和安全边界，准备把自然语言问题转成可执行计划。",
  },
  {
    key: "plan",
    label: "Plan",
    start: 6,
    title: "正在生成研究计划",
    text: "系统正在生成 ResearchPlan 和 EvidencePlan，把问题拆成必须回答的部分、证据请求和限制条件。",
  },
  {
    key: "data",
    label: "Data",
    start: 16,
    title: "正在读取结构化财务数据",
    text: "系统正在查询本地财务数据库，并整理收入、利润、现金流、估值等可追溯指标。",
  },
  {
    key: "filings",
    label: "Filings",
    start: 32,
    title: "正在检索 SEC filing 证据",
    text: "系统正在查找 10-K / 10-Q 文本片段，用来支撑风险、竞争、业务模式和管理层讨论。",
  },
  {
    key: "synthesis",
    label: "Synthesis",
    start: 54,
    title: "正在合成证据和维度状态",
    text: "系统正在判断哪些分析维度证据充分，哪些只能给出有限结论或需要明确限制。",
  },
  {
    key: "draft",
    label: "Draft",
    start: 78,
    title: "正在生成分析草稿",
    text: "系统正在把证据包转成用户可读的金融分析，并检查是否出现未被证据支持的判断。",
  },
  {
    key: "contract",
    label: "Audit",
    start: 108,
    title: "正在执行证据契约校验",
    text: "系统正在核对数字、引用、风险提示和结论边界，准备生成最终答案和 Trace 审计记录。",
  },
  {
    key: "long",
    label: "Wait",
    start: 160,
    title: "仍在等待后端完成",
    text: "任务仍在运行，通常是 filing 检索、模型生成或契约校验耗时较长；页面会在结果返回后自动更新。",
  },
];

function escapeHtml(text) {
  return (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatElapsedTime(ms) {
  const totalSeconds = Math.max(0, Math.floor(Number(ms || 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function activeProgressIndex(elapsedSeconds) {
  let index = 0;
  for (let i = 0; i < runProgressStages.length; i += 1) {
    if (elapsedSeconds >= runProgressStages[i].start) {
      index = i;
    }
  }
  return index;
}

function progressPercentFor(elapsedSeconds, activeIndex, mode = "running") {
  if (mode === "complete") return 100;
  if (mode === "failed") return Math.max(8, Math.min(92, (activeIndex / runProgressStages.length) * 100));
  const current = runProgressStages[activeIndex] || runProgressStages[0];
  const next = runProgressStages[activeIndex + 1];
  const span = next ? Math.max(1, next.start - current.start) : 60;
  const withinStage = Math.min(1, Math.max(0, (elapsedSeconds - current.start) / span));
  return Math.min(96, ((activeIndex + withinStage) / runProgressStages.length) * 100);
}

function progressStageIndexForEvent(event) {
  const name = typeof event === "string" ? event : String((event && event.event) || "");
  const tool = typeof event === "object" && event && event.metadata ? String(event.metadata.tool || "") : "";
  if (name === "run_started" || name === "intent_resolved") return 0;
  if (name === "research_plan_started" || name === "research_plan_built" || name === "evidence_plan_built") return 1;
  if (name === "tool_started" || name === "tool_finished") {
    return tool === "search_filings" ? 3 : 2;
  }
  if (name === "evidence_evaluated") return 4;
  if (name === "draft_started" || name === "draft_validated") return 5;
  if (name === "contract_checked" || name === "relevance_checked") return 6;
  if (name === "answer_released" || name === "run_failed") return 7;
  return 0;
}

function progressBadgeForEvents(events = []) {
  const latest = events.length ? events[events.length - 1] : null;
  const name = String((latest && latest.event) || "");
  const tool = latest && latest.metadata ? String(latest.metadata.tool || "") : "";
  if (name === "run_failed") return { text: "Failed", tone: "bad" };
  if (name === "answer_released") return { text: "Complete", tone: "good" };
  if (name === "relevance_checked") return { text: "Relevance", tone: "warn" };
  if (name === "contract_checked") return { text: "Contract", tone: "warn" };
  if (name === "draft_started" || name === "draft_validated") return { text: "Drafting", tone: "warn" };
  if ((name === "tool_started" || name === "tool_finished") && tool === "search_filings") return { text: "Retrieval", tone: "warn" };
  if (name === "tool_started" || name === "tool_finished") return { text: "Data", tone: "warn" };
  if (name === "evidence_evaluated") return { text: "Synthesis", tone: "warn" };
  if (name === "research_plan_started" || name === "research_plan_built" || name === "evidence_plan_built") return { text: "Planner", tone: "warn" };
  return { text: "Running", tone: "warn" };
}

function progressPercentForEvents(events = [], elapsedMs = 0) {
  if (!events.length) {
    const activeIndex = activeProgressIndex(elapsedMs / 1000);
    return progressPercentFor(elapsedMs / 1000, activeIndex);
  }
  const latest = events[events.length - 1];
  const activeIndex = progressStageIndexForEvent(latest);
  if (latest.event === "answer_released") return 100;
  return progressPercentFor(elapsedMs / 1000, activeIndex);
}

function safeProgressEvents(trace) {
  const events = Array.isArray(trace && trace.progress_events) ? trace.progress_events : [];
  return events.filter((item) => item && item.event && item.status && item.message && item.timestamp);
}

function generateClientTraceId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  const rand = Math.random().toString(16).slice(2);
  return `client-${Date.now().toString(16)}-${rand}`;
}

function setProgressBadge(text, tone) {
  if (!progressStatusBadge) return;
  progressStatusBadge.textContent = text;
  progressStatusBadge.className = `badge ${tone}`;
}

function renderProgressStages(activeIndex = -1, mode = "idle") {
  if (!progressStageList) return;
  progressStageList.innerHTML = runProgressStages.map((stage, index) => {
    let state = "pending";
    if (mode === "complete") state = "done";
    else if (mode === "failed" && index === activeIndex) state = "failed";
    else if (index < activeIndex) state = "done";
    else if (index === activeIndex) state = "active";
    return `<li class="${state}"><span>${escapeHtml(stage.label)}</span><strong>${escapeHtml(stage.title)}</strong></li>`;
  }).join("");
}

function renderProgressLog() {
  if (!progressActivityLog) return;
  const items = progressLogItems.length
    ? progressLogItems.slice(0, 5)
    : ["系统空闲，等待新的研究任务。"];
  progressActivityLog.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function pushProgressLog(message) {
  const text = String(message || "").trim();
  if (!text) return;
  if (progressLogItems[0] === text) return;
  progressLogItems.unshift(text);
  progressLogItems = progressLogItems.slice(0, 5);
  renderProgressLog();
}

function stopProgressPolling() {
  if (progressPollTimer) {
    window.clearTimeout(progressPollTimer);
    progressPollTimer = null;
  }
  progressPollingTraceId = "";
}

function renderProgressFromEvents(events = []) {
  if (!events.length) return;
  progressUsingRealEvents = true;
  progressLastRealEvent = String(events[events.length - 1].event || "");
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  const latest = events[events.length - 1];
  const latestElapsed = Number(latest.elapsed_ms);
  const elapsedMs = Number.isFinite(latestElapsed)
    ? latestElapsed
    : (progressStartedAt ? Date.now() - progressStartedAt : 0);
  const activeIndex = progressStageIndexForEvent(latest);
  const badge = progressBadgeForEvents(events);
  const mode = latest.event === "answer_released" ? "complete" : (latest.event === "run_failed" ? "failed" : "running");
  setProgressBadge(badge.text, badge.tone);
  setProgressPanel({
    mode,
    title: String(latest.message || runProgressStages[activeIndex]?.title || "分析运行中"),
    narrative: String(latest.message || ""),
    elapsedMs,
    percent: progressPercentForEvents(events, elapsedMs),
    activeIndex,
  });
  progressLogItems = events
    .slice(-8)
    .reverse()
    .map((item) => String(item.message || "").trim())
    .filter(Boolean);
  renderProgressLog();
}

function scheduleProgressPoll() {
  if (!progressPollingTraceId) return;
  const elapsedMs = progressPollStartedAt ? Date.now() - progressPollStartedAt : 0;
  const delay = elapsedMs < 30000 ? 1000 : 2500;
  progressPollTimer = window.setTimeout(pollProgressTrace, delay);
}

async function pollProgressTrace() {
  const traceId = progressPollingTraceId;
  if (!traceId) return;
  try {
    const trace = await requestJson(makeUrl(`/trace/${encodeURIComponent(traceId)}/ui`));
    if (traceId !== progressPollingTraceId) return;
    const events = safeProgressEvents(trace);
    if (events.length) {
      renderProgressFromEvents(events);
      const latest = events[events.length - 1];
      if (latest.event === "answer_released" || latest.event === "run_failed") {
        stopProgressPolling();
        return;
      }
    }
  } catch (error) {
    if (!String(error && error.message || "").includes("404")) {
      pushProgressLog(`进度读取暂时不可用：${String(error.message || error).slice(0, 120)}`);
    }
  }
  scheduleProgressPoll();
}

function startProgressPolling(traceId) {
  stopProgressPolling();
  progressPollingTraceId = String(traceId || "");
  progressPollStartedAt = Date.now();
  if (progressPollingTraceId) {
    progressPollTimer = window.setTimeout(pollProgressTrace, 250);
  }
}

function setProgressPanel({ mode = "idle", title = "", narrative = "", elapsedMs = 0, percent = 0, activeIndex = -1 } = {}) {
  if (progressCurrentTitle) progressCurrentTitle.textContent = title || "等待研究指令";
  if (progressNarrative) progressNarrative.textContent = narrative || "提交问题后，这里会显示当前分析正在推进到哪一步。";
  if (progressElapsed) progressElapsed.textContent = formatElapsedTime(elapsedMs);
  if (progressMeterFill) progressMeterFill.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
  renderProgressStages(activeIndex, mode);
}

function tickRunProgress() {
  if (!progressStartedAt) return;
  const elapsedMs = Date.now() - progressStartedAt;
  const elapsedSeconds = elapsedMs / 1000;
  const activeIndex = activeProgressIndex(elapsedSeconds);
  const stage = runProgressStages[activeIndex] || runProgressStages[0];
  const percent = progressPercentFor(elapsedSeconds, activeIndex);
  const longSuffix = elapsedSeconds > 180
    ? ` 已运行 ${formatElapsedTime(elapsedMs)}，后端还在处理。`
    : "";
  setProgressPanel({
    mode: "running",
    title: stage.title,
    narrative: `${stage.text}${longSuffix}`,
    elapsedMs,
    percent,
    activeIndex,
  });
  pushProgressLog(`${stage.title}：${stage.text}`);
}

function startRunProgress(query, traceId = "") {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  stopProgressPolling();
  progressStartedAt = Date.now();
  progressLogItems = [];
  progressUsingRealEvents = false;
  progressLastRealEvent = "";
  setProgressBadge("Running", "warn");
  const queryLabel = String(query || "").trim().slice(0, 42);
  pushProgressLog(queryLabel ? `收到研究指令：“${queryLabel}${queryLabel.length >= 42 ? "..." : ""}”` : "收到研究指令。");
  tickRunProgress();
  progressTimer = window.setInterval(tickRunProgress, 1200);
  if (traceId) {
    startProgressPolling(traceId);
  }
}

function finishRunProgress(data) {
  const elapsedMs = progressStartedAt ? Date.now() - progressStartedAt : 0;
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  stopProgressPolling();
  if (progressUsingRealEvents && progressLastRealEvent === "answer_released") {
    progressStartedAt = 0;
    return;
  }
  progressStartedAt = 0;
  const task = data && data.task_type ? `任务类型 ${data.task_type}` : "任务类型已识别";
  const tools = data && Array.isArray(data.used_tools) && data.used_tools.length
    ? `使用工具：${data.used_tools.join(", ")}。`
    : "未返回工具列表。";
  const trace = data && data.trace_id ? `Trace ${data.trace_id} 已生成。` : "Trace 正在等待生成。";
  setProgressBadge("Complete", "good");
  setProgressPanel({
    mode: "complete",
    title: "分析完成，正在呈现结果",
    narrative: `${task}，${tools} ${trace}`,
    elapsedMs,
    percent: 100,
    activeIndex: runProgressStages.length - 1,
  });
  pushProgressLog(`分析完成：${task}，总耗时 ${formatElapsedTime(elapsedMs)}。`);
}

function failRunProgress(error) {
  const elapsedMs = progressStartedAt ? Date.now() - progressStartedAt : 0;
  const activeIndex = activeProgressIndex(elapsedMs / 1000);
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  stopProgressPolling();
  if (progressUsingRealEvents && progressLastRealEvent === "run_failed") {
    progressStartedAt = 0;
    return;
  }
  progressStartedAt = 0;
  const message = String((error && error.message) || error || "请求失败。").slice(0, 180);
  setProgressBadge("Failed", "bad");
  setProgressPanel({
    mode: "failed",
    title: "分析未完成",
    narrative: `请求在当前阶段失败：${message}`,
    elapsedMs,
    percent: progressPercentFor(elapsedMs / 1000, activeIndex, "failed"),
    activeIndex,
  });
  pushProgressLog(`分析失败：${message}`);
}

function simpleMarkdownToHtml(rawText) {
  const safe = escapeHtml(rawText || "");
  const lines = safe.split(/\r?\n/);
  const out = [];
  let inUl = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inUl) {
        out.push("</ul>");
        inUl = false;
      }
      out.push("<br/>");
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      if (!inUl) {
        out.push("<ul>");
        inUl = true;
      }
      const item = trimmed.replace(/^[-*]\s+/, "");
      out.push(`<li>${item}</li>`);
      continue;
    }

    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }

    if (/^\|.*\|$/.test(trimmed)) {
      const cells = trimmed
        .split("|")
        .map((c) => c.trim())
        .filter(Boolean);
      out.push(`<p>${cells.join(" | ")}</p>`);
      continue;
    }

    out.push(`<p>${trimmed}</p>`);
  }

  if (inUl) {
    out.push("</ul>");
  }
  return out.join("\n");
}

function renderAnswer(rawText) {
  const text = rawText || "(empty answer)";
  answerText.classList.remove("empty");

  if (window.marked && window.DOMPurify) {
    const parsed = window.marked.parse(text, {
      gfm: true,
      breaks: true,
    });
    answerText.innerHTML = window.DOMPurify.sanitize(parsed, {
      USE_PROFILES: { html: true },
    });
    return;
  }

  answerText.innerHTML = simpleMarkdownToHtml(text);
}

function formatValue(value, unit) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const n = Number(value);
  if (Number.isFinite(n)) {
    const abs = Math.abs(n);
    if (abs >= 1_000_000_000) {
      return `${(n / 1_000_000_000).toFixed(2)}B${unit ? ` ${unit}` : ""}`;
    }
    if (abs >= 1_000_000) {
      return `${(n / 1_000_000).toFixed(2)}M${unit ? ` ${unit}` : ""}`;
    }
    return `${n.toFixed(4).replace(/\.?0+$/, "")}${unit ? ` ${unit}` : ""}`;
  }
  return `${String(value)}${unit ? ` ${unit}` : ""}`;
}

function renderList(container, items, formatter, emptyText) {
  container.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = emptyText;
    container.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = formatter(item);
    container.appendChild(li);
  });
}

function renderHtmlList(container, items, formatter, emptyText) {
  container.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = emptyText;
    container.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = formatter(item);
    container.appendChild(li);
  });
}

function confidenceTone(value) {
  const normalized = String(value || "").toLowerCase();
  if (["high", "passed", "satisfied"].includes(normalized)) return "good";
  if (["medium", "partial", "warning"].includes(normalized)) return "warn";
  if (["low", "missing", "failed"].includes(normalized)) return "bad";
  return "neutral";
}

function evidencePill(label, tone = "neutral") {
  return `<span class="badge ${tone}">${escapeHtml(label || "-")}</span>`;
}

function formatNumericEvidenceHtml(item) {
  const row = item && typeof item === "object" ? item : {};
  const provider = row.source_provider || "-";
  const confidence = row.confidence || "-";
  const warning = row.reconciliation_warning ? `<span class="evidence-warning">${escapeHtml(row.reconciliation_warning)}</span>` : "";
  return `
    <span class="evidence-line">
      <span class="evidence-main">
        <strong>${escapeHtml(row.ticker || "-")}</strong>
        <span>${escapeHtml(row.metric || "-")}</span>
        <span class="metric-value">${escapeHtml(formatValue(row.value, row.unit || ""))}</span>
      </span>
      <span class="evidence-meta">
        ${evidencePill(row.period_end || "-", "neutral")}
        ${evidencePill(provider, "neutral")}
        ${evidencePill(confidence, confidenceTone(confidence))}
        ${warning}
      </span>
    </span>
  `;
}

function formatLimitationHtml(item) {
  const severity = sanitizeTraceDisplayText((item && item.severity) || "info");
  const message = sanitizeTraceDisplayText((item && item.message) || "Evidence is limited.");
  return `<span class="evidence-line"><span class="evidence-main">${evidencePill(severity, confidenceTone(severity))}<span>${escapeHtml(message)}</span></span></span>`;
}

function formatLimitationDisplay(item) {
  const severity = sanitizeTraceDisplayText((item && item.severity) || "info");
  const message = sanitizeTraceDisplayText((item && item.message) || "Evidence is limited.");
  return `[${severity}] ${message}`;
}

function renderTableHtml(table) {
  if (!table || typeof table !== "object") {
    return '<p class="empty">No table data.</p>';
  }
  const columns = Array.isArray(table.columns) ? table.columns : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  if (columns.length === 0 || rows.length === 0) {
    return '<p class="empty">No table data.</p>';
  }
  const head = `<tr>${columns.map((c) => `<th>${escapeHtml(String(c))}</th>`).join("")}</tr>`;
  const body = rows
    .map((row) => {
      const cells = columns.map((col) => {
        const v = row && typeof row === "object" ? row[col] : "";
        return `<td>${escapeHtml(v === null || v === undefined ? "-" : String(v))}</td>`;
      });
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function cleanUserVisibleText(value) {
  const raw = String(value === null || value === undefined ? "" : value);
  if (/yfinance/i.test(raw) && /fallback/i.test(raw)) {
    return "部分结构化财务数据来自 yfinance，可信度为 medium。";
  }
  return sanitizeTraceDisplayText(raw)
    .replace(/Required evidence is missing/gi, "current evidence is limited")
    .replace(/\brequired_evidence_missing\b/g, "current evidence is limited")
    .replace(/\bdependency_metric_id\b/g, "dependency detail")
    .replace(/\bfallback\b/gi, "alternate data source");
}

function shouldRenderTraceResponse(trace, requestedTraceId, expectedTraceId = "", currentAnswerTraceId = "") {
  const requested = String(requestedTraceId || "").trim();
  const expected = String(expectedTraceId || "").trim();
  const current = String(currentAnswerTraceId || "").trim();
  const actual = String((trace && trace.trace_id) || "").trim();
  if (actual && requested && actual !== requested) {
    return { ok: false, reason: `Trace mismatch: requested ${requested}, received ${actual}.` };
  }
  if (expected && requested && requested !== expected) {
    return { ok: false, reason: `Trace request is stale: expected ${expected}, requested ${requested}.` };
  }
  if (expected && current && expected !== current) {
    return { ok: false, reason: `Trace response is stale: current answer trace is ${current}.` };
  }
  if (expected && actual && actual !== expected) {
    return { ok: false, reason: `Trace mismatch: expected ${expected}, received ${actual}.` };
  }
  return { ok: true, reason: "" };
}

function refsInline(refs) {
  if (!Array.isArray(refs)) return "";
  return refs
    .map((ref) => String(ref || "").trim())
    .filter(Boolean)
    .map((ref) => `[${escapeHtml(ref)}]`)
    .join("");
}

function textWithRefs(text, refs) {
  const clean = cleanUserVisibleText(text || "");
  if (!clean) return "";
  const cleanRefs = Array.isArray(refs) ? refs.map((ref) => String(ref || "").trim()).filter(Boolean) : [];
  if (cleanRefs.some((ref) => clean.includes(`[${ref}]`))) {
    return escapeHtml(clean);
  }
  return `${escapeHtml(clean)} ${refsInline(cleanRefs)}`.trim();
}

const methodologyDimensionOrder = [
  "revenue_quality",
  "profitability_quality",
  "moat_and_competitive_risk",
  "valuation_and_risk_boundary",
];

function limitationKey(text) {
  const normalized = String(text || "").toLowerCase();
  if (normalized.includes("估值证据") || normalized.includes("valuation evidence")) return "valuation_missing";
  if (normalized.includes("yfinance")) return "yfinance_provider";
  if (normalized.includes("投资建议") || normalized.includes("investment advice")) return "investment_boundary";
  if (normalized.includes("净利率") || normalized.includes("毛利率") || normalized.includes("营业利润率") || normalized.includes("margin")) return "profitability_scope";
  return normalized
    .replace(/\[[A-Z]\d+\]/g, "")
    .replace(/\bREQ-[A-Za-z0-9_-]+\b/g, "")
    .replace(/\bdependency_[A-Za-z0-9_]+\b/g, "")
    .replace(/\bnumeric_only_[A-Za-z0-9_]+\b/g, "")
    .replace(/[\s。；;，,：:、.-]+/g, "");
}

function dedupeUserLimitations(items, excludeKeys = new Set(), limit = 5) {
  const out = [];
  const seen = new Set(excludeKeys);
  for (const item of Array.isArray(items) ? items : []) {
    const clean = cleanUserVisibleText(item);
    if (!clean) continue;
    const key = limitationKey(clean);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(clean);
    if (out.length >= limit) break;
  }
  return out;
}

function renderMetricTable(metricTable) {
  const rows = Array.isArray(metricTable) ? metricTable.filter((row) => row && typeof row === "object") : [];
  if (rows.length === 0) return "";
  const companies = [];
  for (const row of rows) {
    const values = row.company_values && typeof row.company_values === "object" ? row.company_values : {};
    for (const company of Object.keys(values)) {
      if (company && !companies.includes(company)) companies.push(company);
    }
  }
  if (companies.length < 2) return "";
  const shownCompanies = companies.slice(0, 2);
  const head = `<tr><th>指标</th>${shownCompanies.map((company) => `<th>${escapeHtml(company)}</th>`).join("")}<th>当前判断</th></tr>`;
  const body = rows.map((row) => {
    const values = row.company_values && typeof row.company_values === "object" ? row.company_values : {};
    const cells = shownCompanies.map((company) => `<td>${escapeHtml(cleanUserVisibleText(values[company] || "缺少可验证数据"))}</td>`).join("");
    return `<tr><td>${escapeHtml(cleanUserVisibleText(row.label || row.metric_id || "-"))}</td>${cells}<td>${escapeHtml(cleanUserVisibleText(row.judgment || ""))}</td></tr>`;
  }).join("");
  return `<h4>核心指标对比</h4><table class="methodology-metric-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function renderSingleCompanyMetricTable(metricTable) {
  const rows = Array.isArray(metricTable) ? metricTable.filter((row) => row && typeof row === "object") : [];
  if (rows.length === 0) return "";
  const body = rows.map((row) => {
    const refs = refsInline(row.evidence_refs || []);
    return `<tr><td>${escapeHtml(cleanUserVisibleText(row.label || row.metric_id || "-"))}</td><td>${escapeHtml(cleanUserVisibleText(row.value || "缺少可验证数据"))} ${refs}</td><td>${escapeHtml(cleanUserVisibleText(row.interpretation || ""))}</td></tr>`;
  }).join("");
  return '<h4>核心指标</h4><table class="methodology-metric-table"><thead><tr><th>指标</th><th>数值</th><th>当前解读</th></tr></thead><tbody>' + body + "</tbody></table>";
}

function renderDraftRefs(refs) {
  return refsInline(Array.isArray(refs) ? refs : []);
}

function renderDraftItems(title, items, limit = 6) {
  const rows = Array.isArray(items) ? items.filter((item) => item && typeof item === "object") : [];
  if (rows.length === 0) return "";
  const body = rows.slice(0, limit).map((item) => {
    const text = cleanUserVisibleText(item.statement || item.claim || "");
    const refs = renderDraftRefs(item.citation_refs || item.evidence_refs || []);
    return text ? `<li>${escapeHtml(text)} ${refs}</li>` : "";
  }).join("");
  return body ? `<h4>${escapeHtml(title)}</h4><ul>${body}</ul>` : "";
}

function renderDraftStrings(title, items, limit = 5) {
  const rows = Array.isArray(items) ? items.map(cleanUserVisibleText).filter(Boolean) : [];
  if (rows.length === 0) return "";
  return `<h4>${escapeHtml(title)}</h4><ul>${Array.from(new Set(rows)).slice(0, limit).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderAnalystDraft(output, view) {
  const safeView = view && typeof view === "object" ? view : {};
  const synthesis = output && output.synthesis && typeof output.synthesis === "object" ? output.synthesis : {};
  const draft = safeView.accepted_draft && typeof safeView.accepted_draft === "object"
    ? safeView.accepted_draft
    : (synthesis.accepted_draft && typeof synthesis.accepted_draft === "object" ? synthesis.accepted_draft : {});
  const conclusion = draft.tentative_conclusion && typeof draft.tentative_conclusion === "object"
    ? draft.tentative_conclusion
    : {};
  const conclusionText = cleanUserVisibleText(conclusion.statement || draft.overall_judgment || safeView.short_answer || output.summary || "");
  const conclusionRefs = renderDraftRefs(conclusion.citation_refs || draft.citation_refs || []);
  const dimensions = Array.isArray(draft.dimension_analyses) ? draft.dimension_analyses : [];
  const singleCompanyMetricTable = Array.isArray(safeView.single_company_metric_table)
    ? safeView.single_company_metric_table
    : [];
  const comparisonMetricTable = Array.isArray(safeView.metric_table) ? safeView.metric_table : [];
  const draftLimits = Array.isArray(draft.methodology_limitations) && draft.methodology_limitations.length
    ? draft.methodology_limitations
    : (Array.isArray(safeView.limitations) ? safeView.limitations : []);

  let html = "<h3>Analyst Draft / 模型研判</h3>";
  html += `<h4>核心判断</h4><p>${escapeHtml(conclusionText || "模型分析稿已通过校验，但未返回核心判断。")} ${conclusionRefs}</p>`;
  html += singleCompanyMetricTable.length
    ? renderSingleCompanyMetricTable(singleCompanyMetricTable)
    : renderMetricTable(comparisonMetricTable);
  if (dimensions.length > 0) {
    html += '<h4>维度研判</h4><ol class="methodology-dimensions">';
    html += dimensions.map((item) => {
      const title = cleanUserVisibleText(item.dimension_id || "");
      const claim = cleanUserVisibleText(item.claim || "");
      const refs = renderDraftRefs(item.evidence_refs || []);
      return claim ? `<li><strong>${escapeHtml(title)}：</strong>${escapeHtml(claim)} ${refs}</li>` : "";
    }).join("");
    html += "</ol>";
  }
  html += renderDraftItems("证据依据", draft.decision_basis || [], 6);
  html += renderDraftItems("补充支撑", draft.supporting_points || [], 4);
  html += renderDraftItems("反方观点", draft.counterpoints || [], 4);
  html += renderDraftItems("风险与取舍", draft.risk_tradeoffs || [], 5);
  html += renderDraftItems("不确定性", draft.uncertainty_notes || [], 5);
  html += renderDraftStrings("限制", draftLimits, 5);
  html += renderDraftStrings("后续跟踪", draft.follow_up_metrics || [], 5);
  html += renderDraftItems("合规边界", draft.safety_notes || [], 3);
  return html;
}

function renderMethodologyComparison(output, view) {
  const safeOutput = output && typeof output === "object" ? output : {};
  const safeView = view && typeof view === "object" ? view : {};
  const answer = safeView.methodology_answer && typeof safeView.methodology_answer === "object"
    ? safeView.methodology_answer
    : ((safeOutput.synthesis && safeOutput.synthesis.methodology_answer && typeof safeOutput.synthesis.methodology_answer === "object")
      ? safeOutput.synthesis.methodology_answer
      : {});
  const sections = Array.isArray(answer.dimension_sections)
    ? answer.dimension_sections
    : (Array.isArray(safeView.dimension_sections) ? safeView.dimension_sections : []);
  const sectionsByDimension = Object.fromEntries(
    sections
      .filter((item) => item && item.dimension_id)
      .map((item) => [String(item.dimension_id), item]),
  );
  const dimensionItems = [];
  const dimensionLimitationKeys = new Set();
  for (const dimensionId of methodologyDimensionOrder) {
    const item = sectionsByDimension[dimensionId];
    if (!item) continue;
    const title = item.title || item.dimension_id || "";
    const status = String(item.status || "");
    let summary = "";
    if (status === "missing") {
      summary = item.limitation || "";
      if (!summary && dimensionId === "valuation_and_risk_boundary") {
        summary = "当前缺少估值证据，因此不能判断谁更便宜或更值得买。";
      }
      if (!summary) summary = "当前缺少该维度证据。";
      dimensionLimitationKeys.add(limitationKey(summary));
    } else {
      summary = textWithRefs(item.summary || "", item.evidence_refs || []);
      if (!summary) {
        summary = "当前缺少该维度证据。";
        dimensionLimitationKeys.add(limitationKey(summary));
      }
    }
    dimensionItems.push({ title, summary, html: status !== "missing" });
  }
  for (const item of sections) {
    if (!item || methodologyDimensionOrder.includes(String(item.dimension_id || ""))) continue;
    const title = item.title || item.dimension_id || "";
    let summary = String(item.status || "") === "missing"
      ? cleanUserVisibleText(item.limitation || "当前缺少该维度证据。")
      : textWithRefs(item.summary || "", item.evidence_refs || []);
    if (!summary) summary = "当前缺少该维度证据。";
    dimensionItems.push({ title, summary, html: String(item.status || "") !== "missing" });
    if (String(item.status || "") === "missing") {
      dimensionLimitationKeys.add(limitationKey(summary));
    }
  }
  const uniqueLimits = dedupeUserLimitations(answer.limitations || safeView.methodology_limitations || [], dimensionLimitationKeys, 5);
  if (uniqueLimits.length === 0) {
    uniqueLimits.push("以下内容仅是基于已验证证据的基本面比较，不构成投资建议。");
  }
  const judgment = cleanUserVisibleText(answer.judgment || safeView.short_answer || safeOutput.summary || "");
  const counterpoint = cleanUserVisibleText(answer.counterpoint || "");

  let html = "<h3>基本面方法论比较</h3>";
  html += `<h4>比较判断</h4><p>${escapeHtml(judgment || "当前证据不足以形成完整比较判断。")}</p>`;
  html += renderMetricTable(answer.metric_table || safeView.metric_table || []);
  if (dimensionItems.length > 0) {
    html += '<h4>维度分析</h4><ol class="methodology-dimensions">';
    html += dimensionItems.map((item) => {
      const body = item.html ? item.summary : escapeHtml(cleanUserVisibleText(item.summary));
      return `<li><strong>${escapeHtml(item.title)}：</strong>${body}</li>`;
    }).join("");
    html += "</ol>";
  }
  if (counterpoint) {
    html += `<h4>反方观点</h4><p>${escapeHtml(counterpoint)}</p>`;
  }
  if (uniqueLimits.length > 0) {
    html += `<h4>限制</h4><ul>${uniqueLimits.slice(0, 6).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  return html;
}

function renderMethodologySingleCompany(output, view) {
  const safeOutput = output && typeof output === "object" ? output : {};
  const safeView = view && typeof view === "object" ? view : {};
  const answer = safeView.methodology_answer && typeof safeView.methodology_answer === "object"
    ? safeView.methodology_answer
    : ((safeOutput.synthesis && safeOutput.synthesis.methodology_answer && typeof safeOutput.synthesis.methodology_answer === "object")
      ? safeOutput.synthesis.methodology_answer
      : {});
  const sections = Array.isArray(answer.dimension_sections)
    ? answer.dimension_sections
    : (Array.isArray(safeView.dimension_sections) ? safeView.dimension_sections : []);
  const dimensionOrder = [
    "business_model",
    "revenue_quality",
    "profitability_quality",
    "cash_flow_quality",
    "balance_sheet_and_capital_intensity",
    "moat_and_competitive_risk",
    "valuation_and_risk_boundary",
  ];
  const sectionsByDimension = Object.fromEntries(
    sections
      .filter((item) => item && item.dimension_id)
      .map((item) => [String(item.dimension_id), item]),
  );
  const dimensionItems = [];
  const dimensionLimitationKeys = new Set();
  for (const dimensionId of dimensionOrder) {
    const item = sectionsByDimension[dimensionId];
    if (!item) continue;
    const title = item.title || item.dimension_id || "";
    const status = String(item.status || "");
    let summary = "";
    if (status === "missing") {
      summary = item.limitation || "";
      if (!summary && dimensionId === "moat_and_competitive_risk") summary = "当前缺少风险文本证据，不能做具体风险判断。";
      if (!summary && dimensionId === "valuation_and_risk_boundary") summary = "当前缺少估值证据，因此不能判断估值吸引力。";
      if (!summary && dimensionId === "cash_flow_quality") summary = "当前缺少经营现金流、自由现金流或资本开支证据，无法验证利润能否转化为现金。";
      if (!summary && dimensionId === "balance_sheet_and_capital_intensity") summary = "当前缺少现金、债务、资本开支、应收款或存货证据，不能判断抗风险能力和资本投入强度。";
      if (!summary) summary = "当前缺少该维度证据。";
      dimensionLimitationKeys.add(limitationKey(summary));
    } else {
      summary = textWithRefs(item.summary || "", item.evidence_refs || []);
      if (!summary) {
        summary = "当前缺少该维度证据。";
        dimensionLimitationKeys.add(limitationKey(summary));
      }
    }
    dimensionItems.push({ title, summary, html: status !== "missing" });
  }
  const uniqueLimits = dedupeUserLimitations(answer.limitations || safeView.methodology_limitations || [], dimensionLimitationKeys, 5);
  const followUps = Array.isArray(answer.follow_up_metrics)
    ? answer.follow_up_metrics.map(cleanUserVisibleText).filter(Boolean)
    : (Array.isArray(safeView.follow_up_metrics) ? safeView.follow_up_metrics.map(cleanUserVisibleText).filter(Boolean) : []);
  const judgment = cleanUserVisibleText(answer.judgment || safeView.short_answer || safeOutput.summary || "");

  let html = "<h3>基本面快速分析</h3>";
  html += `<h4>初步判断</h4><p>${escapeHtml(judgment || "当前证据不足以形成完整基本面判断。")}</p>`;
  html += renderSingleCompanyMetricTable(answer.single_company_metric_table || safeView.single_company_metric_table || []);
  if (dimensionItems.length > 0) {
    html += '<h4>维度分析</h4><ol class="methodology-dimensions">';
    html += dimensionItems.map((item) => {
      const body = item.html ? item.summary : escapeHtml(cleanUserVisibleText(item.summary));
      return `<li><strong>${escapeHtml(item.title)}：</strong>${body}</li>`;
    }).join("");
    html += "</ol>";
  }
  if (uniqueLimits.length > 0) {
    html += `<h4>限制</h4><ul>${uniqueLimits.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  if (followUps.length > 0) {
    html += `<h4>后续应关注指标</h4><ul>${Array.from(new Set(followUps)).slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  return html;
}

function renderRiskFocusedAnalysis(output, view) {
  const safeOutput = output && typeof output === "object" ? output : {};
  const safeView = view && typeof view === "object" ? view : {};
  const answer = safeView.risk_focused_answer && typeof safeView.risk_focused_answer === "object"
    ? safeView.risk_focused_answer
    : ((safeOutput.synthesis && safeOutput.synthesis.risk_focused_answer && typeof safeOutput.synthesis.risk_focused_answer === "object")
      ? safeOutput.synthesis.risk_focused_answer
      : {});
  const direct = cleanUserVisibleText(answer.direct_judgment || safeView.short_answer || safeOutput.summary || "");
  const why = Array.isArray(answer.why_core_issue) ? answer.why_core_issue.map(cleanUserVisibleText).filter(Boolean) : [];
  const filingEvidence = Array.isArray(answer.filing_evidence) ? answer.filing_evidence : [];
  const financialContext = Array.isArray(answer.financial_context)
    ? answer.financial_context.map(cleanUserVisibleText).filter(Boolean)
    : (Array.isArray(safeView.financial_context) ? safeView.financial_context.map(cleanUserVisibleText).filter(Boolean) : []);
  const secondary = Array.isArray(answer.secondary_risks) ? answer.secondary_risks : [];
  const boundaries = dedupeUserLimitations(answer.evidence_boundaries || safeView.evidence_boundaries || [], new Set(), 5);

  function themeLine(theme) {
    const item = theme && typeof theme === "object" ? theme : {};
    const name = cleanUserVisibleText(item.theme_name || "风险主题");
    const whyText = cleanUserVisibleText(item.why_it_matters || "该主题来自已验证风险披露。");
    return `${escapeHtml(name)}：${escapeHtml(whyText)} ${refsInline(item.evidence_refs || [])}`.trim();
  }

  let html = "<h3>风险专题分析</h3>";
  html += `<h4>风险判断</h4><p>${escapeHtml(direct || "当前缺少足够风险文本证据，不能可靠判断最大问题。")}</p>`;
  if (why.length > 0) {
    html += `<h4>为什么这是核心问题</h4><ol>${why.slice(0, 4).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>`;
  }
  if (filingEvidence.length > 0) {
    html += "<h4>财报证据</h4><ul>";
    html += filingEvidence.slice(0, 4).map((item) => `<li>${themeLine(item)}</li>`).join("");
    html += "</ul>";
  }
  if (financialContext.length > 0) {
    html += `<h4>相关财务背景</h4><ul>${financialContext.slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  if (secondary.length > 0) {
    html += `<h4>其他需要关注的风险</h4><ul>${secondary.slice(0, 3).map((item) => `<li>${themeLine(item)}</li>`).join("")}</ul>`;
  }
  if (boundaries.length > 0) {
    html += `<h4>证据边界</h4><ul>${boundaries.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  return html;
}

function contractStatusFromChatData(data) {
  const safe = data && typeof data === "object" ? data : {};
  const output = safe.output && typeof safe.output === "object" ? safe.output : {};
  const contract = output.contract && typeof output.contract === "object" ? output.contract : {};
  return String(safe.contract_status || contract.status || "").toLowerCase();
}

function isBlockedContractResponse(data) {
  const safe = data && typeof data === "object" ? data : {};
  const output = safe.output && typeof safe.output === "object" ? safe.output : {};
  const contract = output.contract && typeof output.contract === "object" ? output.contract : {};
  return String(safe.contract_status || "").toLowerCase() === "blocked"
    || String(contract.status || "").toLowerCase() === "blocked";
}

function limitationText(item) {
  if (item && typeof item === "object") {
    return item.message || item.code || "";
  }
  return String(item || "");
}

function buildBlockedPrimaryHtml(data) {
  const safe = data && typeof data === "object" ? data : {};
  const output = safe.output && typeof safe.output === "object" ? safe.output : {};
  const contract = output.contract && typeof output.contract === "object" ? output.contract : {};
  const answer = cleanUserVisibleText(safe.answer || "");
  const summary = cleanUserVisibleText(contract.public_summary || "");
  const limitations = [
    ...(Array.isArray(safe.limitations) ? safe.limitations : []),
    ...(Array.isArray(output.limitations) ? output.limitations : []),
  ]
    .map(limitationText)
    .map(cleanUserVisibleText)
    .filter(Boolean);
  const uniqueLimitations = Array.from(new Set(limitations)).slice(0, 5);

  let html = "<h3>证据不足，未发布分析</h3>";
  html += simpleMarkdownToHtml(answer || "目前证据不足以支持一个完整且通过契约校验的结论。");
  if (summary && summary !== answer) {
    html += `<h4>契约状态</h4><p>${escapeHtml(summary)}</p>`;
  }
  if (uniqueLimitations.length > 0) {
    html += `<h4>限制</h4><ul>${uniqueLimitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
  return html;
}

function renderBlockedPrimaryResponse(data) {
  outputView.classList.remove("empty");
  outputView.classList.add("blocked");
  outputView.innerHTML = buildBlockedPrimaryHtml(data);
  return true;
}

function renderOutput(output) {
  const hasRenderableFallback = output && typeof output === "object" && (
    output.summary
    || output.final_answer
    || (output.report && output.report.markdown)
    || (Array.isArray(output.key_points) && output.key_points.length > 0)
  );
  if (!output || typeof output !== "object" || (!output.task_type && !hasRenderableFallback)) {
    outputView.classList.add("empty");
    outputView.innerHTML = "No structured output returned.";
    return false;
  }

  const title = output.title || "Structured Output";
  const summary = output.summary || output.final_answer || (output.report && output.report.markdown) || "";
  const keyPoints = Array.isArray(output.key_points) ? output.key_points : [];
  const comparisonBasis = output.comparison_basis || "";
  const view = output.view && typeof output.view === "object" ? output.view : {};
  const kind = view.kind || output.task_type || "generic_summary";
  const marketReaction = output.market_reaction && typeof output.market_reaction === "object"
    ? output.market_reaction
    : null;

  let html = "";
  if (kind === "methodology_comparison_brief") {
    html = renderMethodologyComparison(output, view);
  } else if (kind === "methodology_single_company_brief") {
    html = renderMethodologySingleCompany(output, view);
  } else if (kind === "analyst_draft_brief") {
    html = renderAnalystDraft(output, view);
  } else if (kind === "risk_focused_analysis_brief") {
    html = renderRiskFocusedAnalysis(output, view);
  } else {
    html = `<h3>${escapeHtml(title)}</h3>`;
    html += `<p>${escapeHtml(summary)}</p>`;
    if (comparisonBasis) {
      html += `<p class="basis-line"><strong>Comparison Basis:</strong> ${escapeHtml(comparisonBasis)}</p>`;
    }
    if (keyPoints.length > 0) {
      html += `<h4>Key Points</h4><ul>${keyPoints.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
  }

  if (kind !== "methodology_comparison_brief" && kind === "fact_qa") {
    const headline = view.headline_metric && typeof view.headline_metric === "object" ? view.headline_metric : null;
    if (headline && Object.keys(headline).length > 0) {
      html += "<h4>Headline Metric</h4>";
      html += `<p><strong>${escapeHtml(headline.ticker || "-")} ${escapeHtml(headline.metric || "-")}</strong>: ${escapeHtml(formatValue(headline.value, headline.unit || ""))} (${escapeHtml(headline.period_end || "-")})</p>`;
    }
    if (view.period_note) {
      html += `<p><strong>Period:</strong> ${escapeHtml(view.period_note)}</p>`;
    }
    if (Array.isArray(view.supporting_points) && view.supporting_points.length > 0) {
      html += `<h4>Supporting Points</h4><ul>${view.supporting_points.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
  } else if (kind !== "methodology_comparison_brief" && kind === "trend_analysis") {
    if (view.trend_conclusion) {
      html += `<h4>Trend Conclusion</h4><p>${escapeHtml(view.trend_conclusion)}</p>`;
    }
    if (Array.isArray(view.change_points) && view.change_points.length > 0) {
      html += `<h4>Change Points</h4><ul>${view.change_points.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
    html += `<h4>Trend Table</h4>${renderTableHtml(view.trend_table)}`;
  } else if (kind !== "methodology_comparison_brief" && kind === "company_comparison") {
    if (view.comparison_basis_line) {
      html += `<h4>Comparison Basis</h4><p>${escapeHtml(view.comparison_basis_line)}</p>`;
    }
    html += `<h4>Comparison Table</h4>${renderTableHtml(view.comparison_table)}`;
    if (view.delta_summary) {
      html += `<h4>Delta Summary</h4><p>${escapeHtml(view.delta_summary)}</p>`;
    }
  } else if (kind !== "methodology_comparison_brief" && kind === "report_summary") {
    if (view.executive_summary) {
      html += `<h4>Executive Summary</h4><p>${escapeHtml(view.executive_summary)}</p>`;
    }
    if (Array.isArray(view.key_data_points) && view.key_data_points.length > 0) {
      html += `<h4>Key Data Points</h4><ul>${view.key_data_points.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
    if (Array.isArray(view.text_findings) && view.text_findings.length > 0) {
      html += `<h4>Text Findings</h4><ul>${view.text_findings.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
    if (view.risk_and_limits) {
      html += `<h4>Risk and Limits</h4><p>${escapeHtml(view.risk_and_limits)}</p>`;
    }
  }

  if (marketReaction) {
    const events = Array.isArray(marketReaction.events) ? marketReaction.events : [];
    const highlights = Array.isArray(marketReaction.highlights) ? marketReaction.highlights : [];
    const limits = Array.isArray(marketReaction.limitations) ? marketReaction.limitations : [];
    html += `<h4>${escapeHtml(marketReaction.title || "Market Reaction")}</h4>`;
    if (marketReaction.anchor_rule) {
      html += `<p><strong>Anchor:</strong> ${escapeHtml(marketReaction.anchor_rule)}</p>`;
    }
    if (highlights.length > 0) {
      html += `<ul>${highlights.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
    if (events.length > 0) {
      const table = {
        columns: ["ticker", "event_date", "form_type", "fiscal_period", "return_1d", "return_5d", "return_10d", "coverage_flag"],
        rows: events.map((e) => ({
          ticker: e.ticker || "-",
          event_date: e.event_date || "-",
          form_type: e.form_type || "-",
          fiscal_period: e.fiscal_period || "-",
          return_1d: e.return_1d,
          return_5d: e.return_5d,
          return_10d: e.return_10d,
          coverage_flag: e.coverage_flag || "-",
        })),
      };
      html += renderTableHtml(table);
    }
    if (limits.length > 0) {
      html += `<p><strong>Limitations:</strong> ${escapeHtml(limits.join("; "))}</p>`;
    }
  }

  outputView.classList.remove("empty");
  outputView.innerHTML = html;
  return true;
}

function isMethodologyPrimaryOutput(output) {
  const safeOutput = output && typeof output === "object" ? output : {};
  const view = safeOutput.view && typeof safeOutput.view === "object" ? safeOutput.view : {};
  return view.kind === "methodology_comparison_brief"
    || view.kind === "methodology_single_company_brief"
    || view.kind === "analyst_draft_brief"
    || view.kind === "risk_focused_analysis_brief"
    || !!(view.methodology_answer && typeof view.methodology_answer === "object" && Object.keys(view.methodology_answer).length > 0)
    || !!(view.risk_focused_answer && typeof view.risk_focused_answer === "object" && Object.keys(view.risk_focused_answer).length > 0)
    || !!(safeOutput.synthesis && safeOutput.synthesis.methodology_answer && typeof safeOutput.synthesis.methodology_answer === "object")
    || !!(safeOutput.synthesis && safeOutput.synthesis.risk_focused_answer && typeof safeOutput.synthesis.risk_focused_answer === "object");
}

function setLegacyDebugMode(isDebug) {
  if (!hasDocument) return;
  const legacy = document.querySelector(".legacy-answer");
  const advanced = legacy ? legacy.closest(".advanced-debug") : null;
  if (legacy) {
    legacy.open = false;
    legacy.classList.toggle("debug-muted", !!isDebug);
  }
  if (advanced) {
    advanced.open = false;
    advanced.classList.toggle("debug-muted", !!isDebug);
  }
  document.querySelectorAll(".legacy-citations").forEach((node) => {
    node.open = false;
    const wrapper = node.closest(".advanced-debug");
    if (wrapper) {
      wrapper.open = false;
      wrapper.classList.toggle("debug-muted", !!isDebug);
    }
  });
}

function renderEvidenceFromOutput(output) {
  const numeric = output && Array.isArray(output.numeric_evidence) ? output.numeric_evidence : [];
  const text = output && Array.isArray(output.text_evidence) ? output.text_evidence : [];
  const limits = output && Array.isArray(output.limitations) ? output.limitations : [];
  const dedupedLimits = [];
  const seenLimitKeys = new Set();
  for (const item of limits) {
    const message = formatLimitationDisplay(item);
    const key = item && item.code ? String(item.code) : limitationKey(message);
    if (seenLimitKeys.has(key)) continue;
    seenLimitKeys.add(key);
    dedupedLimits.push(item);
    if (dedupedLimits.length >= 5) break;
  }
  const synthesisMode = output && typeof output.synthesis_mode === "string" ? output.synthesis_mode : "";
  const textEmptyMessage = ["limited_judgment", "limited_analysis", "limited_outlook", "insufficient_comparison"].includes(synthesisMode)
    ? "No validated text evidence in final bundle."
    : "No text evidence returned.";

  renderHtmlList(
    numericEvidenceList,
    numeric,
    formatNumericEvidenceHtml,
    "No numeric evidence returned.",
  );
  renderTextEvidenceList(textEvidenceList, text, textEmptyMessage);
  renderHtmlList(
    limitationsList,
    dedupedLimits,
    formatLimitationHtml,
    "No limitations reported.",
  );
}

function setHealthBadge(label, tone) {
  healthStatus.textContent = label;
  healthStatus.className = `badge ${tone}`;
}

function getApiBase() {
  return window.location.origin;
}

function makeUrl(path) {
  return new URL(path, getApiBase()).toString();
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null
      ? payload.detail || JSON.stringify(payload)
      : String(payload);
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return payload;
}

function renderCitations(citations) {
  citationsList.innerHTML = "";
  if (!Array.isArray(citations) || citations.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No citations returned.";
    citationsList.appendChild(li);
    return;
  }

  const structured = [];
  const documents = [];
  for (const c of citations) {
    if ((c.source_kind || "").toLowerCase() === "structured") {
      structured.push(c);
    } else {
      documents.push(c);
    }
  }

  function appendGroupHeader(text) {
    const li = document.createElement("li");
    li.textContent = text;
    li.className = "citation-group";
    citationsList.appendChild(li);
  }

  if (structured.length > 0) {
    appendGroupHeader("Structured Sources");
  }
  for (const c of structured) {
    const li = document.createElement("li");
    const metric = c.metric || "-";
    const periodType = c.period_type || "-";
    const periodEnd = c.period_end || c.period || "-";
    const provider = c.source_provider || "-";
    const confidence = c.confidence || "-";
    const warning = c.reconciliation_warning ? ` | warning=${c.reconciliation_warning}` : "";
    li.textContent = `${c.source || "-"} | ${metric} | ${periodType} | ${periodEnd} | provider=${provider} | confidence=${confidence}${warning}`;
    citationsList.appendChild(li);
  }

  if (documents.length > 0) {
    appendGroupHeader("Document Evidence");
  }
  for (const c of documents) {
    const li = document.createElement("li");
    const header = `${c.source || "-"} | ${c.filing_type || "-"} | ${c.period || "-"}`;
    const fallback = c.section_fallback ? " | section_fallback" : "";
    const detail = `${c.section || "-"}${fallback} | ${c.text_snippet || ""}`.trim();
    li.textContent = `${header} | ${detail}`;
    citationsList.appendChild(li);
  }
}

function summarizeTextEvidence(item) {
  const row = item && typeof item === "object" ? item : {};
  const explicitTheme = row.risk_theme || row.theme_name || row.evidence_summary;
  if (explicitTheme) {
    return cleanUserVisibleText(explicitTheme);
  }
  const ticker = String(row.ticker || "").toUpperCase();
  const section = String(row.section || "").toUpperCase();
  const dimension = String(row.dimension_id || "").toLowerCase();
  const text = String(row.supporting_snippet || row.text_snippet || "").toLowerCase();
  const pieces = [];
  if (dimension === "business_model" || section === "ITEM_1" || section === "BUSINESS") {
    if (/gpu|graphics processing|data center|datacenter|gaming|geforce|professional visualization|automotive|products|services|customers|markets|platforms/.test(text)) {
      pieces.push("业务模式 / 产品与服务");
    }
  }
  if (dimension === "moat_and_competitive_risk" || section === "ITEM_1A") {
    const riskPieces = [];
    if (/competition|competitive|competitor/.test(text)) riskPieces.push("竞争");
    if (/demand/.test(text)) riskPieces.push("需求");
    if (/supply chain|supply/.test(text)) riskPieces.push("供应链");
    if (/regulation|regulatory|legal|litigation/.test(text)) riskPieces.push("监管/法律");
    if (riskPieces.length > 0) {
      pieces.push(`${riskPieces.slice(0, 3).join(" / ")}风险`);
    }
  }
  if (/new product|new service|product introduction|service introduction|launch|demand/.test(text)) {
    pieces.push("新产品和需求不确定性");
  }
  if (/competition|competitive|competitor|market pressure|competitive pressure/.test(text)) {
    pieces.push(ticker === "AMZN" ? "多业务线竞争" : "市场竞争风险");
  }
  if (/macro|macroeconomic|economic|inflation|foreign exchange|interest rate/.test(text)) {
    pieces.push("宏观不确定性");
  }
  if (/regulation|regulatory|legal|litigation|antitrust|compliance/.test(text)) {
    pieces.push("监管/法律事项");
  }
  if (/reinvestment|operating leverage|fulfillment|logistics|cost pressure/.test(text)) {
    pieces.push("再投资和运营压力");
  }
  if (pieces.length === 0) {
    pieces.push(section === "ITEM_1" || section === "BUSINESS" ? "业务模式披露文本" : "已验证披露文本风险背景");
  }
  return Array.from(new Set(pieces)).slice(0, 2).join("和");
}

function formatTextEvidenceItem(item) {
  const row = item && typeof item === "object" ? item : {};
  const header = [
    row.ticker || "-",
    row.form_type || "-",
    row.section || "-",
    summarizeTextEvidence(row),
  ].map((part) => escapeHtml(cleanUserVisibleText(part))).join(" | ");
  const raw = String(row.supporting_snippet || row.text_snippet || "").trim();
  if (!raw) {
    return `<span>${header}</span>`;
  }
  return `<span>${header}</span><details class="raw-snippet"><summary>Show raw snippet</summary><pre>${escapeHtml(raw)}</pre></details>`;
}

function renderTextEvidenceList(container, items, emptyText) {
  container.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = emptyText;
    container.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = formatTextEvidenceItem(item);
    container.appendChild(li);
  });
}

function sanitizeTraceDisplayText(value) {
  const raw = String(value === null || value === undefined ? "" : value);
  if (/yfinance/i.test(raw) && /fallback/i.test(raw)) {
    return "部分结构化财务数据来自 yfinance，可信度为 medium。";
  }
  return raw
    .replace(/\bREQ-[A-Za-z0-9_-]+/g, "[requirement]")
    .replace(/\bdependency_[A-Za-z0-9_]+/g, "dependency detail")
    .replace(/\bnumeric_only_[A-Za-z0-9_]+/g, "limited numeric evidence")
    .replace(/\b(?:unsupported_claims_present|requirement_partial|requirement_missing|numeric_only_comparison|required_evidence_missing)\b/g, "limited evidence")
    .replace(/Required evidence is missing/gi, "Current evidence is limited")
    .replace(/\bfallback\b/gi, "alternate data source");
}

function cleanTraceFlag(flag) {
  const item = flag && typeof flag === "object" ? flag : {};
  const refs = Array.isArray(item.evidence_refs)
    ? item.evidence_refs.map((ref) => sanitizeTraceDisplayText(ref)).filter(Boolean)
    : [];
  return {
    severity: sanitizeTraceDisplayText(item.severity || "info"),
    category: sanitizeTraceDisplayText(item.category || "").replaceAll("_", " "),
    message: sanitizeTraceDisplayText(item.message || ""),
    evidence_refs: refs,
  };
}

function traceMetricListText(metrics) {
  const items = Array.isArray(metrics)
    ? metrics.map((item) => sanitizeTraceDisplayText(item)).filter(Boolean)
    : [];
  return items.length > 0 ? items.join(", ") : "";
}

function buildMethodologyTraceSummary(trace) {
  const packet = trace && trace.evidence_packet && typeof trace.evidence_packet === "object"
    ? trace.evidence_packet
    : {};
  const selectedFramework = trace.selected_framework
    || (trace.trace_summary && trace.trace_summary.analysis_framework_id)
    || (trace.selected_analysis_framework && trace.selected_analysis_framework.framework_id)
    || (trace.selected_analysis_framework && trace.selected_analysis_framework.id)
    || "";
  const activeDimensions = Array.isArray(trace.active_dimensions) && trace.active_dimensions.length > 0
    ? trace.active_dimensions
    : (Array.isArray(packet.active_dimensions) ? packet.active_dimensions : []);
  const dimensionStatusById = trace.dimension_status_by_id && typeof trace.dimension_status_by_id === "object"
    ? trace.dimension_status_by_id
    : {};
  const legacyDimensionStatusMap = trace.dimension_status_map && typeof trace.dimension_status_map === "object"
    ? trace.dimension_status_map
    : {};
  const dimensionStatusSource = Object.keys(dimensionStatusById).length > 0
    ? dimensionStatusById
    : legacyDimensionStatusMap;
  const dimensionStatus = dimensionStatusSource && typeof dimensionStatusSource === "object"
    ? dimensionStatusSource
    : {};
  const listFromTrace = (items) => (Array.isArray(items)
    ? items.map((item) => sanitizeTraceDisplayText(item)).filter(Boolean)
    : []);
  const dimensionsWithStatus = (statusName) => Object.entries(dimensionStatus)
    .filter(([, item]) => item && typeof item === "object" && String(item.status || "") === statusName)
    .map(([dimensionId]) => sanitizeTraceDisplayText(dimensionId))
    .filter(Boolean);
  const satisfiedDimensions = listFromTrace(trace.satisfied_dimensions).length > 0
    ? listFromTrace(trace.satisfied_dimensions)
    : (listFromTrace(trace.covered_dimensions).length > 0 ? listFromTrace(trace.covered_dimensions) : dimensionsWithStatus("satisfied"));
  const partialDimensions = listFromTrace(trace.partial_dimensions).length > 0
    ? listFromTrace(trace.partial_dimensions)
    : dimensionsWithStatus("partial");
  const missingDimensions = listFromTrace(trace.missing_dimensions).length > 0
    ? listFromTrace(trace.missing_dimensions)
    : dimensionsWithStatus("missing");
  const redFlags = Array.isArray(trace.red_flags) && trace.red_flags.length > 0
    ? trace.red_flags
    : (Array.isArray(packet.red_flags) ? packet.red_flags : []);
  const missingFlags = Array.isArray(trace.missing_evidence_flags) && trace.missing_evidence_flags.length > 0
    ? trace.missing_evidence_flags
    : (Array.isArray(packet.missing_evidence_flags) ? packet.missing_evidence_flags : []);
  const requiredMissing = [];
  const caveats = [];
  const caveatKeys = new Set();
  for (const [dimensionId, item] of Object.entries(dimensionStatus)) {
    const status = item && typeof item === "object" ? item : {};
    const dimension = sanitizeTraceDisplayText(dimensionId);
    const missingRequired = traceMetricListText(status.required_missing || []);
    if (missingRequired) {
      requiredMissing.push({
        severity: "high",
        category: dimension,
        message: `Missing required evidence: ${missingRequired}`,
        evidence_refs: [],
      });
    }
    const enhancedMissing = traceMetricListText(status.enhanced_missing || []);
    if (enhancedMissing) {
      const message = `Enhanced/optional evidence unavailable: ${enhancedMissing}`;
      const key = `${dimension}:${message}`;
      if (!caveatKeys.has(key)) {
        caveatKeys.add(key);
        caveats.push({
          severity: "medium",
          category: dimension,
          message,
          evidence_refs: [],
        });
      }
    }
    const limitations = Array.isArray(status.limitations)
      ? status.limitations
      : (status.limitation ? [status.limitation] : []);
    for (const limitation of limitations) {
      const message = sanitizeTraceDisplayText(limitation);
      if (!message) continue;
      const key = `${dimension}:${message}`;
      if (!caveatKeys.has(key)) {
        caveatKeys.add(key);
        caveats.push({
          severity: String(status.status || "") === "satisfied" ? "low" : "medium",
          category: dimension,
          message,
          evidence_refs: [],
        });
      }
    }
  }
  const fallbackMissingEvidence = requiredMissing.length > 0
    ? requiredMissing
    : (Object.keys(dimensionStatus).length > 0 ? [] : missingFlags.map(cleanTraceFlag).filter((item) => item.message));
  return {
    selected_framework: sanitizeTraceDisplayText(selectedFramework || "None"),
    active_dimensions: activeDimensions.map((item) => sanitizeTraceDisplayText(item)).filter(Boolean),
    dimension_status: Object.fromEntries(
      Object.entries(dimensionStatus).map(([dimensionId, item]) => [
        sanitizeTraceDisplayText(dimensionId),
        sanitizeTraceDisplayText(item && typeof item === "object" ? item.status || "unknown" : String(item || "unknown")),
      ]),
    ),
    satisfied_dimensions: satisfiedDimensions,
    partial_dimensions: partialDimensions,
    missing_dimensions: missingDimensions,
    red_flags: redFlags.map(cleanTraceFlag).filter((item) => item.message),
    missing_evidence: fallbackMissingEvidence,
    caveats,
  };
}

function formatMethodologyTraceSummary(trace) {
  const methodology = buildMethodologyTraceSummary(trace || {});
  const statusLines = Object.entries(methodology.dimension_status).map(([dimension, status]) => `- ${dimension}: ${status}`);
  const flagLines = methodology.red_flags.map(
    (flag) => `- [${flag.severity}] ${flag.category ? `${flag.category}: ` : ""}${flag.message}`,
  );
  const missingLines = methodology.missing_evidence.map(
    (flag) => `- [${flag.severity}] ${flag.category ? `${flag.category}: ` : ""}${flag.message}`,
  );
  const caveatLines = (methodology.caveats || []).map(
    (flag) => `- [${flag.severity}] ${flag.category ? `${flag.category}: ` : ""}${flag.message}`,
  );
  return [
    "Methodology Trace",
    `Selected Framework: ${methodology.selected_framework || "None"}`,
    `Active Dimensions: ${methodology.active_dimensions.length > 0 ? methodology.active_dimensions.join(", ") : "None"}`,
    `Satisfied Dimensions: ${methodology.satisfied_dimensions.length > 0 ? methodology.satisfied_dimensions.join(", ") : "None"}`,
    `Partial Dimensions: ${methodology.partial_dimensions.length > 0 ? methodology.partial_dimensions.join(", ") : "None"}`,
    `Missing Dimensions: ${methodology.missing_dimensions.length > 0 ? methodology.missing_dimensions.join(", ") : "None"}`,
    "",
    "Dimension Status",
    ...(statusLines.length > 0 ? statusLines : ["- None"]),
    "",
    "Red Flags",
    ...(flagLines.length > 0 ? flagLines : ["- No red flags reported."]),
    "",
    "Missing Evidence",
    ...(missingLines.length > 0 ? missingLines : ["- No missing evidence reported."]),
    "",
    "Caveats",
    ...(caveatLines.length > 0 ? caveatLines : ["- No caveats reported."]),
  ].join("\n");
}

function renderTraceSummary(trace) {
  const output = trace && typeof trace.output === "object" ? trace.output : {};
  const view = output && typeof output.view === "object" ? output.view : {};
  const backendSummary = trace && typeof trace.trace_summary === "object" ? trace.trace_summary : null;
  const evidencePlan = trace && typeof trace.evidence_plan === "object" ? trace.evidence_plan : {};
  const requirements = Array.isArray(trace.evidence_requirements)
    ? trace.evidence_requirements
    : (Array.isArray(evidencePlan.evidence_requirements) ? evidencePlan.evidence_requirements : []);
  const sufficiency = trace && typeof trace.evidence_sufficiency === "object" ? trace.evidence_sufficiency : {};
  const collected = trace && typeof trace.collected_evidence_by_requirement === "object"
    ? trace.collected_evidence_by_requirement
    : {};
  const retryHistory = Array.isArray(trace.retry_history)
    ? trace.retry_history
    : (Array.isArray(trace.evidence_retry_history) ? trace.evidence_retry_history : []);
  const fallbackSummary = {
    trace_id: trace.trace_id,
    user_query: trace.user_query,
    task_type: trace.task_type,
    data_route: trace.data_route,
    companies: trace.companies,
    comparison_target: trace.comparison_target,
    time_range: trace.time_range,
    requested_metrics: trace.requested_metrics,
    selected_tools: trace.selected_tools,
    market_reaction_requested: !!trace.market_reaction_requested,
    market_reaction_events: Array.isArray(trace.event_results)
      ? trace.event_results.reduce((acc, x) => acc + ((x && x.data && Array.isArray(x.data.events)) ? x.data.events.length : 0), 0)
      : 0,
    evidence_loop_count: trace.evidence_loop_count,
    evidence_requirements_count: requirements.length,
    collected_requirement_count: Object.keys(collected).length,
    missing_requirements_count: Array.isArray(trace.missing_requirements)
      ? trace.missing_requirements.length
      : (Array.isArray(sufficiency.missing_requirements) ? sufficiency.missing_requirements.length : 0),
    missing_required_requirements_count: Number.isFinite(Number(trace.missing_required_requirements_count))
      ? Number(trace.missing_required_requirements_count)
      : 0,
    missing_optional_requirements_count: Number.isFinite(Number(trace.missing_optional_requirements_count))
      ? Number(trace.missing_optional_requirements_count)
      : 0,
    missing_enhanced_requirements_count: Number.isFinite(Number(trace.missing_enhanced_requirements_count))
      ? Number(trace.missing_enhanced_requirements_count)
      : 0,
    sufficiency_status: sufficiency.overall_status || output.sufficiency_status || "",
    degradation_reason: trace.degradation_reason || sufficiency.degradation_reason || output.degradation_reason || "",
    retry_count: retryHistory.length,
    output_protocol: output.protocol_version || "",
    output_view_kind: view.kind || "",
    limitations_count: Array.isArray(output.limitations) ? output.limitations.length : 0,
  };
  const summary = backendSummary && Object.keys(backendSummary).length > 0
    ? {
        trace_id: trace.trace_id,
        user_query: trace.user_query,
        task_type: trace.task_type,
        answer_mode: trace.answer_mode,
        safety_intent: trace.safety_intent,
        evidence_requirements_count: requirements.length,
        validated_numeric_evidence_count: trace.validated_numeric_evidence_count,
        validated_text_evidence_count: trace.validated_text_evidence_count,
        limitations_count: Array.isArray(output.limitations) ? output.limitations.length : 0,
        ...backendSummary,
      }
    : fallbackSummary;
  traceSummary.textContent = `${formatMethodologyTraceSummary(trace)}\n\nTrace Summary JSON\n${JSON.stringify(summary, null, 2)}`;
  traceJson.textContent = JSON.stringify(trace, null, 2);
}

function statusClass(status) {
  const value = String(status || "").toLowerCase();
  if (["passed", "satisfied", "complete", "high", "ok", "used", "live"].includes(value)) return "good";
  if (["passed_with_warnings", "repaired", "partial", "retried", "medium", "warning", "planned", "not_run", "optional_missing", "optional missing", "optional", "diagnostic", "optional diagnostic", "optional_context"].includes(value)) return "warn";
  if (["blocked", "failed", "missing", "error", "low"].includes(value)) return "bad";
  return "neutral";
}

function badge(text) {
  const value = String(text || "unknown");
  return `<span class="badge ${statusClass(value)}">${escapeHtml(value)}</span>`;
}

function listHtml(items, emptyText, formatter = (item) => escapeHtml(String(item))) {
  if (!Array.isArray(items) || items.length === 0) {
    return `<p class="empty">${escapeHtml(emptyText)}</p>`;
  }
  return `<ul class="audit-list">${items.map((item) => `<li>${formatter(item)}</li>`).join("")}</ul>`;
}

function renderTraceWorkflow(ui) {
  const nodes = Array.isArray(ui.nodes) ? ui.nodes : [];
  const edges = Array.isArray(ui.edges) ? ui.edges : [];
  const toolCalls = Array.isArray(ui.tool_calls) ? ui.tool_calls : [];
  const companies = Array.isArray(ui.companies)
    ? ui.companies.map((item) => (item && item.ticker ? item.ticker : "")).filter(Boolean)
    : [];
  const timeline = Array.isArray(ui.timeline) ? ui.timeline : [];
  const overviewHtml = `
    <div class="audit-kpis">
      <div><span>Companies</span><strong>${escapeHtml(companies.length ? companies.join(", ") : "-")}</strong></div>
      <div><span>Task</span><strong>${escapeHtml(ui.task_type || "-")}</strong></div>
      <div><span>Contract</span><strong>${badge(ui.contract_status || "not_run")}</strong></div>
      <div><span>Events</span><strong>${escapeHtml(String(timeline.length))}</strong></div>
    </div>
  `;
  const nodeHtml = nodes.map((node) => `
    <div class="workflow-node ${statusClass(node.status)}">
      <strong>${escapeHtml(node.label || node.id || "")}</strong>
      <span>${escapeHtml(node.id || "")}</span>
      ${badge(node.status)}
    </div>
  `).join("");
  const takenEdges = edges.filter((edge) => edge.taken);
  const edgeHtml = takenEdges.length
    ? listHtml(takenEdges, "No workflow edges recorded.", (edge) => `${escapeHtml(edge.source)} -> ${escapeHtml(edge.target)}${edge.label ? ` (${escapeHtml(edge.label)})` : ""}`)
    : "<p class=\"empty\">No executed edge data available.</p>";
  const toolHtml = toolCalls.length
    ? `<table class="audit-table"><thead><tr><th>Tool</th><th>Requirement</th><th>Status</th><th>Returned</th><th>Latency</th></tr></thead><tbody>${toolCalls.map((call) => `
        <tr>
          <td>${escapeHtml(call.tool_name || "-")}</td>
          <td>${escapeHtml(call.requirement_id || "-")}</td>
          <td>${badge(call.ok ? "passed" : "failed")}</td>
          <td>${escapeHtml(String(call.returned_count ?? "-"))}</td>
          <td>${escapeHtml(call.latency_ms === null || call.latency_ms === undefined ? "-" : `${call.latency_ms} ms`)}</td>
        </tr>
      `).join("")}</tbody></table>`
    : "<p class=\"empty\">No protocol tool call summaries recorded.</p>";
  return `
    ${overviewHtml}
    <div class="workflow-grid">${nodeHtml}</div>
    <h3>Executed Edges</h3>
    ${edgeHtml}
    <h3>Tool Calls</h3>
    ${toolHtml}
  `;
}

function renderTracePlan(ui) {
  const requirements = ui.evidence_plan && Array.isArray(ui.evidence_plan.requirements)
    ? ui.evidence_plan.requirements
    : [];
  if (!requirements.length) return "<p class=\"empty\">No EvidencePlan requirements recorded.</p>";
  return `<table class="audit-table"><thead><tr><th>ID</th><th>Dimension</th><th>Type</th><th>Company</th><th>Tool</th><th>Scope</th><th>Status</th><th>Missing Reason</th></tr></thead><tbody>${requirements.map((req) => `
    <tr>
      <td>${escapeHtml(req.requirement_id || "-")}</td>
      <td>${escapeHtml(req.dimension || "-")}</td>
      <td>${escapeHtml(req.evidence_type || "-")}</td>
      <td>${escapeHtml(req.company || "-")}</td>
      <td>${escapeHtml(req.tool || "-")}</td>
      <td>${badge(req.scope || (req.required === false ? "optional" : "required"))}</td>
      <td>${badge(req.status_label || req.status || "planned")}</td>
      <td>${escapeHtml(req.missing_reason || "")}</td>
    </tr>
  `).join("")}</tbody></table>`;
}

function jsonBlock(value) {
  const safe = value && typeof value === "object" ? value : {};
  const text = Object.keys(safe).length ? JSON.stringify(safe, null, 2) : "{}";
  return `<pre class="pre audit-json">${escapeHtml(text)}</pre>`;
}

function renderRequiredAnswerParts(parts, statuses = {}, gaps = {}) {
  if (!Array.isArray(parts) || parts.length === 0) {
    return "<p class=\"empty\">No required answer parts recorded.</p>";
  }
  return `<table class="audit-table"><thead><tr><th>ID</th><th>Description</th><th>Required</th><th>Status</th><th>Gap</th></tr></thead><tbody>${parts.map((part) => {
    const id = String((part && part.id) || "");
    const status = statuses && typeof statuses === "object" ? (statuses[id] || {}) : {};
    const gap = gaps && typeof gaps === "object" ? (gaps[id] || "") : "";
    const statusLabel = status && typeof status === "object" ? (status.status || "planned") : (status || "planned");
    const gapText = typeof gap === "object" ? JSON.stringify(gap) : String(gap || "");
    return `<tr>
      <td>${escapeHtml(id || "-")}</td>
      <td>${escapeHtml((part && part.description) || "-")}</td>
      <td>${badge(part && part.required === false ? "optional" : "required")}</td>
      <td>${badge(statusLabel)}</td>
      <td>${escapeHtml(gapText || "-")}</td>
    </tr>`;
  }).join("")}</tbody></table>`;
}

function renderTraceResearchPlan(ui) {
  const research = ui.research_plan && typeof ui.research_plan === "object" ? ui.research_plan : {};
  const summary = research.summary || {};
  const used = research.used || ui.research_plan_used || {};
  const validation = research.validation || ui.research_plan_validation || {};
  const relevance = research.relevance_decision || ui.relevance_decision || {};
  const parts = research.required_answer_parts || ui.required_answer_parts || [];
  const gaps = research.evidence_gap_by_answer_part || ui.evidence_gap_by_answer_part || {};
  const statuses = research.answer_part_status_by_id || ui.answer_part_status_by_id || {};
  const legacy = research.legacy_evidence_plan || ui.legacy_evidence_plan || {};
  return `
    <div class="audit-kpis">
      <div><span>Question Type</span><strong>${escapeHtml(summary.question_type || used.question_type || "-")}</strong></div>
      <div><span>Plan Used</span><strong>${badge(summary.used ? "used" : "not_run")}</strong></div>
      <div><span>Validation</span><strong>${badge(summary.valid ? "passed" : "warning")}</strong></div>
      <div><span>Relevance</span><strong>${badge(ui.relevance_status || research.relevance_status || "not_run")}</strong></div>
      <div><span>Source</span><strong>${escapeHtml(summary.source || research.source || ui.research_plan_source || "-")}</strong></div>
      <div><span>Duration</span><strong>${escapeHtml(String(summary.duration_ms || research.duration_ms || ui.research_plan_duration_ms || 0))}ms</strong></div>
    </div>
    <h3>User Goal</h3>
    <p>${escapeHtml(summary.user_goal || used.user_goal || "-")}</p>
    <h3>Required Answer Parts</h3>
    ${renderRequiredAnswerParts(parts, statuses, gaps)}
    <h3>Fallback Policy</h3>
    <p>${escapeHtml(used.fallback_answer_policy || "-")}</p>
    <h3>Plan Source</h3>
    ${jsonBlock({
      source: summary.source || research.source || ui.research_plan_source,
      fallback_reason: summary.fallback_reason || research.fallback_reason || ui.research_plan_fallback_reason,
      duration_ms: summary.duration_ms || research.duration_ms || ui.research_plan_duration_ms,
      partial_required_answer_parts: research.partial_required_answer_parts || ui.partial_required_answer_parts || [],
    })}
    <h3>Relevance Decision</h3>
    ${jsonBlock(relevance)}
    <h3>Validation</h3>
    ${jsonBlock(validation)}
    <h3>Raw LLM Plan</h3>
    ${jsonBlock(research.raw || ui.research_plan_raw)}
    <h3>Validated Plan</h3>
    ${jsonBlock(research.validated || ui.research_plan_validated)}
    <h3>Used Plan</h3>
    ${jsonBlock(used)}
    <h3>Legacy EvidencePlan</h3>
    ${jsonBlock(legacy)}
  `;
}

function evidenceTable(rows, columns, emptyText) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return `<p class="empty">${escapeHtml(emptyText)}</p>`;
  }
  return `<table class="audit-table"><thead><tr>${columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `
    <tr>${columns.map((col) => `<td>${escapeHtml(String(row[col.key] ?? ""))}</td>`).join("")}</tr>
  `).join("")}</tbody></table>`;
}

function renderTracePacket(ui) {
  const packet = ui.evidence_packet || {};
  return `
    <h3>Numeric Evidence</h3>
    ${evidenceTable(packet.numeric_evidence, [
      { key: "evidence_id", label: "ID" },
      { key: "ticker", label: "Ticker" },
      { key: "metric", label: "Metric" },
      { key: "value", label: "Value" },
      { key: "period_end", label: "Period" },
      { key: "source_provider", label: "Provider" },
      { key: "confidence", label: "Confidence" },
    ], "No numeric evidence.")}
    <h3>Text Evidence</h3>
    ${evidenceTable(packet.text_evidence, [
      { key: "evidence_id", label: "ID" },
      { key: "ticker", label: "Ticker" },
      { key: "section", label: "Section" },
      { key: "filing_id", label: "Filing" },
      { key: "supporting_snippet", label: "Snippet" },
    ], "No text evidence.")}
    <h3>Computed / Event Evidence</h3>
    ${evidenceTable([...(packet.computed_metrics || []), ...(packet.event_evidence || [])], [
      { key: "evidence_id", label: "ID" },
      { key: "ticker", label: "Ticker" },
      { key: "metric", label: "Metric" },
      { key: "value", label: "Value" },
      { key: "period_end", label: "Period" },
      { key: "source_provider", label: "Provider" },
    ], "No computed or event evidence.")}
    <h3>Limitations</h3>
    ${listHtml(packet.limitations, "No limitations recorded.")}
  `;
}

function renderTraceDimensions(ui) {
  const dimensions = Array.isArray(ui.dimensions) ? ui.dimensions : [];
  if (!dimensions.length) return "<p class=\"empty\">No DimensionStatus data recorded.</p>";
  return `<table class="audit-table"><thead><tr><th>Dimension</th><th>Status</th><th>Evidence</th><th>Missing</th><th>Limitations</th></tr></thead><tbody>${dimensions.map((dim) => {
    const missing = [...(dim.required_missing || []), ...(dim.enhanced_missing || [])].filter(Boolean).join(", ");
    return `<tr>
      <td>${escapeHtml(dim.dimension_id || "-")}</td>
      <td>${badge(dim.status || "unknown")}</td>
      <td>${escapeHtml(String(dim.evidence_count ?? 0))}</td>
      <td>${escapeHtml(missing || "-")}</td>
      <td>${escapeHtml((dim.limitations || []).join("; "))}</td>
    </tr>`;
  }).join("")}</tbody></table>`;
}

function renderTraceCitations(ui) {
  const citations = Array.isArray(ui.citations) ? ui.citations : [];
  if (!citations.length) return "<p class=\"empty\">No citations or evidence IDs recorded.</p>";
  return `<table class="audit-table"><thead><tr><th>Citation</th><th>Type</th><th>Company</th><th>Metric / Section</th><th>Period</th><th>Used</th><th>Valid</th></tr></thead><tbody>${citations.map((citation) => `
    <tr>
      <td>${escapeHtml(citation.citation_id || "-")}</td>
      <td>${escapeHtml(citation.evidence_type || "-")}</td>
      <td>${escapeHtml(citation.company || "-")}</td>
      <td>${escapeHtml(citation.metric || citation.section || "-")}</td>
      <td>${escapeHtml(citation.period || "-")}</td>
      <td>${badge(citation.used_in_answer ? "used" : "unused")}</td>
      <td>${badge(citation.valid ? "passed" : "failed")}</td>
    </tr>
  `).join("")}</tbody></table>`;
}

function renderTraceContract(ui) {
  const contract = ui.contract || {};
  return `
    <div class="contract-summary">
      <p><strong>Status:</strong> ${badge(contract.status || ui.contract_status || "not_run")}</p>
      <p><strong>Repair attempts:</strong> ${escapeHtml(String(contract.repair_attempts ?? ui.repair_attempts ?? 0))}</p>
      <p><strong>Evidence retries:</strong> ${escapeHtml(String(contract.evidence_retry_count ?? ui.evidence_retry_count ?? 0))}</p>
      <p>${escapeHtml(contract.public_summary || "No public contract summary recorded.")}</p>
    </div>
    <h3>Violation Codes</h3>
    ${listHtml(contract.violation_codes, "No contract violations recorded.")}
    <h3>Warnings</h3>
    ${listHtml(contract.warnings, "No contract warnings recorded.", (item) => escapeHtml(JSON.stringify(item)))}
    <h3>Repair Actions</h3>
    ${listHtml(contract.repair_actions, "No repair actions recorded.", (action) => escapeHtml(JSON.stringify(action)))}
  `;
}

function renderTraceReport(ui) {
  const report = ui.report;
  if (!report) return "<p class=\"empty\">This trace did not produce a full company-analysis report.</p>";
  const sections = Array.isArray(report.sections) ? report.sections : [];
  const sectionNav = sections.map((section) => `
    <li>${escapeHtml(section.title || section.section_id || "-")} ${badge(section.section_status || "unknown")}</li>
  `).join("");
  const markdown = report.markdown || sections.map((section) => section.markdown || "").join("\n\n");
  const rendered = window.marked && window.DOMPurify
    ? window.DOMPurify.sanitize(window.marked.parse(markdown || "", { gfm: true, breaks: true }), { USE_PROFILES: { html: true } })
    : simpleMarkdownToHtml(markdown || "");
  return `
    <div class="report-layout">
      <aside><h3>${escapeHtml(report.title || "Report")}</h3><ul class="audit-list">${sectionNav}</ul></aside>
      <article class="report-markdown">${rendered || "<p class=\"empty\">No report markdown recorded.</p>"}</article>
    </div>
  `;
}

function renderTraceTab(ui, tabId) {
  if (!traceTabContent) return;
  const renderers = {
    workflow: renderTraceWorkflow,
    research: renderTraceResearchPlan,
    plan: renderTracePlan,
    packet: renderTracePacket,
    dimensions: renderTraceDimensions,
    citations: renderTraceCitations,
    contract: renderTraceContract,
    report: renderTraceReport,
  };
  const render = renderers[tabId] || renderTraceWorkflow;
  traceTabContent.innerHTML = render(ui || {});
}

function setTraceTab(tabId) {
  currentTraceTab = tabId || "workflow";
  if (!hasDocument) return;
  document.querySelectorAll(".trace-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.traceTab === currentTraceTab);
  });
  renderTraceTab(currentTraceUi, currentTraceTab);
}

function formatTraceUiSummary(ui) {
  const planSummary = ui.evidence_plan && ui.evidence_plan.summary ? ui.evidence_plan.summary : {};
  const packetSummary = ui.evidence_packet && ui.evidence_packet.summary ? ui.evidence_packet.summary : {};
  const contract = ui.contract || {};
  const research = ui.research_plan && typeof ui.research_plan === "object" ? ui.research_plan : {};
  const researchSummary = research.summary && typeof research.summary === "object" ? research.summary : {};
  return [
    "Trace Audit Summary",
    `Trace ID: ${ui.trace_id || ""}`,
    `Query: ${ui.query || ""}`,
    `Task Type: ${ui.task_type || ""}`,
    `Answer Mode: ${ui.answer_mode || ""}`,
    `Canonical Intent: ${((ui.canonical_intent || {}).intent_family) || ""}`,
    `Research Question Type: ${researchSummary.question_type || ((ui.research_plan_used || {}).question_type) || ""}`,
    `Research Goal: ${researchSummary.user_goal || ((ui.research_plan_used || {}).user_goal) || ""}`,
    `Evidence Policy: ${ui.evidence_policy_id || ((ui.evidence_policy || {}).policy_id) || ""}`,
    `Contract Status: ${ui.contract_status || "not_run"}`,
    `Relevance Status: ${ui.relevance_status || "not_run"}`,
    `Repair Attempts: ${ui.repair_attempts || 0}`,
    `Evidence Retries: ${ui.evidence_retry_count || 0}`,
    `Requirements: ${planSummary.requirement_count || 0} (${planSummary.missing_count || 0} missing, ${planSummary.partial_count || 0} partial)`,
    `Evidence: ${packetSummary.numeric_count || 0} numeric, ${packetSummary.text_count || 0} text, ${packetSummary.computed_count || 0} computed, ${packetSummary.event_count || 0} event`,
    `Dimensions: ${Array.isArray(ui.dimensions) ? ui.dimensions.length : 0}`,
    `Citations: ${Array.isArray(ui.citations) ? ui.citations.length : 0}`,
    `Contract Summary: ${contract.public_summary || ""}`,
  ].join("\n");
}

function markdownObjectValue(value) {
  if (!value || typeof value !== "object") return "N/A";
  const orderedKeys = [
    "code",
    "field",
    "reason",
    "message",
    "severity",
    "requirement_id",
    "requirement_ids",
    "dimension",
    "dimension_id",
    "validation_error_code",
    "validation_error_message",
    "status",
    "decision",
    "route",
    "source",
    "value",
  ];
  const parts = orderedKeys
    .filter((key) => Object.prototype.hasOwnProperty.call(value, key) && value[key] !== null && value[key] !== undefined && value[key] !== "")
    .map((key) => `${key}=${markdownValue(value[key])}`);
  if (parts.length) return parts.join(", ");
  const text = JSON.stringify(value);
  return text && text !== "{}" ? text : "N/A";
}

function markdownValue(value) {
  if (Array.isArray(value)) {
    const filtered = value.map((item) => markdownValue(item)).filter((item) => item && item !== "N/A");
    return filtered.length ? filtered.join(", ") : "N/A";
  }
  if (value && typeof value === "object") {
    return markdownObjectValue(value);
  }
  const text = String(value ?? "").trim();
  return text || "N/A";
}

function truncateMarkdownText(value, limit = 800) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (!text) return "N/A";
  return text.length > limit ? `${text.slice(0, limit).trim()}...` : text;
}

function debugLine(label, value) {
  return `${label}: ${markdownValue(value)}`;
}

function bulletLines(items, formatter) {
  if (!Array.isArray(items) || items.length === 0) return ["- N/A"];
  return items.map((item) => `- ${formatter(item)}`);
}

function toolBackend(call) {
  const input = call && typeof call.input_summary === "object" ? call.input_summary : {};
  return call.backend || input.backend || input.retrieval_backend || "N/A";
}

function toolFallback(call) {
  const input = call && typeof call.input_summary === "object" ? call.input_summary : {};
  if (call.fallback_after_timeout || input.fallback_after_timeout) return "fallback_after_timeout";
  if (call.fallback_after_error || input.fallback_after_error) return "fallback_after_error";
  if (input.lexical_first) return "lexical_first";
  return "N/A";
}

function toolStatus(call) {
  return call && call.ok === true ? "passed" : "failed";
}

function toolErrorText(call) {
  const error = call && call.error;
  if (!error) return "N/A";
  if (typeof error === "string") return error;
  if (typeof error === "object") return [error.code, error.message].filter(Boolean).join(": ") || JSON.stringify(error);
  return String(error);
}

function requirementRawStatus(req) {
  return String((req && (req.raw_status || req.status)) || "").toLowerCase();
}

function requirementIsRequired(req) {
  return !(req && req.required === false);
}

function requirementScope(req) {
  const scope = String((req && req.scope) || (req && req.requirement_scope) || "").toLowerCase();
  return scope === "" || scope === "required" ? "core" : scope;
}

function requirementIsBlocking(req) {
  return requirementIsRequired(req) && requirementScope(req) === "core";
}

function requirementScopeDescription(req) {
  const normalized = requirementScope(req).replace(/\s+/g, "_");
  if (normalized === "diagnostic" || normalized === "optional_diagnostic") return "non-blocking diagnostic";
  if (normalized === "optional" || normalized === "optional_context") return "non-blocking optional context";
  if (normalized && normalized !== "core") return `non-blocking ${normalized.replace(/_/g, " ")}`;
  return "non-blocking optional context";
}

function requirementReturned(req, callsByRequirement) {
  const call = callsByRequirement.get(String((req && req.requirement_id) || ""));
  if (call && call.returned_count !== undefined && call.returned_count !== null && call.returned_count !== "") {
    return markdownValue(call.returned_count);
  }
  const fallback = req && (req.returned ?? req.collected_count ?? req.usable_hit_count ?? req.raw_hit_count);
  return fallback !== undefined && fallback !== null && fallback !== "" ? markdownValue(fallback) : "N/A";
}

function requirementBackend(req, callsByRequirement) {
  const call = callsByRequirement.get(String((req && req.requirement_id) || ""));
  return call ? toolBackend(call) : "N/A";
}

function requirementFallback(req, callsByRequirement) {
  const call = callsByRequirement.get(String((req && req.requirement_id) || ""));
  return call ? toolFallback(call) : "N/A";
}

const debugProgressMetadataKeys = [
  "tool",
  "requirement_id",
  "company",
  "question_type",
  "contract_status",
  "relevance_status",
  "draft_release_decision",
  "final_status",
];

function progressMetadataSummary(metadata) {
  const source = metadata && typeof metadata === "object" ? metadata : {};
  return debugProgressMetadataKeys
    .filter((key) => Object.prototype.hasOwnProperty.call(source, key) && source[key] !== null && source[key] !== undefined && source[key] !== "")
    .map((key) => `${key}=${markdownValue(source[key])}`)
    .join("; ");
}

function progressEventDebugLine(item) {
  const event = markdownValue(item && item.event);
  const status = markdownValue(item && item.status);
  const elapsed = item && item.elapsed_ms !== undefined && item.elapsed_ms !== null ? `${markdownValue(item.elapsed_ms)}ms` : "N/A";
  const node = markdownValue(item && item.node);
  const message = truncateMarkdownText(item && item.message, 240);
  const metadata = progressMetadataSummary(item && item.metadata);
  const metadataText = metadata ? `; metadata: ${metadata}` : "";
  return `${event}: status=${status}; elapsed=${elapsed}; node=${node}; message=${message}${metadataText}`;
}

function buildDebugBundle(trace = {}, ui = {}) {
  const safeTrace = trace && typeof trace === "object" ? trace : {};
  const safeUi = ui && typeof ui === "object" && Object.keys(ui).length > 0 ? ui : safeTrace;
  const plan = safeUi.evidence_plan && typeof safeUi.evidence_plan === "object" ? safeUi.evidence_plan : {};
  const planSummary = plan.summary && typeof plan.summary === "object" ? plan.summary : {};
  const packet = safeUi.evidence_packet && typeof safeUi.evidence_packet === "object" ? safeUi.evidence_packet : {};
  const packetSummary = packet.summary && typeof packet.summary === "object" ? packet.summary : {};
  const contract = safeUi.contract && typeof safeUi.contract === "object" ? safeUi.contract : {};
  const researchPlan = safeUi.research_plan && typeof safeUi.research_plan === "object" ? safeUi.research_plan : {};
  const researchSummary = researchPlan.summary && typeof researchPlan.summary === "object" ? researchPlan.summary : {};
  const relevanceDecision = safeUi.relevance_decision && typeof safeUi.relevance_decision === "object"
    ? safeUi.relevance_decision
    : (researchPlan.relevance_decision && typeof researchPlan.relevance_decision === "object" ? researchPlan.relevance_decision : {});
  const contractDecision = safeUi.contract_decision && typeof safeUi.contract_decision === "object"
    ? safeUi.contract_decision
    : (safeTrace.contract_decision && typeof safeTrace.contract_decision === "object" ? safeTrace.contract_decision : {});
  const draftReleaseDecision = safeUi.draft_release_decision && typeof safeUi.draft_release_decision === "object"
    ? safeUi.draft_release_decision
    : (safeTrace.draft_release_decision && typeof safeTrace.draft_release_decision === "object" ? safeTrace.draft_release_decision : {});
  const semanticParser = safeUi.semantic_parser && typeof safeUi.semantic_parser === "object" ? safeUi.semantic_parser : {};
  const semanticDisagreement = semanticParser.disagreement && typeof semanticParser.disagreement === "object" ? semanticParser.disagreement : {};
  const canonicalIntent = safeUi.canonical_intent && typeof safeUi.canonical_intent === "object"
    ? safeUi.canonical_intent
    : (safeTrace.canonical_intent && typeof safeTrace.canonical_intent === "object" ? safeTrace.canonical_intent : {});
  const intentMergeDecision = safeUi.intent_merge_decision && typeof safeUi.intent_merge_decision === "object"
    ? safeUi.intent_merge_decision
    : (canonicalIntent.intent_merge_decision && typeof canonicalIntent.intent_merge_decision === "object" ? canonicalIntent.intent_merge_decision : {});
  const evidencePolicy = safeUi.evidence_policy && typeof safeUi.evidence_policy === "object"
    ? safeUi.evidence_policy
    : (safeTrace.evidence_policy && typeof safeTrace.evidence_policy === "object" ? safeTrace.evidence_policy : {});
  const requirements = Array.isArray(plan.requirements) ? plan.requirements : [];
  const toolCalls = Array.isArray(safeUi.tool_calls) ? safeUi.tool_calls : [];
  const dimensions = Array.isArray(safeUi.dimensions) ? safeUi.dimensions : [];
  const citations = Array.isArray(safeUi.citations) ? safeUi.citations : [];
  const rawProgressEvents = Array.isArray(safeUi.progress_events)
    ? safeUi.progress_events
    : (Array.isArray(safeTrace.progress_events) ? safeTrace.progress_events : []);
  const progressEvents = rawProgressEvents
    .filter((item) => item && typeof item === "object" && item.event && item.status && item.message)
    .slice(-20);
  const callsByRequirement = new Map();
  toolCalls.forEach((call) => {
    const rid = String((call && call.requirement_id) || "");
    if (!rid) return;
    const existing = callsByRequirement.get(rid);
    const existingReturned = Number((existing && existing.returned_count) || 0);
    const returned = Number((call && call.returned_count) || 0);
    const existingFallback = existing && toolFallback(existing) !== "N/A";
    const fallback = call && toolFallback(call) !== "N/A";
    if (!existing || (fallback && !existingFallback) || returned > existingReturned) {
      callsByRequirement.set(rid, call);
    }
  });
  const preferredToolCalls = [];
  const addedPreferredRequirements = new Set();
  requirements.forEach((req) => {
    const rid = String((req && req.requirement_id) || "");
    const call = callsByRequirement.get(rid);
    if (rid && call && !addedPreferredRequirements.has(rid)) {
      preferredToolCalls.push(call);
      addedPreferredRequirements.add(rid);
    }
  });
  toolCalls.forEach((call) => {
    const rid = String((call && call.requirement_id) || "");
    if (rid && callsByRequirement.has(rid)) {
      if (!addedPreferredRequirements.has(rid)) {
        preferredToolCalls.push(callsByRequirement.get(rid));
        addedPreferredRequirements.add(rid);
      }
      return;
    }
    preferredToolCalls.push(call);
  });

  const traceId = safeUi.trace_id || safeTrace.trace_id || "";
  const query = safeUi.query || safeUi.user_query || safeTrace.user_query || safeTrace.query || "";
  const finalAnswer = safeUi.final_answer || safeTrace.final_answer || safeTrace.answer || (safeUi.report && safeUi.report.markdown) || "";
  const blockingMissing = requirements.filter((req) => requirementRawStatus(req) === "missing" && requirementIsBlocking(req));
  const optionalMissing = requirements.filter((req) => requirementRawStatus(req) === "missing" && !requirementIsBlocking(req));
  const partialRequirements = requirements.filter((req) => requirementRawStatus(req) === "partial");
  const toolProblems = preferredToolCalls.filter((call) => (
    call && (call.ok === false || call.error || call.fallback_after_timeout || call.fallback_after_error || toolFallback(call) !== "N/A")
  ));
  const invalidCitations = citations.filter((citation) => citation && citation.valid === false);
  const answerPartStatuses = researchPlan.answer_part_status_by_id || safeUi.answer_part_status_by_id || {};
  const requiredAnswerParts = researchPlan.required_answer_parts || safeUi.required_answer_parts || [];
  const analyticalReasoning = safeUi.analytical_reasoning || {};
  const claimTiers = analyticalReasoning.claim_tiers || safeUi.claim_tiers || {};
  const analyticalClaims = analyticalReasoning.analytical_claims || safeUi.analytical_claims || [];
  const planCoverage = (safeUi.plan_coverage_decision && typeof safeUi.plan_coverage_decision === "object") ? safeUi.plan_coverage_decision : {};
  const requirementMerge = (safeUi.requirement_merge_summary && typeof safeUi.requirement_merge_summary === "object") ? safeUi.requirement_merge_summary : {};
  const evidenceValidationRecords = Array.isArray(safeUi.evidence_validation_records) ? safeUi.evidence_validation_records : [];
  const evidenceScope = (safeUi.evidence_scope && typeof safeUi.evidence_scope === "object") ? safeUi.evidence_scope : {};
  const evidenceScopeRows = Array.isArray(evidenceScope.rows)
    ? evidenceScope.rows
    : Object.values((safeUi.evidence_scope_by_ref && typeof safeUi.evidence_scope_by_ref === "object") ? safeUi.evidence_scope_by_ref : {});
  const scopeOverclaimCheck = (safeUi.scope_overclaim_check && typeof safeUi.scope_overclaim_check === "object")
    ? safeUi.scope_overclaim_check
    : ((contract.scope_overclaim_check && typeof contract.scope_overclaim_check === "object") ? contract.scope_overclaim_check : {});
  const scopeOverclaimViolations = Array.isArray(safeUi.scope_overclaim_violations)
    ? safeUi.scope_overclaim_violations
    : (Array.isArray(contract.scope_overclaim_violations) ? contract.scope_overclaim_violations : []);
  const evidenceSummaryWarnings = Array.isArray(scopeOverclaimCheck.evidence_summary_warnings)
    ? scopeOverclaimCheck.evidence_summary_warnings
    : [];
  const driverScopeCounts = analyticalReasoning.driver_scope_counts || safeUi.driver_scope_counts || {};
  const answerPartLines = Array.isArray(requiredAnswerParts) && requiredAnswerParts.length
    ? requiredAnswerParts.map((part) => {
      const id = String((part && part.id) || "");
      const status = (answerPartStatuses && answerPartStatuses[id]) || {};
      return `- ${markdownValue(id)}: ${markdownValue((status && status.status) || "planned")}${status && status.reason ? ` (${markdownValue(status.reason)})` : ""}`;
    })
    : ["- N/A"];

  const lines = [
    "# Agent Feedback Debug Bundle",
    "",
    "## 1. Key Issue Snapshot",
    debugLine("Trace ID", traceId),
    debugLine("Query", query),
    debugLine("Task Type", safeUi.task_type || safeTrace.task_type),
    debugLine("Answer Mode", safeUi.answer_mode || safeTrace.answer_mode),
    debugLine("Canonical Intent", `${markdownValue(canonicalIntent.intent_family)} / ${markdownValue(canonicalIntent.analysis_scope)} / ${markdownValue(canonicalIntent.time_focus)}`),
    debugLine("Evidence Policy", safeUi.evidence_policy_id || safeTrace.evidence_policy_id || evidencePolicy.policy_id),
    debugLine("Research Question Type", researchSummary.question_type || (safeUi.research_plan_used || {}).question_type),
    debugLine("Research Plan Used", researchSummary.used === true ? "yes" : "no"),
    debugLine("Research Plan Source", researchSummary.source || researchPlan.source || safeUi.research_plan_source),
    debugLine("Research Plan Fallback", researchSummary.fallback_reason || researchPlan.fallback_reason || safeUi.research_plan_fallback_reason),
    debugLine("Research Plan Duration", `${markdownValue(researchSummary.duration_ms || researchPlan.duration_ms || safeUi.research_plan_duration_ms || 0)}ms`),
    debugLine("Contract Status", safeUi.contract_status || safeTrace.contract_status),
    debugLine("Relevance Status", safeUi.relevance_status || safeTrace.relevance_status),
    debugLine("Relevance Decision", relevanceDecision.decision),
    debugLine("Analytical Reasoning", analyticalReasoning.analytical_reasoning_status || safeUi.analytical_reasoning_status),
    debugLine("Evidence Health", analyticalReasoning.evidence_health || safeUi.evidence_health),
    debugLine("Contract Decision", contract.decision || contractDecision.decision),
    debugLine("Draft Release", draftReleaseDecision.decision),
    debugLine("Semantic Parser", `${markdownValue(safeUi.semantic_parser_mode || semanticParser.mode)} / ${semanticParser.ok === true ? "ok" : "not used"} / ${markdownValue(semanticParser.source)}`),
    debugLine("Requirements", `${markdownValue(planSummary.requirement_count)} (${markdownValue(planSummary.missing_count)} blocking missing, ${markdownValue(planSummary.partial_count)} partial, ${markdownValue(planSummary.missing_optional_count)} optional missing)`),
    debugLine("Requirement Scope Counts", `core=${markdownValue(planSummary.core_count ?? (planSummary.scope_counts || {}).core)}, optional_context=${markdownValue(planSummary.optional_context_count ?? (planSummary.scope_counts || {}).optional_context)}, diagnostic=${markdownValue(planSummary.diagnostic_count ?? (planSummary.scope_counts || {}).diagnostic)}`),
    debugLine("Evidence", `${markdownValue(packetSummary.numeric_count)} numeric, ${markdownValue(packetSummary.text_count)} text, ${markdownValue(packetSummary.computed_count)} computed, ${markdownValue(packetSummary.event_count)} event`),
    debugLine("Dimensions", dimensions.length),
    debugLine("Citations", citations.length),
    debugLine("Contract Summary", contract.public_summary || safeTrace.contract_public_summary),
    "",
    "## 2. User-Facing Answer",
    markdownValue(finalAnswer),
    "",
    "## 3. Plan Coverage",
    `- strategy: ${markdownValue(planCoverage.strategy)}`,
    `- legacy core requirements: ${markdownValue(planCoverage.legacy_core_count)}`,
    `- research core requirements: ${markdownValue(planCoverage.research_core_count)}`,
    `- retained legacy core: ${markdownValue(planCoverage.retained_legacy_core_count)}`,
    `- added research requirements: ${markdownValue(planCoverage.added_research_requirement_ids || [])}`,
    `- coverage ratio: ${markdownValue(planCoverage.coverage_ratio)}`,
    `- warnings: ${markdownValue(planCoverage.warnings || [])}`,
    `- reason: ${markdownValue(planCoverage.reason)}`,
    "",
    "## 4. Requirement Merge",
    `- merged total requirements: ${markdownValue(requirementMerge.merged_total_requirements)}`,
    `- deduped requirements: ${markdownValue(requirementMerge.deduped_requirements)}`,
    `- legacy-only: ${markdownValue(requirementMerge.legacy_only_count)}`,
    `- research-only: ${markdownValue(requirementMerge.research_only_count)}`,
    `- legacy+research: ${markdownValue(requirementMerge.legacy_research_count)}`,
    `- retained legacy core: ${markdownValue(requirementMerge.retained_legacy_core_count)}`,
    "",
    "## 5. Important Debug Signals",
    "### Answer Part Status",
    ...answerPartLines,
    "",
    "### Analytical Reasoning",
    `- evidence-backed claims: ${markdownValue(claimTiers.evidence_backed || 0)}`,
    `- inferred claims: ${markdownValue(claimTiers.evidence_inferred || 0)}`,
    `- hypotheses to verify: ${markdownValue(claimTiers.hypothesis_to_verify || 0)}`,
    `- evidence health: ${markdownValue(analyticalReasoning.evidence_health || safeUi.evidence_health || "N/A")}`,
    `- company-level driver claims: ${markdownValue(driverScopeCounts.company || 0)}`,
    `- segment-level driver claims: ${markdownValue(driverScopeCounts.segment || 0)}`,
    `- product-level driver claims: ${markdownValue(driverScopeCounts.product || 0)}`,
    `- scope-bounded inferences: ${markdownValue(driverScopeCounts.scope_bounded_inferences || 0)}`,
    `- relevance: ${markdownValue(safeUi.relevance_status || safeTrace.relevance_status)}`,
    `- claim count: ${markdownValue(Array.isArray(analyticalClaims) ? analyticalClaims.length : 0)}`,
    "",
    "### Research Plan",
    `- question type: ${markdownValue(researchSummary.question_type || (safeUi.research_plan_used || {}).question_type)}`,
    `- user goal: ${markdownValue(researchSummary.user_goal || (safeUi.research_plan_used || {}).user_goal)}`,
    `- required answer parts: ${markdownValue(researchPlan.required_answer_parts || safeUi.required_answer_parts)}`,
    `- answer part gaps: ${markdownValue(researchPlan.evidence_gap_by_answer_part || safeUi.evidence_gap_by_answer_part)}`,
    `- missing but analyzable: ${markdownValue(researchPlan.missing_but_analyzable_answer_parts || safeUi.missing_but_analyzable_answer_parts)}`,
    `- fallback policy: ${markdownValue((researchPlan.used || safeUi.research_plan_used || {}).fallback_answer_policy)}`,
    `- relevance: ${markdownValue(safeUi.relevance_status || safeTrace.relevance_status)} / ${markdownValue(relevanceDecision.decision)} / ${markdownValue(relevanceDecision.route)}`,
    "",
    "### Evidence Health",
    `- status: ${markdownValue(analyticalReasoning.evidence_health || safeUi.evidence_health || "N/A")}`,
    `- tool error context: ${markdownValue(safeUi.tool_error_context || analyticalReasoning.tool_error_context || [])}`,
    `- missing but analyzable parts: ${markdownValue(safeUi.missing_but_analyzable_answer_parts || [])}`,
    `- missing and unanswerable parts: ${markdownValue(safeUi.missing_and_unanswerable_answer_parts || [])}`,
    "",
    "### Evidence Validation",
    ...bulletLines(evidenceValidationRecords, (row) => `${markdownValue(row.requirement_id)} (${markdownValue(row.evidence_type)}): returned=${markdownValue(row.tool_returned_count)}, validated=${markdownValue(row.validated_evidence_count)}, rejected=${markdownValue(row.rejected_evidence_reason)}, status=${markdownValue(row.status)}`),
    "",
    "### Evidence Scope",
    ...bulletLines(evidenceScopeRows, (row) => `${markdownValue(row.evidence_id || row.citation_id)}: claim_scope=${markdownValue(row.claim_scope)}, allowed_claim_strength=${markdownValue(row.allowed_claim_strength)}, driver_level=${markdownValue(row.driver_level)}, warning=${markdownValue(row.summary_scope_warning || "")}, reason=${markdownValue(row.scope_reason)}`),
    "",
    "### Scope Overclaim Check",
    `- status: ${markdownValue(scopeOverclaimCheck.status || (scopeOverclaimViolations.length ? "repairable" : "passed"))}`,
    `- checked claims: ${markdownValue(scopeOverclaimCheck.checked_claims)}`,
    `- violations: ${markdownValue(scopeOverclaimViolations.map((item) => item.code || item.type || item.message || "scope_overclaim"))}`,
    `- affected citations: ${markdownValue(scopeOverclaimViolations.flatMap((item) => Array.isArray(item.affected_citations) ? item.affected_citations : []))}`,
    `- evidence summary warnings: ${markdownValue(evidenceSummaryWarnings.map((item) => item.code || item.type || item.message || "evidence_summary_scope_overclaim"))}`,
    "",
    "### Legacy Dimension Status",
    ...bulletLines(dimensions, (dim) => `${markdownValue(dim.dimension_id)}: ${markdownValue(dim.status)}${Array.isArray(dim.limitations) && dim.limitations.length ? ` (${dim.limitations.join("; ")})` : ""}`),
    "",
    "### Missing / Partial Requirements",
    "Blocking Missing:",
    ...bulletLines(blockingMissing, (req) => `${markdownValue(req.requirement_id)} (${markdownValue(req.dimension)}, ${markdownValue(req.evidence_type)}): ${markdownValue(req.missing_reason)}`),
    "Optional Missing:",
    ...bulletLines(optionalMissing, (req) => `${markdownValue(req.requirement_id)} (${markdownValue(req.dimension)}, ${markdownValue(req.evidence_type)}): ${requirementScopeDescription(req)}; ${markdownValue(req.missing_reason)}`),
    "Partial:",
    ...bulletLines(partialRequirements, (req) => `${markdownValue(req.requirement_id)} (${markdownValue(req.dimension)}, ${markdownValue(req.evidence_type)}): ${markdownValue(req.missing_reason)}`),
    "",
    "### Tool Call Problems",
    ...bulletLines(toolProblems, (call) => `tool: ${markdownValue(call.tool_name)}; requirement: ${markdownValue(call.requirement_id)}; status: ${toolStatus(call)}; returned: ${markdownValue(call.returned_count)}; latency: ${markdownValue(call.latency_ms)}; backend: ${toolBackend(call)}; fallback: ${toolFallback(call)}; error: ${toolErrorText(call)}`),
    "",
    "### Evidence Issues",
    `- numeric evidence count: ${markdownValue(packetSummary.numeric_count)}`,
    `- text evidence count: ${markdownValue(packetSummary.text_count)}`,
    `- computed evidence count: ${markdownValue(packetSummary.computed_count)}`,
    `- citation count: ${citations.length}`,
    `- suspicious or unsupported citations: ${invalidCitations.length ? invalidCitations.map((item) => item.citation_id || item.evidence_id || "unknown").join(", ") : "N/A"}`,
    "",
    "### Semantic Parser",
    `- mode: ${markdownValue(safeUi.semantic_parser_mode || semanticParser.mode)}`,
    `- source: ${markdownValue(semanticParser.source)}`,
    `- ok: ${semanticParser.ok === true ? "yes" : "no"}`,
    `- error: ${markdownValue(semanticParser.error)}`,
    `- injected: ${semanticDisagreement.injected === true ? "yes" : "no"}`,
    `- proposed intent: ${markdownValue(semanticDisagreement.proposed_methodology_intent || (semanticParser.proposal || {}).methodology_intent)}`,
    `- rule intent: ${markdownValue(semanticDisagreement.rule_methodology_intent || safeUi.rule_methodology_intent)}`,
    `- final intent: ${markdownValue(semanticDisagreement.final_methodology_intent)}`,
    `- proposal warnings: ${Array.isArray(semanticParser.warnings) && semanticParser.warnings.length ? semanticParser.warnings.map((item) => `${item.field || "field"}:${item.reason || "warning"}`).join(", ") : "N/A"}`,
    "",
    "### Canonical Intent / Evidence Policy",
    `- final intent family: ${markdownValue(canonicalIntent.intent_family)}`,
    `- analysis scope: ${markdownValue(canonicalIntent.analysis_scope)}`,
    `- requested dimensions: ${markdownValue(canonicalIntent.requested_dimensions)}`,
    `- merge source: ${markdownValue(intentMergeDecision.source)}`,
    `- merge reason: ${markdownValue(intentMergeDecision.reason)}`,
    `- policy id: ${markdownValue(safeUi.evidence_policy_id || safeTrace.evidence_policy_id || evidencePolicy.policy_id)}`,
    `- core requirements: ${markdownValue(evidencePolicy.core_requirements)}`,
    `- optional context: ${markdownValue(evidencePolicy.optional_context_requirements)}`,
    `- diagnostic requirements: ${markdownValue(evidencePolicy.diagnostic_requirements)}`,
    "",
    "## Progress Events",
    ...bulletLines(progressEvents, progressEventDebugLine),
    "",
    "## 6. Evidence Matrix",
    "### Numeric Evidence",
    ...bulletLines(packet.numeric_evidence || [], (row) => `${markdownValue(row.evidence_id)}: ${markdownValue(row.ticker)}, ${markdownValue(row.metric)}, ${markdownValue(row.value)}, ${markdownValue(row.period_end || row.period)}, role=${markdownValue(row.role || row.evidence_role)}, quality=${markdownValue(row.quality_status)}, source_req=${markdownValue(row.source_requirement_id || row.requirement_id)}, provider=${markdownValue(row.source_provider)}, confidence=${markdownValue(row.confidence)}`),
    "",
    "### Text Evidence",
    ...bulletLines(packet.text_evidence || [], (row) => `${markdownValue(row.evidence_id)}: ${markdownValue(row.ticker)}, ${markdownValue(row.form_type)}, ${markdownValue(row.section)}, ${markdownValue(row.theme_name || row.claim || row.source_title)}, scope=${markdownValue(row.claim_scope)}, strength=${markdownValue(row.allowed_claim_strength)}, backend=${markdownValue(row.retrieval_backend || row.backend)}\n  snippet: ${truncateMarkdownText(row.supporting_snippet || row.text_snippet || row.snippet, 800)}`),
    "",
    "### Computed Evidence",
    ...bulletLines(packet.computed_metrics || [], (row) => `${markdownValue(row.evidence_id)}: ${markdownValue(row.metric)}, ${markdownValue(row.value)}, dependencies=${markdownValue(row.input_evidence_ids || row.formula)}`),
    "",
    "## 7. EvidencePlan / Requirement Ledger",
    ...bulletLines(requirements, (req) => [
      `requirement_id: ${markdownValue(req.requirement_id)}`,
      `dimension: ${markdownValue(req.dimension)}`,
      `type: ${markdownValue(req.evidence_type)}`,
      `role: ${markdownValue(req.evidence_role)}`,
      `required: ${markdownValue(req.required)}`,
      `scope: ${markdownValue(req.scope)}`,
      `status: ${markdownValue(req.status_label || req.status)}`,
      `tool: ${markdownValue(req.tool)}`,
      `returned: ${requirementReturned(req, callsByRequirement)}`,
      `missing_reason: ${markdownValue(req.missing_reason)}`,
      `backend / fallback: ${requirementBackend(req, callsByRequirement)} / ${requirementFallback(req, callsByRequirement)}`,
    ].join("; ")),
    "",
    "## 8. Tool Calls",
    ...bulletLines(preferredToolCalls, (call) => `tool: ${markdownValue(call.tool_name)}; requirement: ${markdownValue(call.requirement_id)}; status: ${toolStatus(call)}; returned: ${markdownValue(call.returned_count)}; latency: ${markdownValue(call.latency_ms)}; backend: ${toolBackend(call)}; fallback_after_timeout: ${call.fallback_after_timeout === true || toolFallback(call) === "fallback_after_timeout" ? "yes" : "no"}; fallback_after_error: ${call.fallback_after_error === true || toolFallback(call) === "fallback_after_error" ? "yes" : "no"}; error: ${toolErrorText(call)}`),
    "",
    "## 9. Contract / Validation",
    debugLine("Contract status", contract.status || safeUi.contract_status),
    debugLine("Contract decision", contract.decision || contractDecision.decision),
    debugLine("Draft release decision", draftReleaseDecision.decision),
    debugLine("Draft released", draftReleaseDecision.released),
    debugLine("Draft release warnings", draftReleaseDecision.warnings || []),
    debugLine("Repair attempts", contract.repair_attempts ?? safeUi.repair_attempts),
    debugLine("Violations", contract.violation_codes || []),
    debugLine("Warnings", contract.warning_codes || (Array.isArray(contract.warnings) ? contract.warnings.map((item) => item.code || item.message) : [])),
    debugLine("Missing caveats", [
      ...((contract.violation_codes || []).filter((code) => String(code).includes("caveat"))),
      ...((contract.warning_codes || []).filter((code) => String(code).includes("caveat"))),
    ]),
    debugLine("Unsupported claims", (contract.violation_codes || []).filter((code) => String(code).includes("unsupported"))),
    debugLine("Scope overclaim check", scopeOverclaimCheck.status || (scopeOverclaimViolations.length ? "repairable" : "passed")),
    debugLine("Scope overclaim violations", scopeOverclaimViolations.map((item) => item.code || item.type || item.message || "scope_overclaim")),
    debugLine("Final route", safeUi.final_route || contract.route),
    "",
    "## 10. Raw Trace Pointers",
    debugLine("Trace ID", traceId),
    debugLine("Local trace path if available", safeUi.local_trace_path || safeTrace.local_trace_path || (traceId ? `data/traces/${traceId}.json` : "")),
  ];
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function renderTraceUiModel(ui) {
  currentTraceUi = ui || {};
  if (traceAudit) {
    traceAudit.classList.remove("empty");
  }
  traceSummary.textContent = formatTraceUiSummary(currentTraceUi);
  traceJson.textContent = JSON.stringify(currentTraceUi, null, 2);
  setTraceTab(currentTraceTab || "workflow");
  const traceEvents = safeProgressEvents(currentTraceUi);
  if (traceEvents.length && (!latestAnswerTraceId || currentTraceUi.trace_id === latestAnswerTraceId || currentTraceUi.trace_id === progressPollingTraceId)) {
    renderProgressFromEvents(traceEvents);
  }
  if (currentTraceUi.trace_id && currentTraceUi.trace_id === latestAnswerTraceId && !progressStartedAt) {
    const policy = currentTraceUi.evidence_policy_id || ((currentTraceUi.evidence_policy || {}).policy_id) || "policy recorded";
    const contract = currentTraceUi.contract_status || ((currentTraceUi.contract || {}).status) || "not_run";
    const relevance = currentTraceUi.relevance_status || "not_run";
    const normalizedContract = String(contract || "").toLowerCase();
    const blocked = normalizedContract === "blocked";
    const warning = ["passed_with_warnings", "repaired", "warning"].includes(normalizedContract);
    setProgressBadge(blocked ? "Blocked" : (warning ? "Warnings" : "Complete"), blocked ? "bad" : (warning ? "warn" : "good"));
    setProgressPanel({
      mode: blocked ? "failed" : "complete",
      title: "Trace 已覆盖最终状态",
      narrative: `最终意图 ${((currentTraceUi.canonical_intent || {}).intent_family) || "unknown"}，证据策略 ${policy}，契约状态 ${contract}，相关性 ${relevance}。`,
      elapsedMs: 0,
      percent: 100,
      activeIndex: runProgressStages.length - 1,
    });
    pushProgressLog(`Trace final：policy=${policy}，contract=${contract}，relevance=${relevance}`);
  }
}

function traceClipboardText(summaryText) {
  const summary = String(summaryText || "").trim();
  if (!summary || summary === "No trace loaded." || summary === "Loading trace..." || summary === "Please input a trace ID.") {
    return "";
  }
  return summary;
}

function setCopyTraceStatus(message, state = "") {
  if (!copyTraceStatus) return;
  copyTraceStatus.textContent = message;
  copyTraceStatus.className = state ? `hint ${state}` : "hint";
}

function fallbackCopyText(text) {
  if (!hasDocument) return false;
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "readonly");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } finally {
    textarea.remove();
  }
  return ok;
}

async function copyTraceSummaryToClipboard() {
  const text = traceClipboardText(traceSummary ? traceSummary.textContent : "");
  if (!text) {
    setCopyTraceStatus("暂无可复制摘要。", "warn");
    return false;
  }
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text);
    } else if (!fallbackCopyText(text)) {
      throw new Error("Clipboard API unavailable.");
    }
    setCopyTraceStatus("已复制。", "success");
    return true;
  } catch (error) {
    if (fallbackCopyText(text)) {
      setCopyTraceStatus("已复制。", "success");
      return true;
    }
    setCopyTraceStatus("复制失败。", "error");
    return false;
  }
}

function currentDebugBundleTraceUi() {
  if (currentTraceUi && typeof currentTraceUi === "object" && Object.keys(currentTraceUi).length > 0) {
    return currentTraceUi;
  }
  const raw = traceJson ? String(traceJson.textContent || "").trim() : "";
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch (error) {
    return null;
  }
}

async function copyDebugBundleToClipboard() {
  const ui = currentDebugBundleTraceUi();
  if (!ui) {
    setCopyTraceStatus("Copy failed. Please copy manually.", "error");
    return false;
  }
  const text = buildDebugBundle({}, ui);
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text);
    } else if (!fallbackCopyText(text)) {
      throw new Error("Clipboard API unavailable.");
    }
    setCopyTraceStatus("Debug bundle copied", "success");
    return true;
  } catch (error) {
    if (fallbackCopyText(text)) {
      setCopyTraceStatus("Debug bundle copied", "success");
      return true;
    }
    setCopyTraceStatus("Copy failed. Please copy manually.", "error");
    return false;
  }
}

async function checkHealth() {
  setHealthBadge("Checking", "neutral");
  try {
    const data = await requestJson(makeUrl("/health"));
    if (data.status === "ok") {
      const profile = data.llm_provider ? ` · ${data.llm_provider}` : "";
      const draftTokens = data.analyst_draft_max_tokens ? ` · draft ${data.analyst_draft_max_tokens}` : "";
      setHealthBadge(`API OK${profile}${draftTokens}`, "good");
    } else {
      setHealthBadge(`Degraded: ${data.status}`, "warn");
    }
  } catch (error) {
    setHealthBadge("Offline", "bad");
  }
}

async function sendChat() {
  const query = queryInput.value.trim();
  if (!query) {
    chatStatus.textContent = "请输入研究问题。";
    return;
  }

  sendBtn.disabled = true;
  chatStatus.textContent = "分析运行中...";
  const clientTraceId = generateClientTraceId();
  latestAnswerTraceId = clientTraceId;
  traceIdText.textContent = clientTraceId;
  traceInput.value = clientTraceId;
  startRunProgress(query, clientTraceId);
  outputView.textContent = "正在生成结构化分析...";
  outputView.className = "output-view";
  answerText.textContent = "等待模型响应...";
  answerText.className = "answer-body";
  renderEvidenceFromOutput(null);

  try {
    const data = await requestJson(makeUrl("/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, client_trace_id: clientTraceId }),
    });

    const blockedPrimary = isBlockedContractResponse(data);
    const hasOutput = blockedPrimary ? renderBlockedPrimaryResponse(data) : renderOutput(data.output);
    const methodologyPrimary = !blockedPrimary && isMethodologyPrimaryOutput(data.output);
    if (!hasOutput) {
      renderAnswer(data.answer);
    } else {
      renderAnswer(data.answer);
    }
    setLegacyDebugMode(methodologyPrimary);
    taskType.textContent = data.task_type || "-";
    usedTools.textContent = Array.isArray(data.used_tools) && data.used_tools.length > 0
      ? data.used_tools.join(", ")
      : "-";
    traceIdText.textContent = data.trace_id || "-";
    latestAnswerTraceId = data.trace_id || "";

    if (data.trace_id) {
      traceInput.value = data.trace_id;
      await loadTrace({ force: true, expectedTraceId: data.trace_id });
    }
    if (hasOutput) {
      renderEvidenceFromOutput(data.output);
      renderCitations(data.citations || []);
    } else {
      renderEvidenceFromOutput(null);
      renderCitations(data.citations || []);
    }
    finishRunProgress(data);
    chatStatus.textContent = blockedPrimary ? "证据不足，未发布分析。" : "分析完成。";
  } catch (error) {
    outputView.textContent = error.message;
    outputView.className = "output-view error";
    answerText.textContent = error.message;
    answerText.className = "answer-body error";
    taskType.textContent = "-";
    usedTools.textContent = "-";
    traceIdText.textContent = "-";
    latestAnswerTraceId = "";
    renderEvidenceFromOutput(null);
    renderCitations([]);
    failRunProgress(error);
    chatStatus.textContent = "请求失败。";
  } finally {
    sendBtn.disabled = false;
  }
}

async function loadTrace({ force = false, expectedTraceId = "" } = {}) {
  const traceId = traceInput.value.trim();
  if (!traceId) {
    traceSummary.textContent = "Please input a trace ID.";
    setCopyTraceStatus("", "");
    return;
  }
  if (!force && traceId === lastLoadedTraceId && String(traceJson.textContent || "").trim()) {
    return;
  }

  traceBtn.disabled = true;
  if (debugBundleBtn) debugBundleBtn.disabled = true;
  setCopyTraceStatus("", "");
  traceSummary.textContent = "Loading trace...";
  traceJson.textContent = "";

  try {
    const trace = await requestJson(makeUrl(`/trace/${encodeURIComponent(traceId)}/ui`));
    const guard = shouldRenderTraceResponse(trace, traceId, expectedTraceId, latestAnswerTraceId);
    if (!guard.ok) {
      traceSummary.textContent = guard.reason;
      traceJson.textContent = "";
      setCopyTraceStatus("Trace 不匹配。", "error");
      return;
    }
    renderTraceUiModel(trace);
    lastLoadedTraceId = traceId;
  } catch (error) {
    traceSummary.textContent = error.message;
    traceJson.textContent = "";
    if (traceAudit) {
      traceAudit.classList.add("empty");
    }
    lastLoadedTraceId = "";
    setCopyTraceStatus("加载失败。", "error");
  } finally {
    traceBtn.disabled = false;
    if (debugBundleBtn) debugBundleBtn.disabled = false;
  }
}

function scheduleTraceLoad() {
  if (traceLoadTimer) {
    window.clearTimeout(traceLoadTimer);
  }
  setCopyTraceStatus("", "");
  const traceId = traceInput.value.trim();
  if (!traceId) {
    lastLoadedTraceId = "";
    currentTraceUi = null;
    traceSummary.textContent = "No trace loaded.";
    traceJson.textContent = "";
    if (traceAudit) {
      traceAudit.classList.add("empty");
    }
    return;
  }
  traceLoadTimer = window.setTimeout(() => {
    loadTrace({ force: false });
  }, 350);
}

function init() {
  setProgressBadge("Standby", "neutral");
  setProgressPanel({
    mode: "idle",
    title: "等待研究指令",
    narrative: "提交问题后，这里会显示当前分析正在推进到哪一步。",
    elapsedMs: 0,
    percent: 0,
    activeIndex: -1,
  });
  renderProgressLog();

  sendBtn.addEventListener("click", sendChat);
  traceBtn.addEventListener("click", copyTraceSummaryToClipboard);
  if (debugBundleBtn) {
    debugBundleBtn.addEventListener("click", copyDebugBundleToClipboard);
  }
  traceInput.addEventListener("input", scheduleTraceLoad);
  traceInput.addEventListener("change", () => loadTrace({ force: true }));
  document.querySelectorAll(".trace-tab").forEach((btn) => {
    btn.addEventListener("click", () => setTraceTab(btn.dataset.traceTab || "workflow"));
  });
  queryInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      sendChat();
    }
  });

  checkHealth();
}

if (hasDocument) {
  init();
}

export {
  buildMethodologyTraceSummary,
  buildBlockedPrimaryHtml,
  isBlockedContractResponse,
  formatTextEvidenceItem,
  formatLimitationDisplay,
  formatMethodologyTraceSummary,
  renderMethodologyComparison,
  renderMethodologySingleCompany,
  renderBlockedPrimaryResponse,
  renderRiskFocusedAnalysis,
  sanitizeTraceDisplayText,
  shouldRenderTraceResponse,
  buildDebugBundle,
  traceClipboardText,
  progressStageIndexForEvent,
  progressBadgeForEvents,
  safeProgressEvents,
};
