(function () {
  "use strict";

  const STORAGE = Object.freeze({
    session: "yher_demo_session_id",
    user: "yher_demo_user_id"
  });
  const PUBLIC_RIR_ZONES = new Set(["stem", "options", "feedback"]);
  const PHASE_ORDER = ["diagnostic", "learning", "held_out", "report"];
  const PHASE_COPY = Object.freeze({
    diagnostic: {title: "诊断", subtitle: "定位当前起点"},
    learning: {title: "学习", subtitle: "讲解与练习"},
    held_out: {title: "验证", subtitle: "未见题验证"},
    report: {title: "复盘", subtitle: "本次证据"}
  });
  const REPORT_OUTCOME_COPY = Object.freeze({
    verified: "验证通过",
    needs_reinforcement: "继续补强",
    partial: "本次学习已保存"
  });
  const PRIVATE_RESPONSE_KEY = /^(?:answer|(?:standard|final)_answers?|rubric|analysis|(?:item|family|aligned_item)_id)$/i;
  const PHASE_ALIASES = Object.freeze({
    diagnosis: "diagnostic",
    diagnose: "diagnostic",
    assessment: "diagnostic",
    warmup: "diagnostic",
    explanation: "learning",
    recommendation: "learning",
    practice: "learning",
    verification: "held_out",
    verify: "held_out",
    heldout: "held_out",
    complete: "report",
    completed: "report",
    finished: "report",
    summary: "report"
  });

  const state = {
    sessionId: localStorage.getItem(STORAGE.session) || "",
    userId: localStorage.getItem(STORAGE.user) || "",
    phase: "setup",
    assignment: null,
    queuedAssignment: null,
    draftAnswer: "",
    pendingSubmissionId: "",
    continueAction: null,
    retryAction: null,
    synthetic: false,
    busy: false
  };

  const elements = {};

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    [
      "app-main", "setup-view", "session-view", "report-view", "status-region",
      "loading-strip", "loading-label", "synthetic-marker", "setup-form", "setup-error",
      "user-id", "grade-select", "purpose-select", "node-select", "node-notice",
      "resume-panel", "resume-button", "discard-button", "start-button", "stage-title",
      "stage-subtitle", "phase-progress-label", "server-timing", "phase-progress",
      "pause-button",
      "learning-context", "assignment-panel", "assignment-meta", "question-content",
      "answer-form", "answer-fieldset", "answer-options", "free-answer-field",
      "free-answer-label", "free-answer", "answer-error", "submit-button",
      "checkpoint-panel", "checkpoint-title", "checkpoint-content", "continue-button",
      "degraded-panel", "degraded-message", "retry-button", "report-state",
      "report-outcome", "report-metrics", "belief-section", "belief-list", "report-empty",
      "report-error", "new-session-button"
    ].forEach(function (id) {
      elements[id] = document.getElementById(id);
    });

    if (!state.userId) state.userId = makeStudentId();
    elements["user-id"].value = state.userId;
    localStorage.setItem(STORAGE.user, state.userId);
    elements["resume-panel"].hidden = !state.sessionId;

    elements["setup-form"].addEventListener("submit", startSession);
    elements["resume-button"].addEventListener("click", resumeSession);
    elements["discard-button"].addEventListener("click", beginNewSession);
    elements["pause-button"].addEventListener("click", pauseSession);
    elements["answer-form"].addEventListener("submit", submitCurrentAnswer);
    elements["free-answer"].addEventListener("input", function (event) {
      state.draftAnswer = event.target.value;
      syncSubmitButton();
    });
    elements["continue-button"].addEventListener("click", function () {
      if (typeof state.continueAction === "function") state.continueAction();
    });
    elements["retry-button"].addEventListener("click", function () {
      if (typeof state.retryAction === "function") state.retryAction();
    });
    elements["new-session-button"].addEventListener("click", beginNewSession);

    loadNodes();
  }

  async function loadNodes() {
    try {
      const payload = await request("/api/demo/nodes", {method: "GET"});
      const nodes = normalizeNodes(payload);
      if (nodes.length) renderNodeOptions(nodes);
      elements["node-notice"].hidden = true;
    } catch (_error) {
      elements["node-notice"].textContent = "专题目录暂不可用，已保留氧化还原反应。";
      elements["node-notice"].hidden = false;
    }
  }

  function normalizeNodes(payload) {
    const raw = Array.isArray(payload)
      ? payload
      : Array.isArray(payload && payload.nodes)
        ? payload.nodes
        : Array.isArray(payload && payload.open_nodes)
          ? payload.open_nodes
          : payload && payload.open_nodes && typeof payload.open_nodes === "object"
            ? Object.keys(payload.open_nodes).map(function (name) { return {node: name}; })
            : [];
    const seen = new Set();
    const nodes = [];
    raw.forEach(function (entry) {
      const value = typeof entry === "string"
        ? entry
        : entry && (entry.node || entry.node_id || entry.id || entry.name || entry.label);
      const label = typeof entry === "object" && entry
        ? entry.label || entry.name || value
        : value;
      if (typeof value !== "string" || !value.trim() || seen.has(value.trim())) return;
      seen.add(value.trim());
      nodes.push({value: value.trim(), label: String(label || value).trim()});
    });
    nodes.sort(function (left, right) {
      if (left.value === "氧化还原反应") return -1;
      if (right.value === "氧化还原反应") return 1;
      return left.label.localeCompare(right.label, "zh-CN");
    });
    return nodes;
  }

  function renderNodeOptions(nodes) {
    const fragment = document.createDocumentFragment();
    nodes.forEach(function (node) {
      const option = document.createElement("option");
      option.value = node.value;
      option.textContent = node.label;
      if (node.value === "氧化还原反应") option.selected = true;
      fragment.appendChild(option);
    });
    elements["node-select"].replaceChildren(fragment);
  }

  async function startSession(event) {
    event.preventDefault();
    hideInlineError(elements["setup-error"]);
    if (!elements["setup-form"].reportValidity()) return;

    const budget = document.querySelector('input[name="budget_tier"]:checked');
    state.userId = elements["user-id"].value.trim();
    localStorage.setItem(STORAGE.user, state.userId);

    const body = {
      user_id: state.userId,
      node: elements["node-select"].value,
      budget_tier: budget ? budget.value : "30min",
      grade: elements["grade-select"].value,
      learning_purpose: elements["purpose-select"].value
    };

    setBusy(true, "正在建立学习会话");
    try {
      const payload = await request("/api/demo/sessions", {method: "POST", body});
      acceptSessionIdentity(payload);
      renderPayload(payload);
    } catch (error) {
      showInlineError(elements["setup-error"], messageForError(error));
      announce("建立会话失败");
    } finally {
      setBusy(false);
    }
  }

  async function resumeSession() {
    if (!state.sessionId) return;
    const path = `/api/demo/sessions/${encodeURIComponent(state.sessionId)}/resume`;
    setBusy(true, "正在恢复进度");
    try {
      const payload = await request(path, {method: "POST"});
      acceptSessionIdentity(payload);
      renderPayload(payload);
    } catch (error) {
      if (error && error.status === 400) {
        try {
          const payload = await request(`/api/demo/sessions/${encodeURIComponent(state.sessionId)}/next`, {
            method: "GET"
          });
          renderPayload(payload);
          return;
        } catch (nextError) {
          showInlineError(elements["setup-error"], messageForError(nextError));
        }
      } else {
        showInlineError(elements["setup-error"], messageForError(error));
      }
      announce("恢复进度失败");
    } finally {
      setBusy(false);
    }
  }

  async function pauseSession() {
    if (!state.sessionId) return;
    const path = `/api/demo/sessions/${state.sessionId}/pause`;
    setBusy(true, "正在保存进度");
    try {
      const payload = await request(path, {method: "POST"});
      assertPublicPayload(payload);
      elements["resume-panel"].hidden = false;
      showView("setup");
      announce("进度已保存");
    } catch (error) {
      showDegraded(messageForError(error), pauseSession);
    } finally {
      setBusy(false);
    }
  }

  function acceptSessionIdentity(payload) {
    if (!payload || typeof payload.session_id !== "string" || !payload.session_id.trim()) {
      throw new Error("missing_session");
    }
    state.sessionId = payload.session_id;
    localStorage.setItem(STORAGE.session, state.sessionId);
    elements["resume-panel"].hidden = false;
  }

  function renderPayload(payload) {
    assertPublicPayload(payload);
    syncSynthetic(payload);
    const phase = phaseFromServer(payload);
    if (phase) state.phase = phase;

    if (payload && payload.assignment) {
      renderAssignment(sanitizeAssignment(payload.assignment), payload);
      return;
    }
    if (payload && payload.assignment_id) {
      renderAssignment(sanitizeAssignment(payload), payload);
      return;
    }
    if (state.phase === "report" || isReportPayload(payload)) {
      if (isReportPayload(payload)) renderReport(payload.report || payload);
      else fetchReport();
      return;
    }
    if (payload && payload.budget_exhausted === true) {
      showCheckpoint(
        "本档时间已用完",
        "进度已由服务端保存。",
        "返回设置",
        function () { showView("setup"); }
      );
      return;
    }
    if (hasLearningContent(payload)) {
      renderLearningCheckpoint(payload);
      return;
    }
    showDegraded("服务端尚未返回下一步。", advanceSession);
  }

  function assertPublicPayload(payload) {
    walkObject(payload, function (key) {
      if (PRIVATE_RESPONSE_KEY.test(key)) throw new Error("unsafe_response");
    });
    if (payload && payload.assignment) assertPublicAssignment(payload.assignment);
  }

  function assertPublicAssignment(assignment) {
    if (!assignment || typeof assignment !== "object") throw new Error("invalid_assignment");
    if (typeof assignment.assignment_id !== "string" || !assignment.assignment_id) {
      throw new Error("invalid_assignment");
    }
    walkObject(assignment, function (key, value, parentKey) {
      if (PRIVATE_RESPONSE_KEY.test(key)) throw new Error("unsafe_assignment");
      if (parentKey === "zones" && !PUBLIC_RIR_ZONES.has(key)) {
        throw new Error("unsafe_rir_zone");
      }
      if (key === "zones" && Array.isArray(value)) {
        value.forEach(function (zone) {
          if (!PUBLIC_RIR_ZONES.has(String(zone).toLowerCase())) {
            throw new Error("unsafe_rir_zone");
          }
        });
      }
    });
  }

  function walkObject(value, visitor, parentKey) {
    if (Array.isArray(value)) {
      value.forEach(function (child) { walkObject(child, visitor, parentKey); });
      return;
    }
    if (!value || typeof value !== "object") return;
    Object.entries(value).forEach(function (entry) {
      visitor(entry[0], entry[1], parentKey || "");
      walkObject(entry[1], visitor, entry[0]);
    });
  }

  function sanitizeAssignment(raw) {
    assertPublicAssignment(raw);
    const question = raw.question && typeof raw.question === "object" ? raw.question : {};
    return {
      assignment_id: raw.assignment_id,
      phase: raw.phase,
      response_kind: raw.response_kind || raw.kind || question.response_kind || question.kind || "",
      node_id: raw.node_id || "",
      role: raw.role || "",
      source_label: raw.source_label || question.source_label || "",
      progress: raw.progress || null,
      timing: raw.timing || null,
      feedback: raw.feedback || null,
      recommendation: raw.recommendation || raw.recommendations || null,
      explanation: raw.explanation || null,
      question: {
        stem_text: question.stem_text || "",
        prompt: question.prompt || "",
        stem_blocks: question.stem_blocks || null,
        rir: sanitizeRir(question.rir),
        options: question.options || raw.options || null
      }
    };
  }

  function sanitizeRir(rir) {
    if (!rir || typeof rir !== "object") return null;
    const zones = rir.zones && typeof rir.zones === "object" ? rir.zones : {};
    const safeZones = {};
    Object.keys(zones).forEach(function (name) {
      const normalized = String(name).toLowerCase();
      if (!PUBLIC_RIR_ZONES.has(normalized)) throw new Error("unsafe_rir_zone");
      safeZones[normalized] = zones[name];
    });
    return {zones: safeZones};
  }

  function phaseFromServer(payload) {
    const candidates = [
      payload && payload.assignment && payload.assignment.phase,
      payload && payload.phase,
      payload && payload.state && payload.state.phase,
      payload && payload.next_action && payload.next_action.phase,
      payload && typeof payload.status === "string" ? payload.status : null
    ];
    for (const candidate of candidates) {
      if (typeof candidate !== "string") continue;
      const normalized = candidate.trim().toLowerCase().replace(/[\s-]+/g, "_");
      if (PHASE_ORDER.includes(normalized)) return normalized;
      if (PHASE_ALIASES[normalized]) return PHASE_ALIASES[normalized];
    }
    return null;
  }

  function renderAssignment(assignment, envelope) {
    const serverPhase = phaseFromServer({assignment}) || phaseFromServer(envelope);
    if (!serverPhase) {
      showDegraded("服务端未提供阶段信息。", advanceSession);
      return;
    }
    state.phase = serverPhase;
    state.assignment = assignment;
    state.queuedAssignment = null;
    state.draftAnswer = "";
    state.pendingSubmissionId = "";

    syncSynthetic(envelope || assignment);
    showView("session");
    updateStageUi(serverPhase, assignment.progress || (envelope && envelope.progress));
    renderTiming(assignment.timing || (envelope && envelope.timing));
    hideSessionPanels();
    renderLearningContext(assignment);

    if (!hasQuestion(assignment.question)) {
      renderContentAssignment(assignment);
      return;
    }

    elements["assignment-panel"].hidden = false;
    renderAssignmentMeta(assignment);
    renderQuestion(assignment.question);
    renderAnswerControls(assignment);
    hideInlineError(elements["answer-error"]);
    announce(`${PHASE_COPY[serverPhase].title}阶段，题目已加载`);
  }

  function hasQuestion(question) {
    if (!question || typeof question !== "object") return false;
    return Boolean(
      String(question.stem_text || question.prompt || "").trim()
      || question.stem_blocks
      || (question.rir && question.rir.zones && question.rir.zones.stem)
    );
  }

  function renderAssignmentMeta(assignment) {
    const values = [];
    if (assignment.node_id) values.push(assignment.node_id);
    if (assignment.source_label) values.push(assignment.source_label);
    if (assignment.role && !values.includes(assignment.role)) values.push(roleLabel(assignment.role));
    const fragment = document.createDocumentFragment();
    values.filter(Boolean).forEach(function (value) {
      const span = document.createElement("span");
      span.textContent = String(value);
      fragment.appendChild(span);
    });
    elements["assignment-meta"].replaceChildren(fragment);
    elements["assignment-meta"].hidden = !values.length;
  }

  function roleLabel(role) {
    const labels = {
      warmup: "热身",
      diagnostic: "诊断题",
      practice: "练习题",
      held_out: "验证题",
      explanation: "讲解",
      recommendation: "学习资源"
    };
    return labels[String(role).toLowerCase()] || String(role);
  }

  function renderQuestion(question) {
    const target = elements["question-content"];
    target.replaceChildren();
    const stemZone = question.rir && question.rir.zones && question.rir.zones.stem;
    if (stemZone) {
      renderRirZone(stemZone, target);
      return;
    }
    if (question.stem_blocks) {
      renderRirZone(stemBlocksToParagraphs(question.stem_blocks), target);
      return;
    }
    const paragraph = document.createElement("p");
    paragraph.textContent = String(question.stem_text || question.prompt || "题目暂不可用");
    target.appendChild(paragraph);
  }

  function stemBlocksToParagraphs(blocks) {
    const blockList = Array.isArray(blocks) ? blocks : [blocks];
    return blockList.map(function (block) {
      const nodes = block && Array.isArray(block.para) ? block.para : Array.isArray(block) ? block : [block];
      return nodes.map(toPublicRirNode).filter(Boolean);
    }).filter(function (paragraph) { return paragraph.length; });
  }

  function toPublicRirNode(node) {
    if (typeof node === "string") return {kind: "text", text: node};
    if (!node || typeof node !== "object") return null;
    const kind = String(node.kind || node.type || "").toLowerCase();
    if (kind === "text") return {kind: "text", text: String(node.text || "")};
    if (kind === "latex") {
      return {kind: "latex", latex: String(node.latex || node.text || ""), source: "public_stem"};
    }
    if (kind === "image") {
      return {
        kind: "image",
        url: safeMediaUrl(node.url || node.src),
        alt: String(node.alt || "题目图片"),
        inline: node.inline === true,
        w: finiteNumber(node.w)
      };
    }
    if (kind === "placeholder") {
      return {kind: "placeholder", reason: String(node.reason || "")};
    }
    if (kind === "table") {
      const rows = Array.isArray(node.rows) ? node.rows : [];
      return {
        kind: "table",
        rows: rows.map(function (row) {
          return (Array.isArray(row) ? row : []).map(function (cell) {
            const children = Array.isArray(cell) ? cell : [cell];
            return children.map(toPublicRirNode).filter(Boolean);
          });
        })
      };
    }
    return {kind: "text", text: visibleText(node)};
  }

  function safeMediaUrl(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    try {
      const url = new URL(raw, location.origin);
      if (url.origin === location.origin) return url.href;
    } catch (_error) {
      return "";
    }
    return "";
  }

  function renderRirZone(paragraphs, target) {
    if (window.YHerRirRenderer && typeof window.YHerRirRenderer.renderZone === "function") {
      window.YHerRirRenderer.renderZone(Array.isArray(paragraphs) ? paragraphs : [], target, {
        disableReports: true
      });
      return;
    }
    const fallback = document.createElement("p");
    fallback.textContent = visibleText(paragraphs);
    target.appendChild(fallback);
  }

  function visibleText(value) {
    if (typeof value === "string" || typeof value === "number") return String(value);
    if (Array.isArray(value)) return value.map(visibleText).filter(Boolean).join(" ");
    if (!value || typeof value !== "object") return "";
    return [value.text, value.latex, value.label, value.value]
      .map(visibleText)
      .filter(Boolean)
      .join(" ");
  }

  function renderAnswerControls(assignment) {
    const options = normalizeOptions(assignment.question.options);
    elements["answer-options"].replaceChildren();
    elements["free-answer"].value = "";
    elements["submit-button"].disabled = true;

    if (options.length) {
      elements["answer-options"].hidden = false;
      elements["free-answer-field"].hidden = true;
      const fragment = document.createDocumentFragment();
      options.forEach(function (option) {
        const label = document.createElement("label");
        label.className = "option-label";
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "answer-option";
        radio.value = option.key;
        radio.addEventListener("change", function () {
          state.draftAnswer = radio.value;
          syncSubmitButton();
        });
        const key = document.createElement("span");
        key.className = "option-key";
        key.textContent = option.key;
        const copy = document.createElement("span");
        copy.className = "option-copy";
        appendPublicContent(copy, option.value);
        label.append(radio, key, copy);
        fragment.appendChild(label);
      });
      elements["answer-options"].appendChild(fragment);
    } else {
      elements["answer-options"].hidden = true;
      elements["free-answer-field"].hidden = false;
      const kind = String(assignment.response_kind || "").toLowerCase();
      elements["free-answer-label"].textContent = kind === "numeric" ? "数值与单位" : "你的作答";
    }
  }

  function normalizeOptions(options) {
    if (Array.isArray(options)) {
      return options.map(function (entry, index) {
        if (entry && typeof entry === "object") {
          return {
            key: String(entry.key || entry.id || String.fromCharCode(65 + index)),
            value: entry.label || entry.text || entry.value || ""
          };
        }
        return {key: String.fromCharCode(65 + index), value: entry};
      });
    }
    if (options && typeof options === "object") {
      return Object.entries(options).map(function (entry) {
        return {key: String(entry[0]), value: entry[1]};
      });
    }
    return [];
  }

  function appendPublicContent(target, value) {
    if (typeof value === "string" || typeof value === "number") {
      target.textContent = String(value);
      return;
    }
    const paragraphs = stemBlocksToParagraphs(value);
    if (paragraphs.length) renderRirZone(paragraphs, target);
    else target.textContent = visibleText(value);
  }

  function syncSubmitButton() {
    elements["submit-button"].disabled = state.busy || !state.draftAnswer.trim();
  }

  async function submitCurrentAnswer(event) {
    if (event) event.preventDefault();
    if (!state.assignment || !state.sessionId || !state.draftAnswer.trim()) return;
    hideInlineError(elements["answer-error"]);
    const submissionId = state.pendingSubmissionId || makeOpaqueId("submission");
    state.pendingSubmissionId = submissionId;
    const path = `/api/demo/sessions/${state.sessionId}/submit`;

    setBusy(true, "正在提交作答");
    try {
      const result = await request(path, {
        method: "POST",
        body: {
          assignment_id: state.assignment.assignment_id,
          submission_id: submissionId,
          answer: state.draftAnswer
        }
      });
      state.pendingSubmissionId = "";
      renderSubmissionResult(result);
    } catch (error) {
      showInlineError(elements["answer-error"], messageForError(error));
      announce("提交失败，答案已保留");
    } finally {
      setBusy(false);
      syncSubmitButton();
    }
  }

  function renderSubmissionResult(result) {
    assertPublicPayload(result);
    syncSynthetic(result);
    const submittedPhase = state.phase;
    const phase = submittedPhase;
    const returnedPhase = phaseFromServer(result) || submittedPhase;
    state.phase = returnedPhase;
    state.queuedAssignment = result && result.assignment
      ? sanitizeAssignment(result.assignment)
      : null;

    if (submittedPhase === "held_out" && (returnedPhase === "report" || isReportPayload(result))) {
      if (isReportPayload(result)) renderReport(result.report || result);
      else fetchReport();
      return;
    }

    if (phase === "diagnostic" || phase === "held_out") {
      renderNeutralSubmission(result, phase);
      return;
    }

    if (canRevealCorrectness(submittedPhase)) {
      renderPracticeFeedback(result);
      return;
    }

    showDegraded("服务端返回了无法识别的阶段。", advanceSession);
  }

  function canRevealCorrectness(phase) {
    return phase === "learning";
  }

  function renderNeutralSubmission(_result, phase) {
    const title = phase === "held_out" ? "验证作答已记录" : "作答已记录";
    showCheckpoint(title, "", "继续", continueAfterSubmission);
  }

  function renderPracticeFeedback(result) {
    hideSessionPanels();
    showView("session");
    elements["checkpoint-panel"].className = "checkpoint-panel";
    if (typeof result.is_correct === "boolean") {
      elements["checkpoint-panel"].classList.add(result.is_correct ? "success" : "attention");
      elements["checkpoint-title"].textContent = result.is_correct ? "本题通过" : "这一步还需补强";
    } else {
      elements["checkpoint-title"].textContent = "作答已记录";
    }
    elements["checkpoint-content"].replaceChildren();
    appendFeedback(elements["checkpoint-content"], result.feedback || result.summary);
    elements["continue-button"].textContent = "继续";
    state.continueAction = continueAfterSubmission;
    elements["checkpoint-panel"].hidden = false;
    focusCurrentView();
  }

  function continueAfterSubmission() {
    if (state.queuedAssignment) {
      const queued = state.queuedAssignment;
      state.queuedAssignment = null;
      renderAssignment(queued, {phase: queued.phase, synthetic: state.synthetic});
      return;
    }
    if (state.phase === "report") fetchReport();
    else advanceSession();
  }

  async function advanceSession() {
    if (!state.sessionId) {
      showView("setup");
      return;
    }
    const path = `/api/demo/sessions/${state.sessionId}/next`;
    setBusy(true, "正在获取下一步");
    try {
      const payload = await request(path, {method: "GET"});
      renderPayload(payload);
    } catch (error) {
      showDegraded(messageForError(error), advanceSession);
    } finally {
      setBusy(false);
    }
  }

  function renderLearningContext(source) {
    elements["learning-context"].replaceChildren();
    if (state.phase !== "learning" || !hasLearningContent(source)) {
      elements["learning-context"].hidden = true;
      return;
    }
    appendLearningContent(elements["learning-context"], source);
    elements["learning-context"].hidden = !elements["learning-context"].childNodes.length;
  }

  function renderContentAssignment(assignment) {
    hideSessionPanels();
    showView("session");
    renderLearningContext(assignment);
    showCheckpoint("本段完成", "", "继续", advanceSession);
  }

  function renderLearningCheckpoint(payload) {
    showView("session");
    updateStageUi(state.phase, payload.progress);
    renderTiming(payload.timing);
    hideSessionPanels();
    elements["learning-context"].replaceChildren();
    appendLearningContent(elements["learning-context"], payload);
    elements["learning-context"].hidden = false;
    showCheckpoint("准备好后继续", "", "继续", advanceSession, true);
  }

  function hasLearningContent(source) {
    if (!source || typeof source !== "object") return false;
    return Boolean(
      source.feedback || source.summary || source.explanation
      || source.recommendation || source.recommendations
    );
  }

  function appendLearningContent(target, source) {
    const title = firstString(source.title, source.heading, source.feedback && source.feedback.title);
    if (title) {
      const heading = document.createElement("h3");
      heading.textContent = title;
      target.appendChild(heading);
    }
    appendFeedback(target, source.explanation || source.feedback || source.summary);
    const recommendations = Array.isArray(source.recommendations)
      ? source.recommendations
      : Array.isArray(source.recommendation)
        ? source.recommendation
        : source.recommendation
          ? [source.recommendation]
          : [];
    recommendations.forEach(function (recommendation) {
      if (!recommendation || typeof recommendation !== "object") return;
      appendFeedback(target, recommendation);
      const url = safeLearningUrl(recommendation.url);
      if (url) {
        const link = document.createElement("a");
        link.className = "learning-link";
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "观看讲解";
        target.appendChild(link);
      }
    });
  }

  function appendFeedback(target, feedback) {
    if (typeof feedback === "string") {
      appendParagraph(target, feedback);
      return;
    }
    if (Array.isArray(feedback)) {
      feedback.forEach(function (entry) { appendFeedback(target, entry); });
      return;
    }
    if (!feedback || typeof feedback !== "object") return;
    [
      "message", "summary", "explanation", "key_point", "what_you_learn",
      "next_step", "source_label"
    ].forEach(function (key) {
      if (typeof feedback[key] === "string" && feedback[key].trim()) {
        appendParagraph(target, feedback[key]);
      }
    });
  }

  function appendParagraph(target, text) {
    const paragraph = document.createElement("p");
    paragraph.textContent = String(text);
    target.appendChild(paragraph);
  }

  function safeLearningUrl(value) {
    if (typeof value !== "string" || !value.trim()) return "";
    try {
      const url = new URL(value, location.origin);
      const allowed = url.protocol === "https:"
        && (url.hostname === "www.bilibili.com" || url.hostname === "bilibili.com" || url.hostname === "b23.tv");
      return allowed ? url.href : "";
    } catch (_error) {
      return "";
    }
  }

  async function fetchReport() {
    if (!state.sessionId) return;
    const path = `/api/demo/sessions/${state.sessionId}/report`;
    setBusy(true, "正在生成复盘");
    try {
      const payload = await request(path, {method: "GET"});
      assertPublicPayload(payload);
      syncSynthetic(payload);
      renderReport(payload.report || payload);
    } catch (error) {
      showDegraded(messageForError(error), fetchReport);
    } finally {
      setBusy(false);
    }
  }

  function isReportPayload(payload) {
    const value = payload && payload.report && typeof payload.report === "object"
      ? payload.report
      : payload;
    return Boolean(value && typeof value === "object" && (
      Object.prototype.hasOwnProperty.call(value, "outcome")
      || Object.prototype.hasOwnProperty.call(value, "state_label")
      || Object.prototype.hasOwnProperty.call(value, "mastery_probability")
      || Object.prototype.hasOwnProperty.call(value, "evidence_count")
      || Object.prototype.hasOwnProperty.call(value, "belief")
    ));
  }

  function renderReport(report) {
    state.phase = "report";
    state.assignment = null;
    state.queuedAssignment = null;
    syncSynthetic(report);
    showView("report");
    hideInlineError(elements["report-error"]);

    const stateLabel = firstString(report.state_label, report.status);
    elements["report-state"].textContent = stateLabel || "";
    const outcome = REPORT_OUTCOME_COPY[report.outcome] || "";
    elements["report-outcome"].textContent = firstString(outcome, stateLabel, "本次学习已记录");

    const metrics = [];
    const mastery = finiteNumber(report.mastery_probability);
    if (mastery !== null) metrics.push(["掌握概率", formatPercent(mastery)]);
    const evidence = finiteNumber(report.evidence_count);
    if (evidence !== null) metrics.push(["有效证据", `${Math.max(0, Math.round(evidence))} 条`]);
    if (report.as_of) metrics.push(["画像时间", formatDateTime(report.as_of)]);
    if (report.review_due_at) metrics.push(["建议复习", formatDateTime(report.review_due_at)]);
    renderMetrics(metrics);
    const hasBelief = renderBelief(report.belief);

    elements["report-empty"].hidden = Boolean(metrics.length || hasBelief);
    state.sessionId = "";
    localStorage.removeItem(STORAGE.session);
    elements["resume-panel"].hidden = true;
    announce("复盘已生成");
  }

  function renderMetrics(metrics) {
    const fragment = document.createDocumentFragment();
    metrics.forEach(function (entry) {
      const term = document.createElement("dt");
      term.textContent = entry[0];
      const detail = document.createElement("dd");
      detail.textContent = entry[1];
      fragment.append(term, detail);
    });
    elements["report-metrics"].replaceChildren(fragment);
  }

  function renderBelief(rawBelief) {
    const belief = normalizeBelief(rawBelief);
    elements["belief-list"].replaceChildren();
    if (!belief.length) {
      elements["belief-section"].hidden = true;
      return false;
    }
    const labels = {M: "已掌握", P: "步骤不稳", C: "概念混淆", U: "证据不足"};
    const fragment = document.createDocumentFragment();
    belief.forEach(function (entry) {
      const row = document.createElement("div");
      row.className = "belief-row";
      const label = document.createElement("span");
      label.textContent = labels[entry.key] || entry.key;
      const bar = document.createElement("progress");
      bar.max = 1;
      bar.value = entry.value;
      bar.setAttribute("aria-label", label.textContent);
      const output = document.createElement("output");
      output.textContent = formatPercent(entry.value);
      row.append(label, bar, output);
      fragment.appendChild(row);
    });
    elements["belief-list"].appendChild(fragment);
    elements["belief-section"].hidden = false;
    return true;
  }

  function normalizeBelief(value) {
    if (Array.isArray(value)) {
      return ["M", "P", "C", "U"].map(function (key, index) {
        return {key, value: finiteNumber(value[index])};
      }).filter(function (entry) { return entry.value !== null; });
    }
    if (!value || typeof value !== "object") return [];
    const aliases = {
      M: ["M", "m", "mastered"],
      P: ["P", "p", "procedural"],
      C: ["C", "c", "confused"],
      U: ["U", "u", "unknown"]
    };
    return Object.entries(aliases).map(function (entry) {
      const found = entry[1].find(function (key) {
        return Object.prototype.hasOwnProperty.call(value, key);
      });
      return {key: entry[0], value: found ? finiteNumber(value[found]) : null};
    }).filter(function (entry) { return entry.value !== null; });
  }

  function updateStageUi(phase, progress) {
    const activeIndex = PHASE_ORDER.indexOf(phase);
    document.querySelectorAll("[data-stage]").forEach(function (entry) {
      const index = PHASE_ORDER.indexOf(entry.dataset.stage);
      entry.classList.toggle("complete", index >= 0 && index < activeIndex);
      entry.classList.toggle("current", entry.dataset.stage === phase);
      if (entry.dataset.stage === phase) entry.setAttribute("aria-current", "step");
      else entry.removeAttribute("aria-current");
    });
    const copy = PHASE_COPY[phase] || PHASE_COPY.diagnostic;
    elements["stage-title"].textContent = copy.title;
    elements["stage-subtitle"].textContent = copy.subtitle;
    renderProgress(progress);
  }

  function renderProgress(progress) {
    let current = null;
    let total = null;
    let label = "";
    if (typeof progress === "number") {
      current = Math.min(1, Math.max(0, progress));
      total = 1;
    } else if (progress && typeof progress === "object") {
      current = finiteNumber(progress.current ?? progress.completed ?? progress.answered ?? progress.index);
      total = finiteNumber(progress.total);
      label = firstString(progress.label, progress.text);
    }
    if (current !== null && total !== null && total > 0) {
      elements["phase-progress"].max = total;
      elements["phase-progress"].value = Math.min(total, Math.max(0, current));
      elements["phase-progress"].hidden = false;
      elements["phase-progress-label"].textContent = label || `${Math.round(current)} / ${Math.round(total)}`;
    } else {
      elements["phase-progress"].hidden = true;
      elements["phase-progress-label"].textContent = label;
    }
  }

  function renderTiming(timing) {
    if (!timing || typeof timing !== "object") {
      elements["server-timing"].textContent = "";
      return;
    }
    const parts = [];
    const elapsedSeconds = finiteNumber(timing.elapsed_seconds ?? timing.elapsed_sec);
    const remainingSeconds = finiteNumber(timing.remaining_seconds ?? timing.remaining_sec);
    const elapsedMinutes = finiteNumber(timing.elapsed_minutes);
    const remainingMinutes = finiteNumber(timing.remaining_minutes);
    const budgetMinutes = finiteNumber(timing.budget_minutes);
    const expected = finiteNumber(timing.expected_minutes ?? timing.phase_budget_minutes);
    if (elapsedSeconds !== null) parts.push(`已用 ${formatDuration(elapsedSeconds)}`);
    else if (elapsedMinutes !== null) parts.push(`已用 ${formatDuration(elapsedMinutes * 60)}`);
    if (remainingSeconds !== null) parts.push(`剩余 ${formatDuration(remainingSeconds)}`);
    else if (remainingMinutes !== null) parts.push(`剩余 ${formatDuration(remainingMinutes * 60)}`);
    else if (expected !== null) parts.push(`本阶段约 ${formatDuration(expected * 60)}`);
    else if (budgetMinutes !== null) parts.push(`本档 ${formatDuration(budgetMinutes * 60)}`);
    elements["server-timing"].textContent = parts.join(" · ");
  }

  function showCheckpoint(title, message, buttonLabel, action, preserveLearning) {
    if (!preserveLearning) hideSessionPanels();
    showView("session");
    elements["checkpoint-panel"].className = "checkpoint-panel";
    elements["checkpoint-title"].textContent = title;
    elements["checkpoint-content"].replaceChildren();
    if (message) appendParagraph(elements["checkpoint-content"], message);
    elements["continue-button"].textContent = buttonLabel;
    state.continueAction = action;
    elements["checkpoint-panel"].hidden = false;
    focusCurrentView();
  }

  function showDegraded(message, retryAction) {
    showView("session");
    hideSessionPanels();
    elements["degraded-message"].textContent = message || "请求未完成，请重试。";
    state.retryAction = retryAction;
    elements["retry-button"].hidden = typeof retryAction !== "function";
    elements["degraded-panel"].hidden = false;
    announce("当前处于降级状态");
    focusCurrentView();
  }

  function hideSessionPanels() {
    elements["assignment-panel"].hidden = true;
    elements["checkpoint-panel"].hidden = true;
    elements["degraded-panel"].hidden = true;
    elements["learning-context"].hidden = true;
  }

  function beginNewSession() {
    state.sessionId = "";
    state.phase = "setup";
    state.assignment = null;
    state.queuedAssignment = null;
    state.draftAnswer = "";
    state.pendingSubmissionId = "";
    state.synthetic = false;
    localStorage.removeItem(STORAGE.session);
    elements["resume-panel"].hidden = true;
    setSyntheticMarker(false);
    hideInlineError(elements["setup-error"]);
    showView("setup");
  }

  function showView(name) {
    const map = {
      setup: elements["setup-view"],
      session: elements["session-view"],
      report: elements["report-view"]
    };
    Object.values(map).forEach(function (view) { view.hidden = true; });
    if (map[name]) map[name].hidden = false;
    window.scrollTo({top: 0, left: 0, behavior: "auto"});
    focusCurrentView();
  }

  function focusCurrentView() {
    window.requestAnimationFrame(function () {
      elements["app-main"].focus({preventScroll: true});
    });
  }

  function syncSynthetic(payload) {
    const enabled = Boolean(payload && (
      payload.synthetic === true
      || (payload.assignment && payload.assignment.synthetic === true)
      || (payload.report && payload.report.synthetic === true)
    ));
    if (enabled) state.synthetic = true;
    setSyntheticMarker(state.synthetic);
  }

  function setSyntheticMarker(enabled) {
    elements["synthetic-marker"].hidden = enabled !== true;
  }

  function setBusy(busy, label) {
    state.busy = busy;
    elements["loading-strip"].hidden = !busy;
    elements["loading-strip"].setAttribute("aria-hidden", busy ? "false" : "true");
    elements["loading-label"].textContent = label || "加载中";
    document.body.setAttribute("aria-busy", busy ? "true" : "false");
    ["start-button", "resume-button", "pause-button", "continue-button", "retry-button"].forEach(function (id) {
      elements[id].disabled = busy;
    });
    syncSubmitButton();
    if (busy) announce(label || "加载中");
  }

  async function request(path, options) {
    if (typeof path !== "string" || !path.startsWith("/api/demo/")) {
      throw new Error("invalid_api_path");
    }
    const config = options || {};
    const controller = new AbortController();
    const timer = window.setTimeout(function () { controller.abort(); }, 25000);
    const headers = {Accept: "application/json"};
    const init = {
      method: config.method || "GET",
      headers,
      credentials: "same-origin",
      signal: controller.signal
    };
    if (config.body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(config.body);
    }
    try {
      const response = await fetch(path, init);
      const text = await response.text();
      let payload = {};
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch (_error) {
          const invalid = new Error("invalid_json");
          invalid.status = response.status;
          throw invalid;
        }
      }
      if (!response.ok) {
        const failure = new Error("request_failed");
        failure.status = response.status;
        throw failure;
      }
      return payload;
    } catch (error) {
      if (error && error.name === "AbortError") {
        const timeout = new Error("timeout");
        timeout.status = 408;
        throw timeout;
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function messageForError(error) {
    const status = error && error.status;
    if (status === 404) return "这段进度暂时不可用。";
    if (status === 409) return "会话状态已变化，请重新获取。";
    if (status === 422) return "提交内容不完整，请检查后重试。";
    if (status === 429) return "请求较多，请稍后重试。";
    if (status === 408) return "请求超时，答案和进度仍保留在当前页面。";
    return "连接未完成，请重试。";
  }

  function showInlineError(element, message) {
    element.textContent = message;
    element.hidden = false;
  }

  function hideInlineError(element) {
    element.textContent = "";
    element.hidden = true;
  }

  function announce(message) {
    elements["status-region"].textContent = "";
    window.requestAnimationFrame(function () {
      elements["status-region"].textContent = message;
    });
  }

  function finiteNumber(value) {
    const number = typeof value === "number" ? value : Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatPercent(value) {
    const normalized = value > 1 ? value / 100 : value;
    return `${Math.round(Math.min(1, Math.max(0, normalized)) * 100)}%`;
  }

  function formatDuration(seconds) {
    const safe = Math.max(0, Math.round(seconds));
    if (safe < 60) return `${safe} 秒`;
    const minutes = Math.floor(safe / 60);
    const remainder = safe % 60;
    return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
  }

  function formatDateTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    }).format(date);
  }

  function firstString() {
    for (let index = 0; index < arguments.length; index += 1) {
      const value = arguments[index];
      if (typeof value === "string" && value.trim()) return value.trim();
    }
    return "";
  }

  function makeStudentId() {
    return `student_${makeOpaqueId("profile").slice(-12)}`;
  }

  function makeOpaqueId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}-${window.crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }
})();
