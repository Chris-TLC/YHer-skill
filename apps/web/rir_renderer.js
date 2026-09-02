(function () {
  "use strict";

  const NODE_KINDS = new Set(["text", "latex", "image", "table", "placeholder"]);
  let styleInjected = false;

  function injectStyles() {
    if (styleInjected || typeof document === "undefined") return;
    styleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
.rir-zone{display:flex;flex-direction:column;gap:8px;min-width:0}
.rir-para{font-size:17px;line-height:1.6;word-break:break-word;min-width:0}
.rir-para.rir-option{padding-left:1.4em;text-indent:-1.4em}
.rir-text{white-space:pre-wrap}
.rir-latex{display:inline;vertical-align:baseline}
.rir-latex-scroll{display:inline-block;max-width:100%;overflow-x:auto;overflow-y:hidden;vertical-align:middle}
.rir-latex .katex{font-size:1.02em}
.rir-image{display:block;height:auto;margin:10px auto;border-radius:6px}
.rir-image.rir-img-inline{display:inline-block;height:1.35em;width:auto;margin:0 1px;vertical-align:-0.28em;border-radius:0}
.rir-placeholder{display:flex;align-items:center;justify-content:center;min-height:72px;margin:10px 0;border:1px dashed #c7c7cc;border-radius:8px;background:#f7f7fa;color:#8e8e93;font-size:13px}
.rir-table-wrap{width:100%;overflow-x:auto;margin:10px 0}
.rir-table{border-collapse:collapse;width:max-content;max-width:100%;font-size:15px;line-height:1.5}
.rir-table td,.rir-table th{border:1px solid #d2d2d7;padding:6px 9px;vertical-align:top;min-width:40px}
.rir-table .rir-para{font-size:15px;line-height:1.5}
@media(max-width:640px){.rir-zone{gap:7px}.rir-para{font-size:15px;line-height:1.58}.rir-image{margin:8px auto}.rir-table{font-size:13px}.rir-table td,.rir-table th{padding:5px 7px}}
`;
    document.head.appendChild(style);
  }

  function resolveReportUrl(options) {
    if (options && options.reportUrl) return options.reportUrl;
    return "/api/render_report";
  }

  function postRenderReport(payload, options) {
    if (options && options.disableReports) return;
    if (typeof fetch !== "function") return;
    const reportUrl = resolveReportUrl(options || {});
    const body = JSON.stringify({
      item_id: options && options.itemId ? options.itemId : "",
      path: typeof location !== "undefined" ? location.href : "",
      user_agent: typeof navigator !== "undefined" ? navigator.userAgent : "",
      ...payload
    });
    fetch(reportUrl, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body,
      keepalive: true
    }).catch(function () {});
  }

  function textNode(node) {
    const text = String((node && node.text) || "");
    if (!text) return document.createDocumentFragment();
    const span = document.createElement("span");
    span.className = "rir-text";
    span.textContent = text;
    return span;
  }

  function latexNode(node, options) {
    // QA-0:默认裸 inline 渲染(iOS 上 overflow 容器会显示滚动条槽,像"阴影"且撑行距);
    // 渲染后测宽,只有真正超出容器的长公式才包 scroll。
    const target = document.createElement("span");
    target.className = "rir-latex";
    const latex = normalizeLatexForKatex(String((node && node.latex) || ""));
    try {
      if (!window.katex || typeof window.katex.render !== "function") {
        throw new Error("katex_not_loaded");
      }
      window.katex.render(latex, target, {
        throwOnError: false,
        strict: "ignore",
        trust: false
      });
      if (target.querySelector(".katex-error")) {
        postRenderReport({
          issue_type: "katex_error_markup",
          source: node.source || "",
          latex
        }, options);
      }
    } catch (err) {
      target.textContent = latex;
      target.classList.add("katex-error");
      postRenderReport({
        issue_type: "katex_exception",
        source: node.source || "",
        latex,
        error: err && err.message ? err.message : String(err)
      }, options);
    }
    return target;
  }

  function wrapOverflowingLatex(container) {
    // 渲染入 DOM 后调用:超宽公式才加横向滚动容器(短公式保持裸 inline,无滚动条槽)。
    const paraWidth = container.clientWidth;
    if (!paraWidth) return;
    container.querySelectorAll(".rir-latex").forEach(function (el) {
      if (el.parentElement && el.parentElement.classList.contains("rir-latex-scroll")) return;
      if (el.scrollWidth > paraWidth + 2) {
        const wrap = document.createElement("span");
        wrap.className = "rir-latex-scroll";
        el.replaceWith(wrap);
        wrap.appendChild(el);
      }
    });
  }

  function normalizeLatexForKatex(latex) {
    return latex.replace(/(\\(?:,|;|:|!|quad|qquad)|\\ )\s*\^\{/g, "$1{}^{");
  }

  function imageNode(node) {
    const img = document.createElement("img");
    img.className = "rir-image";
    img.loading = "lazy";
    img.decoding = "async";
    img.src = String((node && node.url) || "");
    img.srcset = `${img.src} 1x, ${img.src} 2x`;
    img.alt = String((node && node.alt) || "");
    if (node && node.asset_hash) img.dataset.assetHash = String(node.asset_hash);
    if (node && node.inline) {
      img.classList.add("rir-img-inline");  // 行内小图(公式/角标级),随文字高度,不独占行
    } else if (node && node.w) {
      // QA-0:按原始像素设上限,小图不被拉伸撑满(用户反馈"图片过大");大图仍受 max-width:100% 约束
      img.style.width = `min(${Number(node.w)}px, 100%)`;
    } else {
      img.style.maxWidth = "100%";
    }
    return img;
  }

  function placeholderNode(node) {
    const box = document.createElement("div");
    box.className = "rir-placeholder";
    box.textContent = "图暂缺";
    box.dataset.reason = String((node && node.reason) || "");
    return box;
  }

  function tableNode(node, options) {
    const wrap = document.createElement("div");
    wrap.className = "rir-table-wrap";
    const table = document.createElement("table");
    table.className = "rir-table";
    const tbody = document.createElement("tbody");
    (node.rows || []).forEach(function (row) {
      const tr = document.createElement("tr");
      (row || []).forEach(function (cell) {
        const td = document.createElement("td");
        renderNodes(Array.isArray(cell) ? cell : [], td, options);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function renderNode(node, options) {
    if (!node || !NODE_KINDS.has(node.kind)) {
      const unknown = document.createElement("span");
      unknown.className = "rir-text";
      postRenderReport({
        issue_type: "unknown_rir_kind",
        kind: node && node.kind ? String(node.kind) : ""
      }, options);
      return unknown;
    }
    if (node.kind === "text") return textNode(node);
    if (node.kind === "latex") return latexNode(node, options);
    if (node.kind === "image") return imageNode(node);
    if (node.kind === "table") return tableNode(node, options);
    return placeholderNode(node);
  }

  function renderNodes(nodes, container, options) {
    (nodes || []).forEach(function (node) {
      container.appendChild(renderNode(node, options || {}));
    });
  }

  function renderZone(paragraphs, container, options) {
    injectStyles();
    container.textContent = "";
    container.classList.add("rir-zone");
    (paragraphs || []).forEach(function (paraNodes) {
      const para = document.createElement("div");
      para.className = "rir-para";
      renderNodes(Array.isArray(paraNodes) ? paraNodes : [], para, options || {});
      if (isOptionPara(paraNodes)) para.classList.add("rir-option");
      if (hasVisibleContent(para)) container.appendChild(para);
    });
    // 入 DOM 后测宽:仅超宽公式加横向滚动(requestAnimationFrame 等布局完成)
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(function () { wrapOverflowingLatex(container); });
    }
  }

  function isOptionPara(paraNodes) {
    if (!Array.isArray(paraNodes)) return false;
    for (let i = 0; i < paraNodes.length; i++) {
      const n = paraNodes[i];
      if (!n) continue;
      if (n.kind === "text") {
        const t = String(n.text || "").trim();
        if (!t) continue;
        return /^[A-D][.．、]/.test(t);
      }
      return false;
    }
    return false;
  }

  function hasVisibleContent(el) {
    if ((el.textContent || "").trim()) return true;
    return Boolean(el.querySelector("img,.katex,.rir-placeholder,table"));
  }

  function renderRir(rir, container, options) {
    injectStyles();
    container.textContent = "";
    const zones = (rir && rir.zones) || {};
    Object.keys(zones).forEach(function (zoneName) {
      const section = document.createElement("section");
      section.className = "rir-zone-section";
      section.dataset.zone = zoneName;
      renderZone(zones[zoneName], section, {
        ...(options || {}),
        itemId: (options && options.itemId) || (rir && rir.item_id) || ""
      });
      container.appendChild(section);
    });
  }

  window.YHerRirRenderer = {
    renderRir,
    renderZone,
    renderNodes,
    postRenderReport
  };
})();
