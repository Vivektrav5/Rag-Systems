const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question");
const statusEl = document.getElementById("status");
const fileEl = document.getElementById("pdf-file");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatusEl = document.getElementById("upload-status");
const docsListEl = document.getElementById("docs-list");

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.ready) {
      statusEl.textContent = data.active_pdf
        ? `ready — ${data.active_pdf}`
        : "backend ready";
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "no PDF loaded yet";
      statusEl.className = "status bad";
    }
  } catch {
    statusEl.textContent = "backend unreachable";
    statusEl.className = "status bad";
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = fileEl.files[0];
  if (!file) {
    uploadStatusEl.textContent = "Choose a PDF first";
    uploadStatusEl.className = "upload-status bad";
    return;
  }
  await doUpload(file, false);
});

async function doUpload(file, force) {
  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  uploadStatusEl.textContent = force ? "Re-uploading..." : "Uploading...";
  uploadStatusEl.className = "upload-status";

  try {
    const res = await fetch(`/upload?force=${force}`, { method: "POST", body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Upload failed (${res.status})`);
    }
    const data = await res.json();

    if (data.result === "duplicate") {
      uploadStatusEl.innerHTML = "";
      const warn = document.createElement("span");
      warn.textContent = `⚠️ "${data.filename}" (v${data.existing_version}) already indexed, no changes detected. `;

      const confirmBtn = document.createElement("button");
      confirmBtn.textContent = "Re-index anyway";
      confirmBtn.className = "inline-confirm-btn";
      confirmBtn.addEventListener("click", () => doUpload(file, true));

      uploadStatusEl.appendChild(warn);
      uploadStatusEl.appendChild(confirmBtn);
      uploadStatusEl.className = "upload-status bad";
      uploadBtn.disabled = false;
      return;
    }

    await pollJob(data.job_id, data.result === "new_version");
  } catch (err) {
    uploadStatusEl.textContent = `Error: ${err.message}`;
    uploadStatusEl.className = "upload-status bad";
    uploadBtn.disabled = false;
  }
}

async function pollJob(jobId, isNewVersion) {
  while (true) {
    const res = await fetch(`/upload/status/${jobId}`);
    const job = await res.json();

    if (job.status === "splitting") {
      uploadStatusEl.textContent = isNewVersion
        ? "New version detected — parsing & chunking..."
        : "Parsing & chunking PDF...";
    } else if (job.status === "embedding") {
      const pct = job.total ? Math.round((job.progress / job.total) * 100) : 0;
      uploadStatusEl.textContent = `Embedding chunks: ${job.progress}/${job.total} (${pct}%)`;
    } else if (job.status === "queued") {
      uploadStatusEl.textContent = "Queued...";
    } else if (job.status === "done") {
      const versionLabel = job.version ? ` (v${job.version})` : "";
      uploadStatusEl.textContent = isNewVersion
        ? `Replaced with new version: ${job.filename}${versionLabel}`
        : `Indexed: ${job.filename}${versionLabel} (${job.total} chunks)`;
      uploadStatusEl.className = "upload-status ok";
      chatEl.innerHTML = "";
      fileEl.value = "";
      uploadBtn.disabled = false;
      checkHealth();
      loadDocuments();
      return;
    } else if (job.status === "error") {
      uploadStatusEl.textContent = `Error: ${job.error}`;
      uploadStatusEl.className = "upload-status bad";
      uploadBtn.disabled = false;
      return;
    }

    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const docs = await res.json();
    docsListEl.innerHTML = "";

    const names = Object.keys(docs);
    if (names.length === 0) {
      docsListEl.textContent = "No documents uploaded yet.";
      return;
    }

    names.forEach((name) => {
      const info = docs[name];
      const item = document.createElement("details");
      item.className = "doc-item";

      const summary = document.createElement("summary");
      summary.textContent = `${name} — v${info.current_version} (${info.version_count} version${info.version_count > 1 ? "s" : ""})`;
      item.appendChild(summary);

      info.versions.slice().reverse().forEach((v) => {
        const row = document.createElement("div");
        row.className = "doc-version-row";
        row.textContent = `v${v.version} — uploaded ${v.uploaded_at}`;
        item.appendChild(row);
      });

      docsListEl.appendChild(item);
    });
  } catch {
    docsListEl.textContent = "Could not load document history.";
  }
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);

  sources.forEach((src, i) => {
    const pageLabel = src.page !== null && src.page !== undefined ? ` (page ${src.page})` : "";
    const chunk = document.createElement("div");
    chunk.className = "source-chunk";
    chunk.textContent = `Chunk ${i + 1}${pageLabel}: ${src.content}`;
    details.appendChild(chunk);
  });

  container.appendChild(details);
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;

  addMessage("user", question);
  inputEl.value = "";
  inputEl.disabled = true;
  formEl.querySelector("button").disabled = true;

  const thinkingEl = addMessage("assistant", "Thinking...");

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    thinkingEl.textContent = data.answer;
    addSources(thinkingEl, data.sources);
  } catch (err) {
    thinkingEl.textContent = `Error: ${err.message}`;
    thinkingEl.className = "msg error";
  } finally {
    inputEl.disabled = false;
    formEl.querySelector("button").disabled = false;
    inputEl.focus();
  }
});

loadDocuments();
checkHealth();