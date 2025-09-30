const $ = (id) => document.getElementById(id);

// changed: read apiBase lazily and provide sane default
function _apiBaseInput() {
  return document.getElementById("apiBase") || { value: "http://localhost:8000" };
}

function apiBase() {
  return (_apiBaseInput().value || "").replace(/\/+$/, "");
}

async function postForm(path, form) {
  const url = apiBase() + path;
  let res;
  try {
    res = await fetch(url, { method: "POST", body: form });
  } catch (err) {
    throw new Error(`Network error while POST ${url}: ${err.message}`);
  }
  const text = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  try { return JSON.parse(text); } catch { return text; }
}

async function postEmpty(path) {
  const url = apiBase() + path;
  let res;
  try {
    res = await fetch(url, { method: "POST" });
  } catch (err) {
    throw new Error(`Network error while POST ${url}: ${err.message}`);
  }
  const text = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  try { return JSON.parse(text); } catch { return text; }
}

async function getJson(path) {
  const url = apiBase() + path;
  let res;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new Error(`Network error while GET ${url}: ${err.message}`);
  }
  const text = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  try { return JSON.parse(text); } catch { return text; }
}

function setText(elId, text) {
  $(elId).textContent = typeof text === "string" ? text : JSON.stringify(text, null, 2);
}

/* Upload */
$("uploadBtn").addEventListener("click", async () => {
  const f = $("ebookFile").files[0];
  if (!f) return alert("Select a file first");
  setText("uploadResult", "Uploading...");
  const fd = new FormData();
  fd.append("file", f, f.name);
  try {
    const data = await postForm("/upload_ebook", fd);
    setText("uploadResult", data);
    if (data.session_id) $("sessionId").value = data.session_id;
    if (data.stream_url) $("downloadLink").href = data.stream_url;
  } catch (e) {
    setText("uploadResult", String(e));
  }
});

/* Stream controls */
async function startStreamForSession(sid) {
  if (!sid) return alert("Provide session id");
  const streamUrl = `${apiBase()}/stream/${encodeURIComponent(sid)}`;
  const player = $("player");
  player.src = streamUrl;
  player.play().catch(()=>{ /* autoplay may be blocked by browser */});
  $("downloadLink").href = streamUrl;
}

$("startStream").addEventListener("click", () => startStreamForSession($("sessionId").value));
$("playBtn").addEventListener("click", async () => {
  const sid = $("sessionId").value;
  if (!sid) return alert("Provide session id");
  try {
    await postEmpty(`/play/${encodeURIComponent(sid)}`);
  } catch (e) { console.error(e); alert("Play request failed: "+e.message); }
});
$("pauseBtn").addEventListener("click", async () => {
  const sid = $("sessionId").value;
  if (!sid) return alert("Provide session id");
  try {
    await postEmpty(`/pause/${encodeURIComponent(sid)}`);
  } catch (e) { console.error(e); alert("Pause request failed: "+e.message); }
});
$("stopBtn").addEventListener("click", async () => {
  const sid = $("sessionId").value;
  if (!sid) return alert("Provide session id");
  try {
    await postEmpty(`/stop/${encodeURIComponent(sid)}`);
    const player = $("player");
    player.pause();
    player.removeAttribute("src");
    player.load();
  } catch (e) { console.error(e); alert("Stop request failed: "+e.message); }
});

/* Status */
$("refreshStatus").addEventListener("click", async () => {
  const sid = $("sessionId").value;
  if (!sid) return alert("Provide session id");
  setText("statusResult", "Loading...");
  try {
    const st = await getJson(`/status/${encodeURIComponent(sid)}`);
    setText("statusResult", st);
  } catch (e) {
    setText("statusResult", String(e));
  }
});

/* Ask LLM via audio */
$("askBtn").addEventListener("click", async () => {
  const sid = $("sessionId").value;
  if (!sid) return alert("Provide session id");
  const f = $("questionFile").files[0];
  if (!f) return alert("Select a question audio file");
  setText("askResult", "Sending...");
  const fd = new FormData();
  fd.append("question_audio", f, f.name);
  try {
    const data = await postForm(`/ask/${encodeURIComponent(sid)}`, fd);
    setText("askResult", data);
    if (data.answer_audio_url) {
      const ap = $("answerPlayer");
      ap.src = data.answer_audio_url;
      ap.play().catch(()=>{});
    }
  } catch (e) {
    setText("askResult", String(e));
  }
});