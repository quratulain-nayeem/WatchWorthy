const API_BASE = "https://quratulainnnnn-watchworthy.hf.space";

let currentVideoId = null;

// ── Helpers ───────────────────────────────────────────────────────────────────

function extractVideoId(url) {
  const match = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : null;
}
// ── Panel HTML ────────────────────────────────────────────────────────────────

function createPanel() {
  const panel = document.createElement("div");
  panel.id = "ww-panel";
  panel.innerHTML = `
    <div class="ww-header">
      <span class="ww-logo">▶ WatchWorthy</span>
      <span class="ww-tagline">Is this tutorial worth your time?</span>
    </div>
    <div id="ww-body">
      <div class="ww-loading">Analyzing video...</div>
    </div>
  `;
  return panel;
}

// ── Inject panel into YouTube ─────────────────────────────────────────────────

function injectPanel() {
  if (document.getElementById("ww-panel")) return;

  const target =
    document.querySelector("#secondary") ||
    document.querySelector("#below") ||
    document.querySelector("ytd-watch-flexy");

  if (!target) return;

  const panel = createPanel();

  if (target.id === "secondary") {
    target.prepend(panel);
  } else {
    target.after(panel);
  }
}

// ── Render results ────────────────────────────────────────────────────────────

function renderResults(data) {
  const body = document.getElementById("ww-body");
  if (!body) return;

  const score = data.score ?? 0;
  const depth = data.depth ?? 0;
  const sentiment = data.sentiment ?? 0;
  const engagement = data.engagement ?? 0;
  const filler = data.filler_pct ?? 0;
  const commentPcts = data.comment_pcts ?? {};
  const commentExamples = data.comment_examples ?? {};
  const hasTranscript = data.has_transcript !== false;

  // Score ring circumference
  const radius = 36;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (score / 10) * circ;

  // Comment pills
  const pillLabels = {
    helpful: "✓ Helpful",
    confused: "? Confused",
    outdated: "⚠ Outdated",
    off_topic: "✕ Off-topic",
  };

  const pillsHTML = Object.entries(commentPcts)
    .map(
      ([key, pct]) => `
      <div class="ww-pill" data-key="${key}">
        ${pillLabels[key] || key} ${pct}%
      </div>`
    )
    .join("");

  // Timestamps

  body.innerHTML = `
    <!-- Score -->
    <div class="ww-score-area">
      <div class="ww-ring-wrap">
        <svg width="90" height="90" viewBox="0 0 90 90">
          <circle class="ww-ring-bg" cx="45" cy="45" r="${radius}"/>
          <circle
            class="ww-ring-fill"
            cx="45" cy="45" r="${radius}"
            stroke-dasharray="${circ}"
            stroke-dashoffset="${circ}"
            id="ww-ring-fill"
          />
        </svg>
        <div class="ww-score-label">${score.toFixed(1)}</div>
      </div>

      <div class="ww-subscores">
        <div class="ww-bar-row">
          <span class="ww-bar-label">Depth</span>
          <div class="ww-bar-track">
            <div class="ww-bar-fill" style="width:0%" id="ww-bar-depth"></div>
          </div>
          <span class="ww-bar-val">${depth.toFixed(1)}</span>
        </div>
        <div class="ww-bar-row">
          <span class="ww-bar-label">Sentiment</span>
          <div class="ww-bar-track">
            <div class="ww-bar-fill" style="width:0%" id="ww-bar-sentiment"></div>
          </div>
          <span class="ww-bar-val">${sentiment.toFixed(1)}</span>
        </div>
        <div class="ww-bar-row">
          <span class="ww-bar-label">Engagement</span>
          <div class="ww-bar-track">
            <div class="ww-bar-fill" style="width:0%" id="ww-bar-engagement"></div>
          </div>
          <span class="ww-bar-val">${engagement.toFixed(1)}</span>
        </div>
      </div>
    </div>

    <!-- Filler -->
    <div class="ww-filler">Filler content: <span>${filler}%</span></div>

    <!-- Comment pills -->
    <div class="ww-section-title">Viewer Sentiment</div>
    <div class="ww-pills">${pillsHTML}</div>
    <div class="ww-comment-box" id="ww-comment-box"></div>

    <hr class="ww-divider"/>

    <!-- Q&A -->
    <div class="ww-section-title">Ask About This Video</div>
    <div class="ww-qa-input-row">
      <input
        class="ww-input"
        id="ww-qa-input"
        type="text"
        placeholder="${hasTranscript ? "Ask anything about this video..." : "Transcript unavailable for Q&A"}"
        ${hasTranscript ? "" : "disabled"}
      />
      <button class="ww-btn" id="ww-qa-btn" ${hasTranscript ? "" : "disabled"}>Ask</button>
    </div>
    <div class="ww-answer-box" id="ww-answer-box"></div>

    <hr class="ww-divider"/>

    <!-- Recommendations -->
    <div class="ww-section-title">Find Similar Videos</div>
    <div class="ww-rec-input-row">
      <input
        class="ww-input"
        id="ww-rec-input"
        type="text"
        placeholder="What do you want to learn?"
      />
      <button class="ww-btn" id="ww-rec-btn">Search</button>
    </div>
    <div id="ww-rec-cards" class="ww-rec-cards"></div>
  `;

  // Animate score ring
  requestAnimationFrame(() => {
    setTimeout(() => {
      const ring = document.getElementById("ww-ring-fill");
      if (ring) ring.style.strokeDashoffset = offset;

      const animate = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.style.width = `${(val / 10) * 100}%`;
      };
      animate("ww-bar-depth", depth);
      animate("ww-bar-sentiment", sentiment);
      animate("ww-bar-engagement", engagement);
    }, 100);
  });

  // Comment pills click
  document.querySelectorAll(".ww-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const key = pill.dataset.key;
      const box = document.getElementById("ww-comment-box");
      const examples = commentExamples[key] || [];

      const isActive = pill.classList.contains("active");
      document.querySelectorAll(".ww-pill").forEach((p) => p.classList.remove("active"));
      box.classList.remove("visible");

      if (!isActive) {
        pill.classList.add("active");
        box.innerHTML = examples.length
          ? examples.map((c) => `<div style="margin-bottom:6px">• ${c}</div>`).join("")
          : "<div>No examples available</div>";
        box.classList.add("visible");
      }
    });
  });

  // Q&A
  const qaBtn = document.getElementById("ww-qa-btn");
  const qaInput = document.getElementById("ww-qa-input");
  const answerBox = document.getElementById("ww-answer-box");

  qaBtn.addEventListener("click", async () => {
    const question = qaInput.value.trim();
    if (!question || !currentVideoId) return;
    if (!hasTranscript) {
      answerBox.textContent = "Transcript unavailable for this video, so Q&A cannot run.";
      answerBox.classList.add("visible");
      return;
    }

    qaBtn.disabled = true;
    qaBtn.textContent = "...";
    answerBox.classList.remove("visible");

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: currentVideoId,
          question,
          client_transcript_attempted: true,
        }),
      });
      const json = await res.json();
      answerBox.textContent = json.answer || json.detail || "No answer returned.";
      answerBox.classList.add("visible");
    } catch (err) {
      answerBox.textContent = "Failed to get answer. Try again.";
      answerBox.classList.add("visible");
    }

    qaBtn.disabled = false;
    qaBtn.textContent = "Ask";
  });

  // Q&A on Enter key
  qaInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") qaBtn.click();
  });

  // Recommendations
  const recBtn = document.getElementById("ww-rec-btn");
  const recInput = document.getElementById("ww-rec-input");
  const recCards = document.getElementById("ww-rec-cards");

  recBtn.addEventListener("click", async () => {
    const query = recInput.value.trim();
    if (!query || !currentVideoId) return;

    recBtn.disabled = true;
    recBtn.textContent = "...";
    recCards.innerHTML = `<div class="ww-loading">Finding similar videos...</div>`;

    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: currentVideoId,
          query,
          title: data.title,
        }),
      });
      const json = await res.json();
      const recs = json.recommendations || [];

      if (json.message) {
        recCards.innerHTML = `<div class="ww-error">${json.message}</div>`;
      } else if (recs.length === 0) {
        recCards.innerHTML = `<div class="ww-loading">No relevant videos found.</div>`;
      } else {
        recCards.innerHTML = recs
          .map(
            (r) => `
            <a class="ww-rec-card" href="https://www.youtube.com/watch?v=${r.video_id}" target="_blank">
              <img class="ww-rec-thumb" src="${r.thumbnail}" alt="${r.title}"/>
              <div class="ww-rec-info">
                <div class="ww-rec-title">${r.title}</div>
                <div class="ww-rec-channel">${r.channel}</div>
              </div>
              <div class="ww-rec-score">${r.score.toFixed(1)}</div>
            </a>`
          )
          .join("");
      }
    } catch (err) {
      recCards.innerHTML = `<div class="ww-error">Failed to fetch recommendations.</div>`;
    }

    recBtn.disabled = false;
    recBtn.textContent = "Search";
  });

  recInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") recBtn.click();
  });
}

// ── Main analyze flow ─────────────────────────────────────────────────────────
function parseJsonValueAfter(text, marker) {
  const markerIndex = text.indexOf(marker);
  if (markerIndex === -1) return null;

  const start = text.slice(markerIndex + marker.length).search(/[\[{]/);
  if (start === -1) return null;

  const jsonStart = markerIndex + marker.length + start;
  const open = text[jsonStart];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = jsonStart; i < text.length; i++) {
    const ch = text[i];

    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (ch === open) depth++;
    if (ch === close) depth--;

    if (depth === 0) {
      return JSON.parse(text.slice(jsonStart, i + 1));
    }
  }

  return null;
}

function getCaptionTracks(html) {
  const playerResponse = parseJsonValueAfter(html, "ytInitialPlayerResponse");
  const tracksFromPlayer =
    playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks;

  if (Array.isArray(tracksFromPlayer)) return tracksFromPlayer;

  const tracks = parseJsonValueAfter(html, "\"captionTracks\":");
  return Array.isArray(tracks) ? tracks : [];
}

function selectCaptionTrack(tracks) {
  return (
    tracks.find((t) => t.languageCode === "en" && !t.kind) ||
    tracks.find((t) => t.languageCode === "en") ||
    tracks.find((t) => t.languageCode?.startsWith("en")) ||
    tracks[0]
  );
}

function parseJson3Transcript(payload) {
  const lines = [];

  for (const event of payload.events || []) {
    const text = (event.segs || [])
      .map((seg) => seg.utf8 || "")
      .join("")
      .replace(/\s+/g, " ")
      .trim();

    if (text) lines.push(text);
  }

  return lines.join(" ").trim();
}

function parseXmlTranscript(xml) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, "text/xml");

  return [...doc.querySelectorAll("text")]
    .map((el) => el.textContent.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join(" ")
    .trim();
}

async function fetchTranscriptFromYouTube(videoId) {
  try {
    const pageHtml = document.documentElement?.innerHTML || "";
    let tracks = getCaptionTracks(pageHtml);

    if (!tracks.length) {
      const res = await fetch(`https://www.youtube.com/watch?v=${videoId}`);
      tracks = getCaptionTracks(await res.text());
    }

    const track = selectCaptionTrack(tracks);
    if (!track?.baseUrl) return null;

    const trackUrl = new URL(track.baseUrl);
    trackUrl.searchParams.set("fmt", "json3");

    const transcriptRes = await fetch(trackUrl.toString());
    const transcriptBody = await transcriptRes.text();

    try {
      const transcript = parseJson3Transcript(JSON.parse(transcriptBody));
      return transcript.length > 40 ? transcript : null;
    } catch {
      const transcript = parseXmlTranscript(transcriptBody);
      return transcript.length > 40 ? transcript : null;
    }
  } catch (err) {
    console.error("Transcript fetch failed:", err);
    return null;
  }
}
async function analyze(url) {
  const videoId = extractVideoId(url);
  if (!videoId || videoId === currentVideoId) return;

  currentVideoId = videoId;

  const body = document.getElementById("ww-body");
  if (body) body.innerHTML = `<div class="ww-loading">Fetching transcript...</div>`;

  // Fetch transcript client-side first
  const transcript = await fetchTranscriptFromYouTube(videoId);

  if (body) body.innerHTML = `<div class="ww-loading">Analyzing video...</div>`;

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        transcript,
        client_transcript_attempted: true,
      }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    console.error(err);
    if (body) {
      body.innerHTML = `
        <div class="ww-error">
          Could not reach the server. Wait 30 seconds and click Analyze again.
        </div>
        <button class="ww-btn" id="ww-analyze-btn" style="width:100%; padding:12px; margin-top:12px;">
          Try Again
        </button>
      `;
      document.getElementById("ww-analyze-btn")?.addEventListener("click", () => {
        currentVideoId = null;
        analyze(window.location.href);
      });
    }
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  if (!window.location.href.includes("youtube.com/watch")) return;

  setTimeout(() => {
    injectPanel();
    const body = document.getElementById("ww-body");
    if (body) {
      body.innerHTML = `
        <button class="ww-btn" id="ww-analyze-btn" style="width:100%; padding: 12px;">
          Analyze This Video
        </button>
      `;
        document.getElementById("ww-analyze-btn")
    ?.addEventListener("click", () => {
        currentVideoId = null;
        analyze(window.location.href);
    });
    }
  }, 1500);
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "VIDEO_CHANGED") {
    currentVideoId = null;

    // Remove existing panel immediately
    const existing = document.getElementById("ww-panel");
    if (existing) existing.remove();

    // Retry injecting until #secondary is available
    let attempts = 0;
    const tryInject = () => {
      attempts++;
      const target = document.querySelector("#secondary");
      if (!target && attempts < 10) {
        setTimeout(tryInject, 500);
        return;
      }
      injectPanel();
      const body = document.getElementById("ww-body");
      if (body) {
        body.innerHTML = `
          <button class="ww-btn" id="ww-analyze-btn" style="width:100%; padding: 12px;">
            Analyze This Video
          </button>
        `;
        document.getElementById("ww-analyze-btn")?.addEventListener("click", () => {
          analyze(msg.url);
        });
      }
    };
    setTimeout(tryInject, 800);
  }
});

init();
