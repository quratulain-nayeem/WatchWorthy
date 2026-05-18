from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, re, requests, numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────────────
print("Loading models...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
print("Models ready.")

# ── CAG transcript cache ─────────────────────────────────────────────────────
transcript_cache: dict[str, str] = {}

# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract video ID")

def fetch_video_metadata(video_id: str) -> dict:
    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="Video not found")
    item = items[0]
    snippet = item["snippet"]
    stats   = item.get("statistics", {})
    return {
        "title":        snippet.get("title", ""),
        "channel":      snippet.get("channelTitle", ""),
        "thumbnail":    snippet["thumbnails"].get("high", {}).get("url", ""),
        "views":        int(stats.get("viewCount",    0)),
        "likes":        int(stats.get("likeCount",    0)),
        "comment_count":int(stats.get("commentCount", 0)),
    }

def fetch_comments(video_id: str, max_results: int = 50) -> list[str]:
    url = (
        "https://www.googleapis.com/youtube/v3/commentThreads"
        f"?part=snippet&videoId={video_id}&maxResults={max_results}"
        f"&order=relevance&key={YOUTUBE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return []
    items = r.json().get("items", [])
    return [
        i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for i in items
    ]

def fetch_transcript(video_id: str) -> str | None:
    try:
        ytt = YouTubeTranscriptApi()
        chunks = ytt.fetch(video_id)
        return " ".join(c.text for c in chunks)
    except Exception:
        return None

def score_engagement(views: int, likes: int) -> float:
    if views == 0:
        return 5.0
    ratio = likes / views
    # 4%+ like ratio → ~10, scaled down from there
    return round(min(10.0, (ratio / 0.04) * 10), 2)

def score_sentiment(comments: list[str]) -> tuple[float, dict]:
    if not comments:
        return 5.0, {}
    labels = ["understanding", "confusion", "outdated", "irrelevant"]
    counts = {l: 0 for l in labels}
    for c in comments[:30]:            # limit to keep latency sane
        result = classifier(c, candidate_labels=labels)
        counts[result["labels"][0]] += 1
    total = sum(counts.values()) or 1
    pcts  = {k: round(v / total * 100) for k, v in counts.items()}
    # Understanding contributes positively, others negatively
    score = (
        counts["understanding"] * 1.0
        - counts["confusion"]   * 0.5
        - counts["outdated"]    * 0.4
        - counts["irrelevant"]  * 0.1
    ) / total * 10
    return round(max(0.0, min(10.0, score)), 2), pcts
def fetch_transcript(video_id: str) -> str | None:
    try:
        ytt = YouTubeTranscriptApi()
        chunks = ytt.fetch(video_id)
        return " ".join(c.text for c in chunks)
    except Exception as e:
        print(f"Transcript unavailable for {video_id}: {e}")
        return None

def score_content(transcript: str, title: str) -> tuple[float, float, list[dict]]:
    """Returns (depth_score, filler_pct, timestamps)"""
    if not transcript:
        return 5.0, 0.0, []

    words  = transcript.split()
    n      = len(words)
    # Density-spike timestamp extraction (simple window approach)
    window = max(1, n // 20)
    title_emb  = embedder.encode(title, convert_to_tensor=True)
    timestamps = []
    for i in range(0, n - window, window):
        chunk = " ".join(words[i:i + window])
        chunk_emb = embedder.encode(chunk, convert_to_tensor=True)
        sim = float(util.cos_sim(title_emb, chunk_emb))
        if sim > 0.25:
            seconds = i * 2          # rough estimate: ~2 s/word avg
            timestamps.append({"seconds": seconds, "label": chunk[:60] + "…", "sim": sim})

    timestamps = sorted(timestamps, key=lambda x: -x["sim"])[:5]
    timestamps = sorted(timestamps, key=lambda x:  x["seconds"])

    # Filler = chunks with very low similarity to title
    all_sims = []
    for i in range(0, n - window, window):
        chunk = " ".join(words[i:i + window])
        chunk_emb = embedder.encode(chunk, convert_to_tensor=True)
        all_sims.append(float(util.cos_sim(title_emb, chunk_emb)))

    filler_pct = round(sum(1 for s in all_sims if s < 0.15) / max(len(all_sims), 1) * 100, 1)
    depth_score = round(min(10.0, np.mean(all_sims) * 40), 2)   # scale to 0-10

    return depth_score, filler_pct, timestamps

def compute_final_score(depth, sentiment, engagement, has_transcript: bool) -> float:
    if has_transcript:
        return round(depth * 0.4 + sentiment * 0.3 + engagement * 0.3, 2)
    else:
        return round(sentiment * 0.5 + engagement * 0.5, 2)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    meta       = fetch_video_metadata(video_id)
    comments   = fetch_comments(video_id)
    transcript = fetch_transcript(video_id)

    # Cache transcript for CAG
    if transcript:
        transcript_cache[video_id] = transcript

    engagement_score              = score_engagement(meta["views"], meta["likes"])
    sentiment_score, comment_pcts = score_sentiment(comments)
    depth_score, filler_pct, timestamps = score_content(transcript, meta["title"])
    final_score = compute_final_score(
        depth_score, sentiment_score, engagement_score,
        has_transcript=transcript is not None
    )

    return JSONResponse({
        "video_id":       video_id,
        "title":          meta["title"],
        "channel":        meta["channel"],
        "thumbnail":      meta["thumbnail"],
        "views":          meta["views"],
        "likes":          meta["likes"],
        "comment_count":  meta["comment_count"],
        "score":          final_score,
        "depth":          depth_score,
        "sentiment":      sentiment_score,
        "engagement":     engagement_score,
        "filler_pct":     filler_pct,
        "comment_pcts":   comment_pcts,
        "timestamps":     timestamps,
        "has_transcript": transcript is not None,
    })

class AskRequest(BaseModel):
    video_id: str
    question: str

@app.post("/ask")
async def ask(req: AskRequest):
    transcript = transcript_cache.get(req.video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not available for this video")

    # CAG: inject full transcript into context
    prompt_context = transcript[:4000]   # truncate to safe length
    answer_emb   = embedder.encode(req.question, convert_to_tensor=True)
    chunks       = [prompt_context[i:i+300] for i in range(0, len(prompt_context), 300)]
    chunk_embs   = embedder.encode(chunks, convert_to_tensor=True)
    sims         = util.cos_sim(answer_emb, chunk_embs)[0]
    best_idx     = int(sims.argmax())
    best_chunk   = chunks[best_idx]

    return JSONResponse({
        "answer": f"Based on the transcript: \"{best_chunk.strip()}\""
    })