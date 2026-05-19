from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import re
import json
import pathlib
import requests
import numpy as np

from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from groq import Groq
from spellchecker import SpellChecker

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Models ───────────────────────────────────────────────────────────────────
print("Loading models...")

classifier  = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedder    = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

print("Models ready.")

# ── Transcript Cache ─────────────────────────────────────────────────────────
CACHE_FILE = pathlib.Path("transcript_cache.json")

transcript_cache: dict[str, str] = (
    json.loads(CACHE_FILE.read_text())
    if CACHE_FILE.exists()
    else {}
)

def save_cache():
    CACHE_FILE.write_text(json.dumps(transcript_cache))

# ── Spellchecker ─────────────────────────────────────────────────────────────
QUERY_SLANG_WHITELIST = {
    "vid", "vids", "lol", "tbh", "imo", "btw", "ngl",
    "abt", "rn", "yt", "bro", "bru", "pls", "plz", "thx"
}

def build_spell_checker(transcript: str | None) -> SpellChecker:
    sc = SpellChecker()
    if not transcript:
        return sc
    words_raw = re.findall(r"[a-zA-Z]{3,}", transcript.lower())
    freq: dict[str, int] = {}
    for w in words_raw:
        freq[w] = freq.get(w, 0) + 1
    domain_terms = [w for w, count in freq.items() if count >= 2]
    sc.word_frequency.load_words(domain_terms)
    return sc


def correct_query(text: str, spell: SpellChecker) -> str:
    words     = text.split()
    corrected = []
    for i, word in enumerate(words):
        stripped = word.strip("?.,!;:'\"")
        # Preserve slang
        if stripped.lower() in QUERY_SLANG_WHITELIST:
            corrected.append(word)
            continue
        # Preserve acronyms / very short words
        if stripped.isupper() or len(stripped) <= 2:
            corrected.append(word)
            continue
        # Preserve likely proper nouns (capitalised mid-sentence)
        if i > 0 and stripped[0].isupper():
            corrected.append(word)
            continue
        candidate = spell.correction(stripped)
        if candidate is None or candidate == stripped:
            corrected.append(word)
        else:
            corrected.append(word.replace(stripped, candidate))
    result = " ".join(corrected)
    if result != text:
        print(f"Query corrected: '{text}' → '{result}'")
    return result

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
    item    = items[0]
    snippet = item["snippet"]
    stats   = item.get("statistics", {})
    return {
        "title":         snippet.get("title", ""),
        "channel":       snippet.get("channelTitle", ""),
        "thumbnail":     snippet["thumbnails"].get("high", {}).get("url", ""),
        "views":         int(stats.get("viewCount", 0)),
        "likes":         int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
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
        ytt             = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        try:
            chunks = transcript_list.find_manually_created_transcript(["en"]).fetch()
            return " ".join(c.text for c in chunks)
        except Exception:
            pass
        try:
            chunks = transcript_list.find_generated_transcript(["en"]).fetch()
            return " ".join(c.text for c in chunks)
        except Exception:
            pass
        try:
            transcript = next(iter(transcript_list))
            chunks     = transcript.fetch()
            return " ".join(c.text for c in chunks)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"Transcript unavailable for {video_id}: {e}")
        return None

# ── Scoring ──────────────────────────────────────────────────────────────────
def score_engagement(views: int, likes: int) -> float:
    if views == 0:
        return 5.0
    ratio = likes / views
    return round(min(10.0, (ratio / 0.04) * 10), 2)


def score_sentiment(comments: list[str]) -> tuple[float, dict]:
    if not comments:
        return 5.0, {}
    labels = ["understanding", "confusion", "outdated", "irrelevant"]
    counts = {l: 0 for l in labels}
    for c in comments[:30]:
        result = classifier(c, candidate_labels=labels)
        counts[result["labels"][0]] += 1
    total = sum(counts.values()) or 1
    pcts  = {k: round(v / total * 100) for k, v in counts.items()}
    score = (
        counts["understanding"] * 1.0
        - counts["confusion"]   * 0.5
        - counts["outdated"]    * 0.4
        - counts["irrelevant"]  * 0.1
    ) / total * 10
    return round(max(0.0, min(10.0, score)), 2), pcts


def score_content(transcript: str | None, title: str) -> tuple[float, float, list[dict]]:
    if not transcript:
        return 5.0, 0.0, []
    words  = transcript.split()
    n      = len(words)
    if n < 50:
        return 5.0, 0.0, []
    window = max(1, n // 20)
    chunks = [" ".join(words[i:i + window]) for i in range(0, n - window, window)]
    if len(chunks) < 3:
        return 5.0, 0.0, []
    chunk_embs      = embedder.encode(chunks, convert_to_tensor=False)
    chunk_embs_norm = normalize(chunk_embs)
    # Lexical diversity
    all_words    = transcript.lower().split()
    lex_diversity = len(set(all_words)) / max(len(all_words), 1)
    lex_score    = min(10.0, lex_diversity * 55)
    # Information density
    consecutive_sims = [
        float(np.dot(chunk_embs_norm[i], chunk_embs_norm[i + 1]))
        for i in range(len(chunk_embs_norm) - 1)
    ]
    mean_sim      = float(np.mean(consecutive_sims))
    density_score = min(10.0, (1.0 - mean_sim) * 14)
    # Topic breadth
    n_clusters  = min(6, max(2, len(chunks) // 3))
    km          = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels_km   = km.fit_predict(chunk_embs_norm)
    breadth_score = min(10.0, (len(set(labels_km)) / 6) * 10)
    depth_score = round(lex_score * 0.40 + density_score * 0.40 + breadth_score * 0.20, 2)
    # Filler
    repetition_filler = [i for i, s in enumerate(consecutive_sims) if s > 0.82]
    dominant_cluster  = int(np.bincount(labels_km).argmax())
    dominant_centroid = km.cluster_centers_[dominant_cluster]
    dominant_centroid = dominant_centroid / (np.linalg.norm(dominant_centroid) + 1e-9)
    tangent_filler = [
        i for i, (emb, lbl) in enumerate(zip(chunk_embs_norm, labels_km))
        if lbl != dominant_cluster and float(np.dot(emb, dominant_centroid)) < 0.25
    ]
    filler_pct = round(len(set(repetition_filler) | set(tangent_filler)) / max(len(chunks), 1) * 100, 1)
    # Timestamps
    chunk_novelty = [0.0] + [1.0 - consecutive_sims[i - 1] for i in range(1, len(chunks))]
    top_indices   = sorted(sorted(range(len(chunk_novelty)), key=lambda i: chunk_novelty[i], reverse=True)[:5])
    timestamps    = [{"seconds": idx * window * 2, "label": chunks[idx][:60] + "…"} for idx in top_indices]
    return depth_score, filler_pct, timestamps


def compute_final_score(depth: float, sentiment: float, engagement: float, has_transcript: bool) -> float:
    if has_transcript:
        return round(depth * 0.4 + sentiment * 0.3 + engagement * 0.3, 2)
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
    if transcript:
        transcript_cache[video_id] = transcript
        save_cache()
        print(f"✅ Cached transcript for: {video_id}")
    engagement_score                    = score_engagement(meta["views"], meta["likes"])
    sentiment_score, comment_pcts       = score_sentiment(comments)
    depth_score, filler_pct, timestamps = score_content(transcript, meta["title"])
    final_score = compute_final_score(depth_score, sentiment_score, engagement_score, has_transcript=transcript is not None)
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

# ── CAG constants ─────────────────────────────────────────────────────────────
# ── CAG constants ─────────────────────────────────────────────────────────────
SENTENCE_SIM_THRESHOLD = 0.15
VIDEO_SIM_THRESHOLD    = 0.08

TOPIC_QUESTION_SIGNALS = [
    "is this video about", "does this video cover",
    "what does this video cover", "is this about",
    "does it cover", "what topics", "what aspects",
    "what does it talk about", "is this related to",
    "does this talk about", "tell me about this video",
    "what's this video about", "whats this video about",
    "what does this video discuss", "is this video on",
    "does the video cover", "what is covered",
    "what does the video talk about",
    "what is this vid abt", "what's this vid about",
]

# Separate signals for pure summary questions
SUMMARY_QUESTION_SIGNALS = [
    "what is this video about",
    "what is this vid about",
    "what is this about",
    "what is this video",
    "summarize this video",
    "summarize the video",
    "give me a summary",
    "what exactly is this video about",
    "tell me what this video is about",
    "what's this about",
    "whats this about",
]

SYSTEM_PROMPT_FACTUAL = """You are a strict video transcript assistant.
Answer ONLY using the transcript excerpts provided. Do not use outside knowledge.
If the answer is not in the excerpts, say exactly:
"This topic is not covered in the video you analyzed."
If the question is completely unrelated to the video subject, say exactly:
"This question is not related to the video you analyzed."
Keep answers concise and factual. No speculation."""

SYSTEM_PROMPT_TOPIC = """You are a strict video transcript assistant.
You will be given transcript excerpts from a video.
The user is asking whether a specific topic is covered in this video.
Answer using ONLY the excerpts provided.
Format:
- One sentence: yes or no, whether the topic is covered
- Bullet list of specific aspects covered (max 5 bullets)
If the topic is absent: "This topic is not covered in the video you analyzed."
Do not speculate or use outside knowledge."""

SYSTEM_PROMPT_SUMMARY = """You are a strict video transcript assistant.
You will be given transcript excerpts from a video.
Your job is to describe what this video is about using ONLY the excerpts.
Format your response as:
- One sentence describing the overall subject of the video
- Bullet list of the main topics or themes covered (max 5 bullets)
Base everything strictly on the excerpts. Do not guess or use outside knowledge."""


# ── Mean embedding cache ──────────────────────────────────────────────────────
_mean_emb_cache: dict[str, np.ndarray] = {}

def get_video_mean_embedding(video_id: str, sentences: list[str]) -> np.ndarray:
    if video_id in _mean_emb_cache:
        return _mean_emb_cache[video_id]
    embs     = embedder.encode(sentences, convert_to_tensor=False)
    mean_emb = np.mean(embs, axis=0)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    _mean_emb_cache[video_id] = mean_emb
    print(f"📐 Cached mean embedding for: {video_id}")
    return mean_emb


class AskRequest(BaseModel):
    video_id: str
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    transcript = transcript_cache.get(req.video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not available for this video.")

    # Spell correction
    spell        = build_spell_checker(transcript)
    req.question = correct_query(req.question, spell)

    # Split transcript into sentences
    sentences = [
        s.strip()
        for s in transcript.replace("\n", " ").split(".")
        if len(s.strip()) > 30
    ]
    if not sentences:
        return JSONResponse({"answer": "No transcript content available."})

    # Embed question
    q_emb  = embedder.encode(req.question, convert_to_tensor=True)
    s_embs = embedder.encode(sentences,    convert_to_tensor=True)
    sims   = util.cos_sim(q_emb, s_embs)[0]

    q_lower = req.question.lower().strip()

    # ── Detect question type ──────────────────────────────────────────────────
    is_summary = any(sig in q_lower for sig in SUMMARY_QUESTION_SIGNALS)
    is_topic   = (not is_summary) and any(sig in q_lower for sig in TOPIC_QUESTION_SIGNALS)

    # Semantic anchor fallback for anything not caught by signals
    if not is_summary and not is_topic:
        anchor     = "what is this video about what topics does it cover"
        anchor_emb = embedder.encode(anchor, convert_to_tensor=True)
        topic_sim  = float(util.cos_sim(anchor_emb, q_emb))
        if topic_sim > 0.50:
            is_summary = True   # treat ambiguous meta-questions as summary
            print(f"📌 Summary path via semantic anchor (sim={topic_sim:.2f})")
        else:
            print(f"💬 Factual path (anchor_sim={topic_sim:.2f})")

    # ── Summary path ──────────────────────────────────────────────────────────
    if is_summary:
        # Give more context — 12 evenly spaced sentences
        step    = max(1, len(sentences) // 12)
        indices = list(range(0, len(sentences), step))[:12]
        context = " ".join(sentences[i] for i in indices)
        system  = SYSTEM_PROMPT_SUMMARY
        print(f"📋 Summary path | sentences={len(indices)}")

    # ── Topic confirmation path ───────────────────────────────────────────────
    elif is_topic:
        q_emb_np   = q_emb.cpu().numpy() if hasattr(q_emb, "cpu") else np.array(q_emb)
        q_norm     = q_emb_np / (np.linalg.norm(q_emb_np) + 1e-9)
        video_mean = get_video_mean_embedding(req.video_id, sentences)
        video_sim  = float(np.dot(q_norm, video_mean))
        print(f"🎯 Topic path | video_sim={video_sim:.3f} | threshold={VIDEO_SIM_THRESHOLD}")
        if video_sim < VIDEO_SIM_THRESHOLD:
            return JSONResponse({"answer": "This question is not related to the video you analyzed."})
        step    = max(1, len(sentences) // 8)
        indices = list(range(0, len(sentences), step))[:8]
        context = " ".join(sentences[i] for i in indices)
        system  = SYSTEM_PROMPT_TOPIC

    # ── Factual path ──────────────────────────────────────────────────────────
    else:
        best_sim = float(sims.max())
        print(f"🔍 Factual path | best_sim={best_sim:.3f} | threshold={SENTENCE_SIM_THRESHOLD}")
        if best_sim < SENTENCE_SIM_THRESHOLD:
            return JSONResponse({"answer": "This question is not related to the video you analyzed."})
        top_indices = sorted(sims.topk(min(5, len(sentences))).indices.tolist())
        context     = " ".join(sentences[i] for i in top_indices)
        system      = SYSTEM_PROMPT_FACTUAL

    # ── Groq ──────────────────────────────────────────────────────────────────
    try:
        chat = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"TRANSCRIPT EXCERPTS:\n{context}\n\nQUESTION: {req.question}"},
            ],
            temperature=0.0,
            max_tokens=350,
        )
        answer = chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed.")

    return JSONResponse({"answer": answer})