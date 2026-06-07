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
from html import unescape

from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import google.generativeai as genai
from spellchecker import SpellChecker

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY  = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODELS   = [
    model.strip()
    for model in os.getenv(
        "GEMINI_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash-lite,gemini-2.0-flash",
    ).split(",")
    if model.strip()
]

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
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

print("Models ready.")

# ── Transcript Cache ─────────────────────────────────────────────────────────
CACHE_FILE = pathlib.Path("transcript_cache.json")

transcript_cache: dict[str, str] = (
    json.loads(CACHE_FILE.read_text())
    if CACHE_FILE.exists()
    else {}
)
video_profile_cache: dict[str, dict] = {}

def save_cache():
    CACHE_FILE.write_text(json.dumps(transcript_cache))

# ── Spellchecker ─────────────────────────────────────────────────────────────
QUERY_SLANG_WHITELIST = {
    "vid", "vids", "lol", "tbh", "imo", "btw", "ngl",
    "abt", "rn", "yt", "bro", "bru", "pls", "plz", "thx",
    "api", "apis", "async", "backend", "frontend", "javascript",
    "python", "react", "fastapi", "llm", "rag", "ai", "ml",
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
        print(f"Query corrected: '{text}' -> '{result}'")
    return result


def build_general_query_spell_checker(extra_text: str | None = None) -> SpellChecker:
    sc = SpellChecker()
    sc.word_frequency.load_words(QUERY_SLANG_WHITELIST)
    if extra_text:
        words = re.findall(r"[a-zA-Z]{3,}", extra_text.lower())
        sc.word_frequency.load_words(words)
    return sc


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "resourceexhausted" in text
        or "quota exceeded" in text
        or "rate limit" in text
        or "429" in text
    )


def keyword_text(text: str, limit: int = 24) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", (text or "").lower())
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "your", "you",
        "are", "was", "were", "have", "has", "had", "but", "not", "what",
        "how", "why", "who", "when", "where", "video", "videos", "about",
        "will", "can", "all", "just", "into", "than", "then", "them",
    }
    counts: dict[str, int] = {}
    for word in words:
        if word in stop or len(word) < 3:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda word: (-counts[word], word))
    return " ".join(ranked[:limit])


def build_video_profile(video_id: str, meta: dict, transcript: str | None, comments: list[str] | None) -> dict:
    comment_sample = " ".join((comments or [])[:20])
    source_text = " ".join([
        meta.get("title", ""),
        meta.get("channel", ""),
        keyword_text(transcript or "", limit=36),
        keyword_text(comment_sample, limit=24),
        (transcript or "")[:2400],
        comment_sample[:1200],
    ]).strip()
    profile = {
        "video_id": video_id,
        "title": meta.get("title", ""),
        "channel": meta.get("channel", ""),
        "source_text": source_text,
    }
    video_profile_cache[video_id] = profile
    return profile


def get_video_profile(video_id: str, fallback_title: str | None = None) -> dict:
    profile = video_profile_cache.get(video_id)
    if profile:
        return profile
    meta = fetch_video_metadata(video_id)
    if fallback_title and not meta.get("title"):
        meta["title"] = fallback_title
    transcript = transcript_cache.get(video_id, "")
    return build_video_profile(video_id, meta, transcript, comments=None)


def embedding_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    embs = embedder.encode([left, right], convert_to_tensor=False)
    a = embs[0] / (np.linalg.norm(embs[0]) + 1e-9)
    b = embs[1] / (np.linalg.norm(embs[1]) + 1e-9)
    return float(np.dot(a, b))


def source_intent_match(intent: str, profile: dict) -> tuple[bool, float]:
    source_text = profile.get("source_text", "")
    sim = embedding_similarity(intent, source_text)
    intent_terms = set(keyword_text(intent, limit=12).split())
    source_terms = set(keyword_text(source_text, limit=80).split())
    overlap = len(intent_terms & source_terms) / max(len(intent_terms), 1)
    score = (sim * 0.75) + (overlap * 0.25)
    return score >= 0.18, round(score, 3)

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


def search_youtube_videos(query: str, exclude_video_id: str | None = None, max_results: int = 8) -> list[dict]:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    videos = []
    for item in r.json().get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if not video_id or video_id == exclude_video_id:
            continue
        snippet = item.get("snippet", {})
        videos.append({
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        })
    return videos


def clean_comment(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


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
        clean_comment(i["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        for i in items
    ]


def normalize_transcript_chunks(chunks) -> list[dict]:
    normalized = []
    for c in chunks:
        text = getattr(c, "text", None)
        start = getattr(c, "start", None)
        duration = getattr(c, "duration", None)
        if isinstance(c, dict):
            text = c.get("text", text)
            start = c.get("start", start)
            duration = c.get("duration", duration)
        if text is None or start is None:
            continue
        normalized.append({
            "text": str(text).strip(),
            "start": float(start),
            "duration": float(duration or 0),
        })
    return [c for c in normalized if c["text"]]


def transcript_text(chunks: list[dict] | None) -> str | None:
    if not chunks:
        return None
    return " ".join(c["text"] for c in chunks)


def fetch_transcript(video_id: str) -> list[dict] | None:
    try:
        ytt             = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        try:
            chunks = transcript_list.find_manually_created_transcript(["en"]).fetch()
            return normalize_transcript_chunks(chunks)
        except Exception:
            pass
        try:
            chunks = transcript_list.find_generated_transcript(["en"]).fetch()
            return normalize_transcript_chunks(chunks)
        except Exception:
            pass
        try:
            transcript = next(iter(transcript_list))
            chunks     = transcript.fetch()
            return normalize_transcript_chunks(chunks)
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


COMMENT_LABELS = {
    "helpful": "helpful or understood",
    "confused": "confused or asking for clarification",
    "outdated": "outdated or no longer accurate",
    "off_topic": "off-topic or unrelated",
}


def score_sentiment(comments: list[str]) -> tuple[float, dict, dict]:
    if not comments:
        return 5.0, {}, {}
    candidate_labels = list(COMMENT_LABELS.values())
    label_lookup = {v: k for k, v in COMMENT_LABELS.items()}
    counts = {l: 0 for l in COMMENT_LABELS}
    examples = {l: [] for l in COMMENT_LABELS}
    for c in comments[:30]:
        result = classifier(c, candidate_labels=candidate_labels)
        label = label_lookup[result["labels"][0]]
        counts[label] += 1
        if len(examples[label]) < 6:
            examples[label].append(c)
    total = sum(counts.values()) or 1
    pcts  = {k: round(v / total * 100) for k, v in counts.items()}
    score = (
        counts["helpful"]   * 1.0
        - counts["confused"]  * 0.5
        - counts["outdated"]  * 0.4
        - counts["off_topic"] * 0.1
    ) / total * 10
    return round(max(0.0, min(10.0, score)), 2), pcts, examples


def build_timed_segments(transcript_chunks: list[dict], target_segments: int = 20) -> list[dict]:
    total_words = sum(len(c["text"].split()) for c in transcript_chunks)
    window = max(1, total_words // target_segments)
    segments = []
    current_words = []
    current_start = None
    for chunk in transcript_chunks:
        if current_start is None:
            current_start = chunk["start"]
        current_words.extend(chunk["text"].split())
        if len(current_words) >= window:
            segments.append({"start": current_start, "text": " ".join(current_words)})
            current_words = []
            current_start = None
    if current_words and current_start is not None:
        segments.append({"start": current_start, "text": " ".join(current_words)})
    return segments


def score_content(transcript_chunks: list[dict] | None, title: str) -> tuple[float, float, list[dict]]:
    if not transcript_chunks:
        return 5.0, 0.0, []
    transcript = transcript_text(transcript_chunks)
    words  = transcript.split()
    n      = len(words)
    if n < 50:
        return 5.0, 0.0, []
    segments = build_timed_segments(transcript_chunks)
    chunks = [s["text"] for s in segments]
    if len(segments) < 3:
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
    timestamps    = [
        {"seconds": int(segments[idx]["start"]), "label": chunks[idx][:60] + "…"}
        for idx in top_indices
    ]
    return depth_score, filler_pct, timestamps


def compute_final_score(depth: float, sentiment: float, engagement: float, has_transcript: bool) -> float:
    if has_transcript:
        return round(depth * 0.4 + sentiment * 0.3 + engagement * 0.3, 2)
    return round(sentiment * 0.5 + engagement * 0.5, 2)


def score_video(video_id: str) -> dict:
    meta = fetch_video_metadata(video_id)
    comments = fetch_comments(video_id, max_results=25)
    transcript_chunks = fetch_transcript(video_id)
    transcript = transcript_text(transcript_chunks)
    if transcript:
        transcript_cache[video_id] = transcript
        save_cache()
    engagement_score = score_engagement(meta["views"], meta["likes"])
    sentiment_score, _, _ = score_sentiment(comments)
    depth_score, filler_pct, _ = score_content(transcript_chunks, meta["title"])
    final_score = compute_final_score(
        depth_score,
        sentiment_score,
        engagement_score,
        has_transcript=transcript is not None,
    )
    return {
        **meta,
        "score": final_score,
        "depth": depth_score,
        "sentiment": sentiment_score,
        "engagement": engagement_score,
        "filler_pct": filler_pct,
        "has_transcript": transcript is not None,
        "transcript": transcript or "",
    }


def relevance_score(intent: str, title: str, transcript: str = "") -> float:
    target = f"{title}. {transcript[:1200]}".strip()
    if not intent or not target:
        return 5.0
    embs = embedder.encode([intent, target], convert_to_tensor=False)
    a = embs[0] / (np.linalg.norm(embs[0]) + 1e-9)
    b = embs[1] / (np.linalg.norm(embs[1]) + 1e-9)
    sim = float(np.dot(a, b))
    return round(max(0.0, min(10.0, (sim + 1.0) * 5.0)), 2)


def recommendation_blend(quality: float, relevance: float) -> float:
    return round(quality * 0.65 + relevance * 0.35, 2)


def estimated_quality_score(meta: dict) -> float:
    engagement = score_engagement(meta["views"], meta["likes"])
    comment_signal = min(10.0, np.log10(max(meta["comment_count"], 1) + 1) * 2.4)
    return round(engagement * 0.75 + comment_signal * 0.25, 2)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


class AnalyzeRequest(BaseModel):
    url: str


class RecommendRequest(BaseModel):
    video_id: str
    query: str | None = None
    title: str | None = None


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    meta       = fetch_video_metadata(video_id)
    comments   = fetch_comments(video_id)
    transcript_chunks = fetch_transcript(video_id)
    transcript = transcript_text(transcript_chunks)
    if transcript:
        transcript_cache[video_id] = transcript
        save_cache()
        print(f"Cached transcript for: {video_id}")
    build_video_profile(video_id, meta, transcript, comments)
    engagement_score                    = score_engagement(meta["views"], meta["likes"])
    sentiment_score, comment_pcts, comment_examples = score_sentiment(comments)
    depth_score, filler_pct, timestamps = score_content(transcript_chunks, meta["title"])
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
        "comment_examples": comment_examples,
        "timestamps":     timestamps,
        "has_transcript": transcript is not None,
    })


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    intent = (req.query or "").strip()
    if not intent:
        intent = (req.title or "").strip()
    if not intent:
        try:
            intent = fetch_video_metadata(req.video_id)["title"]
        except Exception:
            raise HTTPException(status_code=400, detail="Add a search intent or analyze a video first.")
    spell = build_general_query_spell_checker(req.title)
    intent = correct_query(intent, spell)
    source_profile = get_video_profile(req.video_id, req.title)
    source_ok, source_match = source_intent_match(intent, source_profile)
    if not source_ok:
        return JSONResponse({
            "intent": intent,
            "source_match": source_match,
            "message": "That search is not relevant to the video you analyzed.",
            "recommendations": [],
        })
    source_title = source_profile.get("title") or req.title or ""
    search_intent = f"{source_title} {intent}".strip()

    try:
        search_results = search_youtube_videos(search_intent, exclude_video_id=req.video_id, max_results=8)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube recommendation search failed: {e}")

    recommendations = []
    for result in search_results:
        try:
            meta = fetch_video_metadata(result["video_id"])
            quality = estimated_quality_score(meta)
            relevance = relevance_score(search_intent, meta["title"])
            source_relevance = relevance_score(source_profile.get("source_text", ""), meta["title"])
            if relevance < 5.4 or source_relevance < 5.4:
                print(f"Recommendation filtered as off-topic: {meta['title']}")
                continue
            recommendations.append({
                "video_id": result["video_id"],
                "title": meta["title"] or result["title"],
                "channel": meta["channel"] or result["channel"],
                "thumbnail": meta["thumbnail"] or result["thumbnail"],
                "score": quality,
                "relevance": relevance,
                "source_relevance": source_relevance,
                "recommendation_score": recommendation_blend(quality, relevance),
                "url": f"https://www.youtube.com/watch?v={result['video_id']}",
            })
        except Exception as e:
            print(f"Recommendation skipped for {result['video_id']}: {e}")

    recommendations.sort(key=lambda item: item["recommendation_score"], reverse=True)
    return JSONResponse({
        "intent": intent,
        "recommendations": recommendations[:3],
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
    print(f"Cached mean embedding for: {video_id}")
    return mean_emb


def compact_context(selected_sentences: list[str], max_chars: int = 4500) -> str:
    kept = []
    total = 0
    for sentence in selected_sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned:
            continue
        extra = len(cleaned) + 1
        if kept and total + extra > max_chars:
            break
        kept.append(cleaned)
        total += extra
    return " ".join(kept)


def generate_answer_with_gemini(system: str, context: str, question: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is missing. Add GEMINI_API_KEY to your .env and restart Uvicorn.")
    if GEMINI_API_KEY.startswith("gsk_"):
        raise RuntimeError("GEMINI_API_KEY contains a Groq key. Replace it with a Gemini key from Google AI Studio.")
    if not GEMINI_MODELS:
        raise RuntimeError("No Gemini models configured. Add GEMINI_MODELS to your .env and restart Uvicorn.")
    prompt = (
        f"{system}\n\n"
        f"TRANSCRIPT EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    errors = []
    for model_name in GEMINI_MODELS:
        try:
            print(f"Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 350,
                },
            )
            answer = (getattr(response, "text", "") or "").strip()
            if answer:
                print(f"Gemini answer generated with: {model_name}")
                return answer
            errors.append(f"{model_name}: empty response")
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            print(f"Gemini model failed ({model_name}): {e}")
            if is_quota_error(e):
                raise RuntimeError(
                    "Gemini free quota or rate limit reached. Answer generation has been stopped to avoid extra usage. "
                    "Try again later or switch GEMINI_MODELS to a model with available quota."
                ) from e
    raise RuntimeError("All Gemini models failed. " + " | ".join(errors))


class AskRequest(BaseModel):
    video_id: str
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    transcript = transcript_cache.get(req.video_id)

    if not transcript:
        chunks = fetch_transcript(req.video_id)
        transcript = transcript_text(chunks)

        if transcript:
            transcript_cache[req.video_id] = transcript
            save_cache()
        else:
            raise HTTPException(
                status_code=404,
                detail="Transcript not available for this video."
            )

    # Spell correction
    spell = build_spell_checker(transcript)
    req.question = correct_query(req.question, spell)

    ...

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
            print(f"Summary path via semantic anchor (sim={topic_sim:.2f})")
        else:
            print(f"Factual path (anchor_sim={topic_sim:.2f})")

    # ── Summary path ──────────────────────────────────────────────────────────
    if is_summary:
        step    = max(1, len(sentences) // 8)
        indices = list(range(0, len(sentences), step))[:8]
        context = compact_context([sentences[i] for i in indices], max_chars=3500)
        system  = SYSTEM_PROMPT_SUMMARY
        print(f"Summary path | sentences={len(indices)}")

    # ── Topic confirmation path ───────────────────────────────────────────────
    elif is_topic:
        q_emb_np   = q_emb.cpu().numpy() if hasattr(q_emb, "cpu") else np.array(q_emb)
        q_norm     = q_emb_np / (np.linalg.norm(q_emb_np) + 1e-9)
        video_mean = get_video_mean_embedding(req.video_id, sentences)
        video_sim  = float(np.dot(q_norm, video_mean))
        print(f"Topic path | video_sim={video_sim:.3f} | threshold={VIDEO_SIM_THRESHOLD}")
        if video_sim < VIDEO_SIM_THRESHOLD:
            return JSONResponse({"answer": "This question is not related to the video you analyzed."})
        step    = max(1, len(sentences) // 6)
        indices = list(range(0, len(sentences), step))[:6]
        context = compact_context([sentences[i] for i in indices], max_chars=3200)
        system  = SYSTEM_PROMPT_TOPIC

    # ── Factual path ──────────────────────────────────────────────────────────
    else:
        best_sim = float(sims.max())
        print(f"Factual path | best_sim={best_sim:.3f} | threshold={SENTENCE_SIM_THRESHOLD}")
        if best_sim < SENTENCE_SIM_THRESHOLD:
            return JSONResponse({"answer": "This question is not related to the video you analyzed."})
        top_indices = sorted(sims.topk(min(5, len(sentences))).indices.tolist())
        context     = compact_context([sentences[i] for i in top_indices], max_chars=3200)
        system      = SYSTEM_PROMPT_FACTUAL

    # ── Gemini ────────────────────────────────────────────────────────────────
    try:
        answer = generate_answer_with_gemini(system, context, req.question)
    except Exception as e:
        error_text = str(e)
        print(f"Gemini error: {error_text}")
        if "api key" in error_text.lower():
            detail = "Gemini API key is missing or invalid. Check GEMINI_API_KEY in your .env and restart Uvicorn."
        elif "quota" in error_text.lower() or "rate limit" in error_text.lower():
            detail = error_text
        else:
            detail = f"Answer generation failed: {error_text}"
        return JSONResponse(
            status_code=500,
            content={"detail": detail},
        )

    return JSONResponse({"answer": answer})
