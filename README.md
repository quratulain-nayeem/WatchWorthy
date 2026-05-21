# WatchWorthy — YouTube Tutorial Quality Evaluator

> Is this tutorial worth your time? Paste any YouTube URL and get a quality score based on transcript depth, viewer sentiment, and engagement — before you commit to watching.

---

## What It Does

WatchWorthy analyzes YouTube videos across three dimensions and returns a composite quality score from 0–10. It is designed to cut through popularity bias — a video with 10 million views can still be a bad tutorial, and WatchWorthy will tell you so.

### Scoring Model

| Signal | Weight | Source |
|---|---|---|
| Content Depth | 40% | Transcript — lexical diversity, information density, topic breadth via KMeans clustering |
| Viewer Sentiment | 30% | Comments — zero-shot classification into understanding / confusion / outdated / irrelevant |
| Engagement Quality | 30% | Likes-to-views ratio, normalized to 0–10 |

When no transcript is available, weights redistribute to 50% sentiment / 50% engagement and the tool continues gracefully.

### Features

- Animated score ring with live sub-score bars
- Comment breakdown pills — click any category to see representative comments
- Transcript-backed key timestamps — jump to the most information-dense moments
- Filler detection — percentage of off-topic or repetitive content
- CAG-powered Q&A — ask anything about the video and get answers grounded strictly in the transcript, no hallucination
- Spell correction on queries using a transcript-aware dictionary

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML, CSS, JavaScript — custom dark theme, canvas background |
| Backend | FastAPI + Uvicorn |
| Comment Classifier | facebook/bart-large-mnli (zero-shot, Hugging Face Transformers) |
| Sentence Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Q&A Generation | Groq API — llama-3.1-8b-instant |
| Transcript Fetch | youtube-transcript-api (no key required) |
| Video Metadata | YouTube Data API v3 |
| Deployment | Hugging Face Spaces |

---

## Project Structure
