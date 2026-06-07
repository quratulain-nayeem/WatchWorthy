---
title: WatchWorthy
emoji: ▶
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
---

# WatchWorthy

WatchWorthy is a YouTube tutorial quality evaluator. Paste a YouTube URL and it scores the video using transcript depth, viewer sentiment, and engagement quality before you commit to watching.

## What It Does

WatchWorthy analyzes videos across three main signals and returns a 0-10 quality score.

| Signal | Weight | Source |
|---|---:|---|
| Content Depth | 40% | Transcript analysis: lexical diversity, information density, topic breadth, filler detection |
| Viewer Sentiment | 30% | YouTube comments classified as helpful, confused, outdated, or off-topic |
| Engagement Quality | 30% | Likes-to-views ratio normalized to a 0-10 score |

When no transcript is available, the score falls back to 50% sentiment and 50% engagement.

## Features

- Animated score ring with sub-score bars
- Comment breakdown pills with representative examples
- Transcript-backed key timestamps
- Filler and repetition detection
- Transcript-grounded Q&A using CAG-style retrieval
- Gemini fallback model list for answer generation
- Quota/rate-limit stop behavior to avoid extra generation attempts
- Typo correction for the Q&A box and similar-video search
- Similar-video recommendations anchored to the analyzed video topic
- Relevance gate to block random unrelated recommendation searches

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML, CSS, JavaScript |
| Backend | FastAPI, Uvicorn |
| Comment Classifier | facebook/bart-large-mnli via Hugging Face Transformers |
| Embeddings | all-MiniLM-L6-v2 via Sentence Transformers |
| Generation | Gemini API, default models: gemini-2.5-flash-lite and gemini-2.5-flash |
| Transcript Fetch | youtube-transcript-api |
| Metadata/Search | YouTube Data API v3 |
| Deployment | Hugging Face Spaces with Docker |

## Environment Variables

Set these as Hugging Face Space secrets:

```text
YOUTUBE_API_KEY
GEMINI_API_KEY
GEMINI_MODELS
```

Recommended `GEMINI_MODELS` value:

```text
gemini-2.5-flash-lite,gemini-2.5-flash
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.
