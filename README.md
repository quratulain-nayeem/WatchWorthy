---
title: WatchWorthy
emoji: "WW"
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
---

# WatchWorthy

WatchWorthy is a YouTube tutorial quality evaluator. It helps users decide whether a video is worth watching by combining transcript analysis, comment sentiment, engagement quality, transcript-grounded Q&A, and topic-safe recommendations.

Live Space: https://huggingface.co/spaces/quratulainnnnn/WatchWorthy

GitHub: https://github.com/quratulain-nayeem/WatchWorthy

## Project Goals

- Score YouTube tutorials before the user spends time watching them.
- Reduce popularity bias by looking beyond views and thumbnails.
- Answer questions only from the analyzed video's transcript.
- Recommend similar quality videos only when the user's request is relevant to the analyzed video.
- Handle minor user typos in both Q&A and recommendation search.
- Stop generation on quota/rate-limit errors to avoid unnecessary API usage.

## High-Level Architecture

```text
Browser UI
  |
  | fetch("/analyze"), fetch("/ask"), fetch("/recommend")
  v
FastAPI backend
  |
  |-- YouTube Data API v3: metadata, comments, search
  |-- youtube-transcript-api: transcript extraction
  |-- Transformers: comment sentiment classifier
  |-- Sentence Transformers: semantic similarity and retrieval
  |-- Gemini API: transcript-grounded answer generation
```

The project is currently deployed on Hugging Face Spaces using Docker. The Docker container runs FastAPI with Uvicorn on port `7860`, which is the port expected by Hugging Face Spaces.

## Frontend

File: `index.html`

The frontend is plain HTML, CSS, and JavaScript. There is no frontend framework.

Main frontend responsibilities:

- Render the landing/analyze UI.
- Animate the canvas background, waveform, score cards, and score ring.
- Submit YouTube URLs to `/analyze`.
- Display the final quality score and sub-scores.
- Display comment breakdown pills and representative comments.
- Display key timestamps and filler score.
- Submit transcript-grounded questions to `/ask`.
- Submit similar-video search requests to `/recommend`.
- Show loading states, validation errors, and recommendation status messages.

Important UI behavior:

- The main URL input validates YouTube URLs before sending them to the backend.
- The "Generating score" label appears only during analysis.
- The background vignette softly pulses only while analysis is running.
- The recommendation UI displays a clear "not relevant" message when the user's search does not match the analyzed video topic.

## Backend

File: `app.py`

The backend is a FastAPI app with three primary routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/analyze` | POST | Analyzes a YouTube video and returns score data |
| `/ask` | POST | Answers questions using transcript-grounded CAG logic |
| `/recommend` | POST | Finds similar quality videos anchored to the analyzed video |

Backend responsibilities:

- Extract YouTube video IDs from multiple URL formats.
- Fetch video metadata with the YouTube Data API.
- Fetch relevant comments with the YouTube Data API.
- Fetch transcripts using `youtube-transcript-api`.
- Cache transcripts locally in `transcript_cache.json`.
- Score engagement, sentiment, and content depth.
- Build a source-video profile for topic-safe recommendations.
- Correct minor spelling mistakes in user queries.
- Generate answers through Gemini with model fallback.
- Stop generation when Gemini quota or rate limits are reached.

## Scoring Logic

WatchWorthy returns a final score from `0` to `10`.

### Default Weights

| Signal | Weight | Function | Description |
|---|---:|---|---|
| Content Depth | 40% | `score_content()` | Uses transcript structure, lexical diversity, density, topic breadth, filler detection |
| Viewer Sentiment | 30% | `score_sentiment()` | Classifies comments into helpful, confused, outdated, and off-topic |
| Engagement Quality | 30% | `score_engagement()` | Scores likes/views ratio |

### No-Transcript Fallback

If no transcript is available, the score uses:

```text
50% viewer sentiment
50% engagement quality
```

This lets the app still return a useful score instead of failing completely.

## Content Depth Model

The content-depth score uses transcript chunks and embeddings.

Signals used:

- Lexical diversity: more varied vocabulary generally indicates richer content.
- Information density: compares consecutive transcript segment embeddings to estimate repetition.
- Topic breadth: uses KMeans clustering on normalized transcript embeddings.
- Filler detection: flags repetitive or weakly related transcript segments.
- Key timestamps: selects high-novelty transcript segments as jump points.

Main libraries:

- `sentence-transformers`
- `all-MiniLM-L6-v2`
- `scikit-learn` KMeans
- `numpy`

## Comment Sentiment Model

Comments are classified with:

```text
facebook/bart-large-mnli
```

The app uses zero-shot classification with these labels:

| Internal Label | Meaning |
|---|---|
| `helpful` | Helpful or understood |
| `confused` | Confused or asking for clarification |
| `outdated` | Outdated or no longer accurate |
| `off_topic` | Off-topic or unrelated |

The sentiment score rewards helpful comments and penalizes confused, outdated, or unrelated comments.

## CAG / Transcript-Grounded Q&A

The `/ask` route uses a CAG-style flow: the answer is generated from selected transcript context rather than from the model's outside knowledge.

Flow:

1. Load the transcript from cache for the analyzed video.
2. Correct minor spelling mistakes in the question.
3. Split the transcript into candidate sentences.
4. Embed the user question and transcript sentences.
5. Detect whether the question is:
   - a summary question,
   - a topic-confirmation question,
   - or a factual question.
6. Select transcript context:
   - Summary questions use sampled transcript coverage.
   - Topic questions compare the question against the full video embedding.
   - Factual questions use top-matching transcript sentences.
7. Send only the selected transcript excerpts to Gemini.
8. Instruct Gemini to answer strictly from the excerpts.

If the question is unrelated, the app returns:

```text
This question is not related to the video you analyzed.
```

If the topic is absent from the selected excerpts, the model is instructed to say:

```text
This topic is not covered in the video you analyzed.
```

## Gemini Model Setup

The app uses the Gemini API for answer generation.

Default model fallback list:

```text
gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash-lite,gemini-2.0-flash
```

For deployment, the recommended Hugging Face secret is:

```text
GEMINI_MODELS=gemini-2.5-flash-lite,gemini-2.5-flash
```

The app reads:

```text
GEMINI_API_KEY
GEMINI_MODELS
```

If a model fails for a normal model-specific reason, the app tries the next model. If Gemini returns a quota, rate-limit, or `429` error, the app stops immediately to avoid extra usage.

## Typo Handling

The app uses `pyspellchecker` to correct minor user typos.

### Q&A Box

For `/ask`, the spell checker is built from the analyzed transcript. This helps preserve domain-specific terms from the video and avoids over-correcting technical language.

### Similar Quality Videos Box

For `/recommend`, the app uses a general query spell checker with a whitelist of common slang and technical terms, including:

```text
api, async, backend, frontend, javascript, python, react, fastapi, llm, rag, ai, ml
```

This prevents corrections from damaging useful technical terms.

## Recommendation Logic

The recommendation system is designed to avoid random video generation.

It uses:

- analyzed video title,
- channel name,
- transcript keywords,
- comment keywords,
- transcript excerpts,
- comment excerpts,
- semantic embeddings,
- YouTube search results,
- candidate video metadata,
- quality score,
- relevance score.

### Source-Video Profile

When a video is analyzed, the backend builds a source profile with:

```text
title + channel + transcript keywords + comment keywords + transcript sample + comment sample
```

This profile is cached in memory and used by `/recommend`.

### Relevance Gate

Before searching YouTube, the user query is compared against the source-video profile.

If the user analyzed a tech video and types an unrelated query such as `neet`, the backend returns:

```text
That search is not relevant to the video you analyzed.
```

This prevents unrelated suggestions from being shown.

### Anchored YouTube Search

If the query is relevant, the backend searches YouTube with:

```text
source video title + user intent
```

This keeps search results close to the original analyzed topic.

### Candidate Filtering

Each candidate recommendation is filtered again using:

- relevance to the combined search intent,
- relevance to the source-video profile,
- estimated quality score.

Only the top three ranked recommendations are returned.

## Restrictions and Safety Guards

The app includes several restrictions to keep outputs grounded:

- Q&A is only allowed after a transcript is available.
- Unrelated Q&A is rejected using semantic thresholds.
- Gemini receives only transcript excerpts, not the whole internet.
- Gemini prompts explicitly forbid guessing and outside knowledge.
- Recommendation search is blocked if the user query is unrelated to the analyzed video.
- Candidate recommendations are filtered against the analyzed video's source profile.
- Generation stops on quota/rate-limit errors.
- `.env`, venvs, logs, pyc files, and transcript cache are ignored by Git.

## Deployment

### GitHub

The main source code is pushed to GitHub:

```text
https://github.com/quratulain-nayeem/WatchWorthy
```

### Hugging Face Spaces

The app is deployed as a Docker Space:

```text
https://huggingface.co/spaces/quratulainnnnn/WatchWorthy
```

Required Hugging Face Space secrets:

```text
YOUTUBE_API_KEY
GEMINI_API_KEY
GEMINI_MODELS
```

The Dockerfile runs:

```text
uvicorn app:app --host 0.0.0.0 --port 7860
```

Hugging Face Spaces expects the app to listen on port `7860`.

## Docker

File: `Dockerfile`

The Docker image:

1. Starts from `python:3.11-slim`.
2. Sets `/app` as the working directory.
3. Installs dependencies from `requirements.txt`.
4. Copies `app.py` and `index.html`.
5. Exposes port `7860`.
6. Starts Uvicorn.

## Local Development

Create or activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Local `.env` example:

```text
YOUTUBE_API_KEY=your_youtube_key
GEMINI_API_KEY=your_gemini_key
GEMINI_MODELS=gemini-2.5-flash-lite,gemini-2.5-flash
```

Do not commit `.env`.

## Git Ignore Policy

The repository ignores:

```text
.env
.venv/
.venv-1/
__pycache__/
*.pyc
*.log
transcript_cache.json
.vscode/
```

This keeps API keys, local virtual environments, generated logs, and cached transcripts out of Git.

## Chrome Extension Roadmap

The next planned step is a Chrome extension.

Potential extension architecture:

- Content script detects the current YouTube video URL.
- Popup UI shows an "Analyze with WatchWorthy" action.
- Extension sends the video URL to the deployed Hugging Face backend.
- Backend returns the same analysis JSON used by the web app.
- Popup displays score, sub-scores, Q&A, and recommendation shortcuts.

The extension should not store API keys in browser code. API keys should remain server-side in Hugging Face Space secrets.

## License

Released under the MIT License.
