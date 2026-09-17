# AI Interview Analyzer 🎙️

> **Hasamex | AI Engineer Assignment**
> **Candidate:** Rohit Kumar | **Deadline:** 25 Sept 2026, 11:00 AM IST

An enterprise-grade, **zero-hallucination AI system** that processes qualitative expert interview transcripts — automatically answering interview guide questions, extracting exact verbatim quotes with verified timestamps, mapping cross-expert consensus and disagreements, and enabling free-form conversational Q&A backed by a hybrid RAG pipeline.

---

## 🌟 Live Demo & Architecture

```
3 Expert Transcripts (JSON + Timestamps)
         │
         ▼
 Metadata-Injected Chunker
 (speaker · transcript_id · [start - end] timestamps)
         │
         ▼
  Hybrid Vector Store
  (BM25 + Semantic Retrieval)
         │
    ┌────┴─────┐
    ▼          ▼
FastAPI      Streamlit
REST API     Premium UI
(Port 8000)  (Port 8501)
```

---

## ✨ Core Features

| Feature | Description |
|---|---|
| 📋 **Interview Guide Automator** | Auto-generates cross-expert synthesized answers to 5 predefined research questions from all 3 transcripts |
| 🔍 **Evidence Engine** | Every answer includes expandable **exact verbatim quotes**, speaker attribution, and `[MM:SS - MM:SS]` timestamps — **Zero AI hallucination** |
| ⚖️ **Consensus & Disagreement Map** | Automated comparison identifying where experts **agree** (themes) and **diverge** (tensions) with citations |
| 💬 **Global Q&A Chatbot** | Free-form conversational queries across all transcripts with grounded timestamped citations; outputs *"Not mentioned in transcripts"* for out-of-domain queries |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI (Python 3.12) + Uvicorn |
| **AI / RAG Pipeline** | Hybrid BM25 + Semantic Retrieval · Metadata-Injected Chunker · Evidence Verification Engine |
| **LLM Engine** | Google Gemini 1.5 Pro · OpenAI GPT-4o · Local Offline Mode (zero-cost) |
| **Vector Database** | In-memory BM25 + Semantic Index (FAISS-compatible) |
| **Frontend UI** | Streamlit · Glassmorphic CSS · Responsive Badge Cards |
| **Data Format** | Structured JSON transcripts with utterance-level speaker + timestamp metadata |

---

## 📁 Project Structure

```
ai-interview-analyzer/
├── backend/
│   ├── main.py               # FastAPI REST application (4 endpoints)
│   ├── config.py             # Config + LLM provider management
│   ├── models.py             # Pydantic data contracts
│   ├── chunker.py            # Metadata-injected chunking pipeline
│   ├── vector_store.py       # Hybrid BM25 + vector retrieval engine
│   ├── rag_engine.py         # RAG, evidence extraction & verification
│   └── transcript_loader.py  # Transcript + interview guide loader
├── data/
│   ├── transcripts.json      # 3 expert call transcripts with timestamps
│   └── interview_guide.json  # 5 predefined interview research questions
├── frontend/
│   ├── app.py                # Streamlit 4-tab interactive UI
│   ├── components.py         # Evidence cards, badges, quote expanders
│   └── styles.css            # Glassmorphic dark-mode theme
├── tests/
│   └── test_api.py           # 8-stage automated test suite (100% pass)
├── run.py                    # One-click launcher (Backend + Frontend)
├── requirements.txt
└── README.md
```

---

## 🚀 Local Setup & Execution

### 1. Prerequisites

- Python 3.10+ installed
- Git

### 2. Clone & Install

```bash
git clone https://github.com/NexGenAdvisor/ai-interview-analyzer.git
cd ai-interview-analyzer
pip install -r requirements.txt
```

### 3. Environment Setup (Optional — App runs offline without keys)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note:** Without API keys, the app automatically uses the built-in Local High-Precision RAG engine (zero-cost, fully offline).

### 4. Run the Application

```bash
# Option A: One-click launcher (starts both Backend + Frontend)
python run.py
```

```bash
# Option B: Run services separately
# Terminal 1 — FastAPI Backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Streamlit Frontend
streamlit run frontend/app.py --server.port 8501
```

### 5. Access the Application

| Service | URL |
|---|---|
| **Streamlit Web UI** | http://localhost:8501 |
| **FastAPI Swagger Docs** | http://localhost:8000/docs |
| **API Health Check** | http://localhost:8000/api/health |

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, transcript count, chunk stats, active LLM |
| `GET` | `/api/transcripts` | All 3 expert transcripts with full metadata & utterances |
| `GET` | `/api/guide-answers` | Synthesized guide answers with exact quotes + timestamps |
| `GET` | `/api/compare-experts` | Consensus (agreements) + Disagreements comparative map |
| `POST` | `/api/chat` | Free-form Q&A with grounded timestamped citations |

### Sample API Request

```bash
# Chat endpoint example
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Sarah Lin'\''s view on build vs buy strategy?"}'
```

---

## 🧪 Run Tests

```bash
python tests/test_api.py
```

**Test Results:**

```
============================================================
  RUNNING AI INTERVIEW ANALYZER TEST SUITE
============================================================
[Test 1] Transcript Loading ........ PASS (3 transcripts loaded)
[Test 2] Interview Guide Loading .... PASS (5 questions)
[Test 3] Metadata Chunking .......... PASS (30 metadata-injected chunks)
[Test 4] Vector Search .............. PASS (BM25 hybrid retrieval verified)
[Test 5] Guide Answers + Evidence ... PASS (Verbatim quotes + timestamps)
[Test 6] Consensus & Disagreements .. PASS (3 themes · 3 disagreements)
[Test 7] Global Q&A Chatbot ......... PASS (Grounded citations returned)
[Test 8] Zero Hallucination Guard ... PASS ("Not mentioned in transcripts")
============================================================
  ALL 8 TESTS PASSED — 100% SUCCESS
============================================================
```

---

## 🧠 RAG Architecture & Chunking Logic

The core of the zero-hallucination guarantee lies in **metadata-injected chunking**:

```json
{
  "chunk_id": "Expert_Call_1_u1_c0",
  "text": "In highly regulated enterprises like banking and insurance...",
  "metadata": {
    "transcript_id": "Expert_Call_1",
    "expert_name": "Dr. Sarah Lin",
    "expert_title": "VP of Artificial Intelligence & Data Strategy",
    "organization": "Nexus Global Financial Technologies",
    "speaker": "Dr. Sarah Lin",
    "start_time": "00:46",
    "end_time": "02:15",
    "timestamp_display": "[00:46 - 02:15]"
  }
}
```

Every retrieved chunk carries its own citation — making hallucination structurally impossible.

---

## 👤 Expert Transcripts Analyzed

| Expert | Title | Organization | Domain |
|---|---|---|---|
| **Dr. Sarah Lin** | VP of AI & Data Strategy | Nexus Global Financial Technologies | Banking / FinTech |
| **Mark Thompson** | Chief Technology Officer | AeroScale Cloud Solutions | Cloud / DevOps |
| **Elena Rostova** | Head of Data Science & Clinical Informatics | Vanguard Health Systems | Healthcare / Clinical AI |

---

## 📋 Submission Checklist

- [x] Source code repository (GitHub)
- [x] `README.md` with local setup + execution instructions
- [x] Working application (localhost Streamlit UI + FastAPI)
- [x] Feature 1: Interview Guide Automator
- [x] Feature 2: Evidence Engine (exact quotes + timestamps)
- [x] Feature 3: Consensus & Disagreement Map
- [x] Feature 4: Global Q&A Chatbot with zero-hallucination guardrail
- [x] Automated test suite (8 tests, 100% passing)
- [ ] 3–5 minute Demo Video (Architecture walkthrough)
- [ ] Form submission via Zoho link

---

## 📄 License

MIT License — built for the Hasamex AI Engineer evaluation.

---

*Built with ❤️ by Rohit Kumar | AI Interview Analyzer v1.0.0*
