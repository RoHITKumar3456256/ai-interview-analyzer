import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from backend.config import settings
from backend.models import (
    HealthResponse, GuideAnswersResponse, CompareExpertsResponse,
    ChatRequest, ChatResponse, Transcript
)
from backend.transcript_loader import transcript_loader
from backend.chunker import chunker
from backend.vector_store import vector_store
from backend.rag_engine import rag_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data and vector store on module load
try:
    logger.info("Initializing AI Interview Analyzer backend...")
    transcripts = transcript_loader.get_transcripts()
    chunks = chunker.chunk_all(transcripts)
    vector_store.build_index(chunks)
    logger.info(f"Loaded {len(transcripts)} transcripts and {len(chunks)} chunks into Vector Store.")
except Exception as err:
    logger.warning(f"Startup initialization notice: {err}")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="FastAPI Backend for AI Interview Analyzer"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status and index stats.
    """
    transcripts_list = transcript_loader.get_transcripts()
    chunks_list = vector_store.get_all_chunks()
    provider = rag_engine.get_active_provider()
    
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        transcripts_loaded=len(transcripts_list),
        total_chunks=len(chunks_list),
        llm_provider=provider
    )

@app.get("/api/transcripts", response_model=List[Transcript], tags=["Transcripts"])
def get_all_transcripts():
    """
    Returns all loaded expert call transcripts with full metadata and utterances.
    """
    return transcript_loader.get_transcripts()

@app.get("/api/guide-answers", response_model=GuideAnswersResponse, tags=["Analysis"])
def get_guide_answers():
    """
    Feature 1 & Feature 2: Interview Guide Automator & Evidence Engine.
    Returns synthesized answers for the predefined interview guide with exact quotes and timestamps.
    """
    try:
        response = rag_engine.generate_all_guide_answers()
        return response
    except Exception as e:
        logger.error(f"Error generating guide answers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate guide answers: {str(e)}"
        )

@app.get("/api/compare-experts", response_model=CompareExpertsResponse, tags=["Analysis"])
def compare_experts():
    """
    Feature 3: Consensus & Disagreement Map.
    Performs cross-transcript comparative analysis identifying Common Themes and Disagreements.
    """
    try:
        response = rag_engine.generate_consensus_and_disagreements()
        return response
    except Exception as e:
        logger.error(f"Error comparing experts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate expert comparison: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_transcripts(request: ChatRequest):
    """
    Feature 4: Global Q&A Chatbot.
    Accepts user query across transcripts, returns grounded answer with timestamped quote citations.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")
        
    try:
        response = rag_engine.chat_query(
            query=request.query,
            selected_transcripts=request.selected_transcripts,
            top_k=request.top_k or settings.TOP_K_CHUNKS
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat query execution failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
