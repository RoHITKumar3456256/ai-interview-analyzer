from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Utterance(BaseModel):
    speaker: str
    start_time: str
    end_time: str
    text: str

class Transcript(BaseModel):
    id: str
    title: str
    expert_name: str
    expert_title: str
    organization: str
    duration: str
    date: str
    summary: str
    utterances: List[Utterance]

class ChunkMetadata(BaseModel):
    transcript_id: str
    expert_name: str
    expert_title: str
    organization: str
    speaker: str
    start_time: str
    end_time: str
    timestamp_display: str
    chunk_index: int
    topic_hint: Optional[str] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

class EvidenceQuote(BaseModel):
    quote: str
    speaker: str
    timestamp: str  # e.g., "[00:46 - 02:15]"
    transcript_id: str
    expert_title: Optional[str] = None
    organization: Optional[str] = None
    relevance_score: Optional[float] = 1.0

class ExpertPerspective(BaseModel):
    expert_name: str
    expert_title: str
    organization: str
    stance_summary: str
    quotes: List[EvidenceQuote]

class GuideQuestionAnswer(BaseModel):
    question_id: str
    topic: str
    question_text: str
    synthesized_answer: str
    expert_perspectives: List[ExpertPerspective]
    evidence_quotes: List[EvidenceQuote]
    grounding_score: float = 1.0
    zero_hallucination_verified: bool = True

class GuideAnswersResponse(BaseModel):
    guide_title: str
    total_questions: int
    transcripts_analyzed: List[str]
    answers: List[GuideQuestionAnswer]

class ConsensusPoint(BaseModel):
    theme: str
    summary: str
    supporting_experts: List[str]
    evidence_quotes: List[EvidenceQuote]

class DisagreementPoint(BaseModel):
    topic: str
    tension_summary: str
    perspectives: List[Dict[str, Any]]
    evidence_quotes: List[EvidenceQuote]

class CompareExpertsResponse(BaseModel):
    study_topic: str
    total_experts: int
    expert_names: List[str]
    consensus_themes: List[ConsensusPoint]
    disagreements: List[DisagreementPoint]
    executive_synthesis: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    selected_transcripts: Optional[List[str]] = None
    top_k: Optional[int] = 6

class ChatResponse(BaseModel):
    query: str
    answer: str
    evidence_quotes: List[EvidenceQuote]
    cited_speakers: List[str]
    is_grounded: bool
    source_chunks: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    transcripts_loaded: int
    total_chunks: int
    llm_provider: str
