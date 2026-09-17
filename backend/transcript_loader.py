import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.models import Transcript, Utterance

class TranscriptLoader:
    def __init__(self, transcripts_path: Optional[Path] = None, guide_path: Optional[Path] = None):
        self.transcripts_path = transcripts_path or settings.TRANSCRIPTS_PATH
        self.guide_path = guide_path or settings.INTERVIEW_GUIDE_PATH
        self._transcripts: List[Transcript] = []
        self._guide: Dict[str, Any] = {}
        self.load_all()
        
    def load_all(self):
        self.load_transcripts()
        self.load_guide()
        
    def load_transcripts(self) -> List[Transcript]:
        if not self.transcripts_path.exists():
            raise FileNotFoundError(f"Transcripts file not found at: {self.transcripts_path}")
            
        with open(self.transcripts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        raw_list = data.get("transcripts", [])
        self._transcripts = [Transcript(**item) for item in raw_list]
        return self._transcripts

    def load_guide(self) -> Dict[str, Any]:
        if not self.guide_path.exists():
            raise FileNotFoundError(f"Interview guide file not found at: {self.guide_path}")
            
        with open(self.guide_path, "r", encoding="utf-8") as f:
            self._guide = json.load(f)
        return self._guide

    def get_transcripts(self) -> List[Transcript]:
        if not self._transcripts:
            self.load_transcripts()
        return self._transcripts

    def get_transcript_by_id(self, transcript_id: str) -> Optional[Transcript]:
        for t in self.get_transcripts():
            if t.id.lower() == transcript_id.lower():
                return t
        return None

    def get_guide(self) -> Dict[str, Any]:
        if not self._guide:
            self.load_guide()
        return self._guide

transcript_loader = TranscriptLoader()
