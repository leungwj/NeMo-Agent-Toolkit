import re
from typing import Callable, Dict, List, Optional
from llama_index.core.schema import TextNode

# Supports "MM:SS Speaker" or "HH:MM:SS Speaker"
TIME_SPEAKER_RE = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s+([A-Za-z0-9_.-]+)\s*$")

class TurnAwareTranscriptSplitter:
    def __init__(
        self,
        target_words: int = 500,
        overlap_turns: int = 1,
        token_counter: Optional[Callable[[str], int]] = None,
        target_tokens: Optional[int] = None,
        metadata_base: Optional[dict] = None
    ) -> None:
        self.target_words = target_words
        self.overlap_turns = max(0, overlap_turns)
        self.token_counter = token_counter
        self.target_tokens = target_tokens
        self.metadata_base = metadata_base or {}

    def _to_seconds(self, h: Optional[str], m: str, s: str) -> int:
        hh = int(h) if h is not None else 0
        return hh * 3600 + int(m) * 60 + int(s)

    def parse_transcript(self, text: str) -> List[Dict]:
        """
        Parse blocks like:
        00:40 S2
        Text...
        01:13 S1
        Next text...
        Returns list of utterances with start_sec, end_sec, speaker, text.
        """
        lines = text.splitlines()
        utts: List[Dict] = []
        cur: Optional[Dict] = None

        def flush():
            nonlocal cur
            if cur and cur["buf"]:
                cur["text"] = "\n".join(cur["buf"]).strip()
                if cur["text"]:
                    utts.append({k: cur[k] for k in ("start_sec", "speaker", "text")})
            cur = None

        for line in lines:
            m = TIME_SPEAKER_RE.match(line)
            if m:
                flush()
                h, mm, ss, spk = m.groups()
                cur = {"start_sec": self._to_seconds(h, mm, ss), "speaker": spk, "buf": []}
            else:
                if cur is None:
                    cur = {"start_sec": 0, "speaker": "S0", "buf": []}
                cur["buf"].append(line)
        flush()

        for i, u in enumerate(utts):
            u["end_sec"] = utts[i + 1]["start_sec"] if i + 1 < len(utts) else None
        return utts

    def _count_words(self, s: str) -> int:
        return len(s.split())

    def _budget(self) -> int:
        return self.target_tokens if (self.token_counter and self.target_tokens) else self.target_words

    def _length(self, s: str) -> int:
        if self.token_counter and self.target_tokens:
            return self.token_counter(s)
        return self._count_words(s)
    
    def split_text(self, text: str, source: Optional[str] = None) -> List[TextNode]:
        utts = self.parse_transcript(text)
        if not utts:
            return [
                TextNode(
                    text=text,
                    metadata={**self.metadata_base, "source": source, "type": "transcript", "chunk_index": 0},
                )
            ]

        chunks: List[TextNode] = []
        buf = []
        idx = 0
        budget = self._budget()
        cur_len = 0

        def commit():
            nonlocal buf, idx, cur_len
            if not buf:
                return
            body = "\n\n".join([f"{u['speaker']}: {u['text']}" for u in buf]).strip()

            # Ensure scalar types for Milvus metadata
            speakers_set = sorted({u["speaker"] for u in buf})
            start_val = float(buf[0]["start_sec"] or 0.0)
            last_end = buf[-1].get("end_sec")
            end_val = float(last_end if last_end is not None else buf[-1]["start_sec"])

            meta = {
                **self.metadata_base,
                "type": "transcript",
                "chunk_index": idx,
                "start_sec": start_val,                 # float
                "end_sec": end_val,                     # float
                "speakers": ", ".join(speakers_set),    # string
                "speakers_count": len(speakers_set),    # int
            }
            if source:
                meta["source"] = source

            # Use TextNode instead of Document
            chunks.append(TextNode(text=body, metadata=meta))
            idx += 1

        i = 0
        while i < len(utts):
            u = utts[i]
            utext = u["text"]
            ulen = self._length(utext)

            # If adding this turn would exceed the budget and buffer has content, commit first.
            if cur_len > 0 and cur_len + ulen > budget:
                commit()
                buf = buf[-self.overlap_turns:] if self.overlap_turns > 0 else []
                cur_len = sum(self._length(x["text"]) for x in buf)

            buf.append(u)
            cur_len += ulen
            i += 1

        if buf:
            commit()

        return chunks