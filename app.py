import os
import uuid
import time
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from input import EbookProcessor
from rag import HybridRAG
from voice import VoiceIO


class SessionState:
    def __init__(
        self,
        session_id: str,
        processor: EbookProcessor,
        rag: HybridRAG,
        voice: VoiceIO,
        book_audio_paths: List[str],
    ):
        self.session_id = session_id
        self.processor = processor
        self.rag = rag
        self.voice = voice

        self.book_audio_paths = book_audio_paths
        self.current_index = 0

        # Playback control
        self.playing_event = threading.Event()
        self.playing_event.set()  # start in playing state
        self.stop_event = threading.Event()

        # Queue for interrupt audios (LLM answers)
        self.interrupts: "queue.Queue[str]" = queue.Queue()

        # Lock for index updates
        self.lock = threading.Lock()


SESSIONS: Dict[str, SessionState] = {}

MEDIA_ROOT = Path("./tts_out").resolve()
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Books Voice RAG API")

# allow the frontend to call the API (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:9000"] for tighter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve frontend files from the package dir at /static
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent)), name="static")


# redirect root to the frontend page
@app.get("/")
def root():
    return RedirectResponse(url="/static/frontend.html")


def _read_file_streamed(path: Path, state: SessionState, chunk_size: int = 64 * 1024):
    with path.open("rb") as f:
        while True:
            # Pause handling
            while not state.playing_event.is_set() and not state.stop_event.is_set():
                time.sleep(0.1)
            if state.stop_event.is_set():
                return

            data = f.read(chunk_size)
            if not data:
                break
            yield data


def _session_streamer(state: SessionState):
    """
    Streams: [interrupt answers]* and then book chunks in order.
    Checks for pause and stop. Interrupts are played before the next book chunk.
    """
    # Continue until book chunks exhausted or stop requested
    while not state.stop_event.is_set():
        # First, drain any interrupts (answers)
        try:
            while True:
                ans_path = Path(state.interrupts.get_nowait())
                if ans_path.exists():
                    for chunk in _read_file_streamed(ans_path, state):
                        yield chunk
                # small gap between answer and resumed content
                for _ in range(5):
                    if state.stop_event.is_set():
                        return
                    time.sleep(0.02)
        except queue.Empty:
            pass

        # If book finished, idle-wait for possible future interrupts
        with state.lock:
            if state.current_index >= len(state.book_audio_paths):
                # Idle and wait for interrupts or stop
                for _ in range(50):
                    if state.stop_event.is_set():
                        return
                    # If an interrupt arrives, break to top loop to play it
                    if not state.interrupts.empty():
                        break
                    time.sleep(0.1)
                continue

            path = Path(state.book_audio_paths[state.current_index])

        if not path.exists():
            # Skip missing files
            with state.lock:
                state.current_index += 1
            continue

        # Stream current book chunk (respects pause)
        for chunk in _read_file_streamed(path, state):
            yield chunk

        # Next chunk
        with state.lock:
            state.current_index += 1

        # Tiny spacer between chunks
        for _ in range(10):
            if state.stop_event.is_set():
                return
            time.sleep(0.02)


def _ensure_session(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    return state


@app.post("/upload_ebook")
async def upload_ebook(file: UploadFile = File(...)):
    """
    - Upload an e-book (PDF/DOCX)
    - Process + index in ChromaDB
    - Generate TTS for the whole book into mp3 chunks
    - Returns a session_id and stream URL
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF/DOCX.")

    # Save uploaded file to temp path
    uploads_dir = Path("./uploads").resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = uploads_dir / f"{uuid.uuid4()}{suffix}"
    with temp_path.open("wb") as f:
        f.write(await file.read())

    # Initialize core components
    processor = EbookProcessor(db_path="./vector_db")
    rag = HybridRAG(processor=processor)
    voice = VoiceIO()

    # Process ebook -> stores chunks in vector DB with metadata 'filename'
    result = processor.process_ebook(str(temp_path))
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to process e-book"))

    filename = result["filename"]

    # Retrieve the stored chunks back from vector DB for TTS (join into one text)
    got = processor.collection.get(
        where={"filename": filename},
        include=["documents"],
        limit=100000,
    )
    docs = got.get("documents", []) or []
    if not docs:
        raise HTTPException(status_code=500, detail="No stored chunks found for TTS")

    full_text = "\n\n".join(docs)

    # Generate TTS mp3 chunks for the book
    session_id = str(uuid.uuid4())
    out_dir = MEDIA_ROOT / session_id / "book"
    out_dir.mkdir(parents=True, exist_ok=True)
    book_paths = voice.tts_book(text=full_text, output_dir=str(out_dir), base_name="book_chunk")

    # Create session state
    state = SessionState(
        session_id=session_id,
        processor=processor,
        rag=rag,
        voice=voice,
        book_audio_paths=book_paths,
    )
    SESSIONS[session_id] = state

    return JSONResponse(
        {
            "success": True,
            "session_id": session_id,
            "chunks": len(book_paths),
            "stream_url": f"/stream/{session_id}",
            "media_base": f"/media/{session_id}",
        }
    )


@app.get("/stream/{session_id}")
def stream_audio(session_id: str):
    """
    Streaming endpoint that plays:
    - any queued LLM answer audio (interrupts)
    - then book chunks in sequence
    Respects pause/play and can be resumed by reusing the same session_id.
    """
    state = _ensure_session(session_id)

    def generator():
        try:
            for data in _session_streamer(state):
                yield data
        finally:
            # Do not delete session here; allow resume and multiple clients
            pass

    return StreamingResponse(generator(), media_type="audio/mpeg")


@app.post("/pause/{session_id}")
def pause(session_id: str):
    state = _ensure_session(session_id)
    state.playing_event.clear()
    return {"success": True, "status": "paused"}


@app.post("/play/{session_id}")
def play(session_id: str):
    state = _ensure_session(session_id)
    state.playing_event.set()
    return {"success": True, "status": "playing"}


@app.post("/stop/{session_id}")
def stop(session_id: str):
    state = _ensure_session(session_id)
    state.stop_event.set()
    return {"success": True, "status": "stopped"}


@app.post("/ask/{session_id}")
async def ask_llm(session_id: str, question_audio: UploadFile = File(...)):
    """
    Pauses reading, transcribes the user's voice question (STT),
    retrieves answer via Hybrid RAG, synthesizes TTS for the answer,
    queues it to be played next, then resumes book reading.
    """
    state = _ensure_session(session_id)

    # Pause reading immediately
    state.playing_event.clear()

    # Save question audio to tmp
    tmp_dir = Path("./tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    q_path = tmp_dir / f"{session_id}_q_{uuid.uuid4()}_{Path(question_audio.filename or '').name}"
    with q_path.open("wb") as f:
        f.write(await question_audio.read())

    # STT
    try:
        transcript = state.voice.transcribe(str(q_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript from STT")

    # RAG answer
    try:
        ans = state.rag.answer(query=transcript)
        answer_text = ans.get("answer", "") or "I don't know."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

    # TTS for answer
    answers_dir = MEDIA_ROOT / session_id / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    ans_path = answers_dir / f"answer_{int(time.time())}.mp3"
    try:
        state.voice.tts_to_file(answer_text, str(ans_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    # Queue the answer to be played ASAP (before next book chunk)
    state.interrupts.put(str(ans_path))

    # Resume reading
    state.playing_event.set()

    return {
        "success": True,
        "transcript": transcript,
        "answer_text": answer_text,
        "answer_audio_url": f"/media/{session_id}/answers/{ans_path.name}",
    }


@app.get("/status/{session_id}")
def status(session_id: str):
    state = _ensure_session(session_id)
    with state.lock:
        return {
            "session_id": session_id,
            "playing": state.playing_event.is_set() and not state.stop_event.is_set(),
            "stopped": state.stop_event.is_set(),
            "current_index": state.current_index,
            "total_chunks": len(state.book_audio_paths),
            "next_book_chunk": state.book_audio_paths[state.current_index]
            if state.current_index < len(state.book_audio_paths)
            else None,
            "pending_interrupts": state.interrupts.qsize(),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
