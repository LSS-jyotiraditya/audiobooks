# 🎧 Voice-Enabled RAG Audiobook System

An intelligent audiobook system that converts e-books into interactive audio experiences. Upload a PDF or DOCX, and the system will read it aloud while allowing you to ask questions via voice - getting instant spoken answers without interrupting your listening flow.

## ✨ Features

- **📚 E-book Processing**: Upload PDF/DOCX files and automatically extract, chunk, and index content
- **🎵 Text-to-Speech**: Convert entire books into high-quality MP3 audio chunks
- **🎙️ Voice Questions**: Ask questions about the book using your voice while listening
- **🤖 Intelligent Answers**: Hybrid RAG system combines semantic and keyword search with LLM generation
- **⏯️ Playback Control**: Full pause/play/stop controls with seamless resume
- **🔄 Interrupt & Resume**: Questions pause the book, play the answer, then automatically resume
- **📊 Session Management**: Multiple concurrent listening sessions with individual state

## 🏗️ Architecture

### Core Components

- **`app.py`**: FastAPI server with streaming audio endpoints
- **`input.py`**: E-book processing and vector database management  
- **`rag.py`**: Hybrid retrieval system (semantic + keyword search) with LLM generation
- **`voice.py`**: Speech-to-text and text-to-speech utilities

### Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Vector Database**: ChromaDB with sentence-transformers embeddings
- **LLM**: Ollama (gemma3:12B) for answer generation
- **Speech Processing**: 
  - STT: OpenAI Whisper (local) or OpenAI API
  - TTS: OpenAI TTS API or offline engines (pyttsx3/espeak)
- **Document Processing**: PyPDF2, python-docx
- **Audio**: MP3 streaming with threading controls

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** with gemma3:12B model:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the model
   ollama pull gemma3:12B
   ```
3. **FFmpeg** (for audio conversion):
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd books
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API** (optional, for better TTS quality):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Start the server**:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:8000`

## 📖 Usage

### 1. Upload an E-book

```bash
curl -X POST "http://localhost:8000/upload_ebook" \
  -F "file=@your-book.pdf"
```

Response:
```json
{
  "success": true,
  "session_id": "uuid-here",
  "chunks": 45,
  "stream_url": "/stream/uuid-here",
  "media_base": "/media/uuid-here"
}
```

### 2. Stream the Audiobook

Open the stream URL in any audio player or browser:
```
http://localhost:8000/stream/{session_id}
```

### 3. Control Playback

```bash
# Pause
curl -X POST "http://localhost:8000/pause/{session_id}"

# Resume
curl -X POST "http://localhost:8000/play/{session_id}"

# Stop
curl -X POST "http://localhost:8000/stop/{session_id}"
```

### 4. Ask Voice Questions

```bash
curl -X POST "http://localhost:8000/ask/{session_id}" \
  -F "question_audio=@question.wav"
```

The system will:
1. Pause the current reading
2. Transcribe your question
3. Search the book content
4. Generate an answer using the LLM
5. Play the answer aloud
6. Resume reading where it left off

### 5. Check Status

```bash
curl "http://localhost:8000/status/{session_id}"
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: For high-quality TTS and STT (optional)

### Model Configuration

Edit the models in the respective classes:

- **Embedding Model**: `all-MiniLM-L6-v2` (in `input.py`)
- **LLM Model**: `gemma3:12B` (in `rag.py`) 
- **TTS Model**: `gpt-4o-mini-tts` (in `voice.py`)
- **Whisper Model**: `base` (in `voice.py`)

### Chunk Settings

Adjust text chunking in `input.py`:
```python
EbookProcessor(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## 📁 Project Structure

```
books/
├── app.py              # FastAPI server & streaming logic
├── input.py            # E-book processing & vector DB
├── rag.py              # Hybrid RAG retrieval & LLM
├── voice.py            # Speech processing (STT/TTS)
├── requirements.txt    # Python dependencies
├── uploads/            # Temporary uploaded files
├── tts_out/           # Generated audio files
├── vector_db/         # ChromaDB storage
└── tmp/               # Temporary audio files
```

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload_ebook` | Upload and process e-book |
| GET | `/stream/{session_id}` | Stream audiobook |
| POST | `/pause/{session_id}` | Pause playback |
| POST | `/play/{session_id}` | Resume playbook |
| POST | `/stop/{session_id}` | Stop playback |
| POST | `/ask/{session_id}` | Ask voice question |
| GET | `/status/{session_id}` | Get session status |
| GET | `/media/{session_id}/*` | Access audio files |

## 🔍 How It Works

### RAG Pipeline

1. **Document Processing**: Extract text from PDF/DOCX and split into semantic chunks
2. **Indexing**: Generate embeddings and store in ChromaDB vector database  
3. **Hybrid Retrieval**: 
   - Semantic search using sentence transformers
   - Keyword search using BM25 algorithm
   - Score fusion and reranking with cross-encoder
4. **Generation**: Context-aware answer generation using Ollama LLM

### Audio Streaming

- **Chunked TTS**: Books are converted to MP3 chunks for efficient streaming
- **Interrupt System**: Voice questions create priority audio that plays immediately
- **State Management**: Thread-safe playback controls with pause/resume capability

## 🛠️ Development

### Running in Development

```bash
# With auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

```bash
# Test upload
curl -X POST "http://localhost:8000/upload_ebook" \
  -F "file=@test.pdf"

# Test streaming (save to file)
curl "http://localhost:8000/stream/{session_id}" \
  --output test_audio.mp3
```

## 📋 Requirements

See `requirements.txt` for full dependency list. Key requirements:

- `fastapi` - Web framework
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `openai` - TTS/STT APIs
- `PyPDF2` - PDF processing
- `python-docx` - Word document processing
- `requests` - HTTP client for Ollama

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please add your preferred license.

## 🙏 Acknowledgments

- OpenAI for Whisper and TTS APIs
- Sentence Transformers for embedding models
- ChromaDB for vector storage
- Ollama for local LLM inference
```

This README provides a comprehensive overview of your voice-enabled RAG audiobook system, including setup instructions, usage examples, and technical details. You can save this as `README.md` in your project root.
