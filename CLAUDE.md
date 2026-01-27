# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course Materials RAG System - a full-stack web app that lets users query educational course content via an AI chatbot. Uses semantic search (ChromaDB + Sentence Transformers) combined with Claude AI for context-aware responses.

## Commands

**Always use `uv` to manage dependencies and run commands - never use `pip` directly.**

**Start the server:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

App runs at http://localhost:8000

## Architecture

### Request Flow
```
Frontend (script.js) → POST /api/query
    → FastAPI (app.py)
    → RAGSystem.query() orchestrates:
        → SessionManager: get conversation history
        → AIGenerator: call Claude API with tools
        → Claude decides to use search_course_content tool
        → CourseSearchTool → VectorStore → ChromaDB semantic search
        → Results returned to Claude for synthesis
        → Final answer + sources returned
```

### Core Components (backend/)

| File | Purpose |
|------|---------|
| `rag_system.py` | Central orchestrator - coordinates all subsystems |
| `ai_generator.py` | Claude API integration with tool calling loop |
| `vector_store.py` | ChromaDB wrapper - two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course docs, extracts metadata, chunks text with overlap |
| `search_tools.py` | `CourseSearchTool` for semantic search, `ToolManager` for tool registry |
| `session_manager.py` | Conversation history per session (limited to MAX_HISTORY exchanges) |
| `config.py` | All configuration (chunk size, model names, API keys) |

### Document Format

Course documents in `docs/` must follow:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
...
```

### Key Patterns

**Tool Calling Flow:** Claude receives tools, decides to search, executes `search_course_content`, gets results, then synthesizes final answer in a second API call.

**Semantic Search:** Text is embedded via `all-MiniLM-L6-v2`, stored in ChromaDB, queried by cosine similarity. Course name resolution uses fuzzy semantic matching.

**Chunking:** Sentence-based with configurable overlap (default 800 chars, 100 overlap) to preserve context across boundaries.

## Configuration (config.py)

Key settings: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `MAX_RESULTS=5`, `MAX_HISTORY=2`, `ANTHROPIC_MODEL=claude-sonnet-4-20250514`

## API Endpoints

- `POST /api/query` - Submit query with optional session_id
- `GET /api/courses` - Get course statistics
