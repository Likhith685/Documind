import json
import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from conversation import DEFAULT_MODEL, DocuMindEngine, ollama_status
from library import DocumentLibrary, safe_name

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="DocuMind", description="Offline PDF question answering", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

library = DocumentLibrary()
engine = DocuMindEngine(index=library.index)


class Question(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=10)
    model: str = Field(DEFAULT_MODEL)
    temperature: float = Field(0.1, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=20)


def sync_engine(model: Optional[str] = None, temperature: Optional[float] = None) -> None:
    engine.set_index(library.index)
    engine.configure(model, temperature)


@app.get("/", response_class=HTMLResponse)
def home():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h1>DocuMind</h1><p>Web assets are missing.</p>", status_code=404)
    with open(index_file, encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


@app.get("/api/health")
def health():
    return {"ollama": ollama_status(), "library": library.stats()}


@app.get("/api/documents")
def documents():
    return {"documents": library.documents(), "stats": library.stats()}


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    pdfs = [f for f in files if (f.filename or "").lower().endswith(".pdf")]
    if not pdfs:
        raise HTTPException(400, "Upload at least one PDF file.")

    saved = [library.store_upload(f.filename, f.file) for f in pdfs]

    try:
        result = library.add(saved)
    except Exception as error:
        for path in saved:
            os.remove(path)
        raise HTTPException(400, str(error))

    sync_engine()
    return {
        "added": [os.path.basename(p) for p in saved],
        "pages": result["pages"],
        "documents": library.documents(),
        "stats": library.stats(),
    }


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    if not library.remove(filename):
        raise HTTPException(404, f"{filename} is not in the library.")
    sync_engine()
    return {"removed": safe_name(filename), "documents": library.documents(), "stats": library.stats()}


@app.get("/api/documents/{filename}/file")
def download_document(filename: str):
    path = os.path.join(library.upload_dir, safe_name(filename))
    if not os.path.exists(path):
        raise HTTPException(404, "Document not found.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))


@app.get("/api/documents/{filename}/summary")
def summarize_document(filename: str):
    pages = library.pages_for(filename)
    if not pages:
        raise HTTPException(404, "Document not found or has no readable text.")
    sync_engine()
    return {"filename": safe_name(filename), "summary": engine.summarize(pages)}


@app.get("/api/suggestions")
def suggestions():
    documents = library.documents()
    if not documents:
        return {"questions": []}
    sync_engine()
    pages = library.pages_for(documents[0]["filename"])
    return {"questions": engine.suggest(pages)}


@app.post("/api/search")
def search(request: SearchRequest):
    sync_engine()
    return {"results": engine.search(request.query, k=request.k)}


@app.post("/api/chat")
def chat(request: Question):
    sync_engine(request.model, request.temperature)
    return engine.ask(request.question, k=request.k)


@app.post("/api/chat/stream")
def chat_stream(request: Question):
    sync_engine(request.model, request.temperature)

    def events():
        for event in engine.ask_stream(request.question, k=request.k):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
def history():
    return {"messages": engine.memory.messages}


@app.get("/api/history/export", response_class=PlainTextResponse)
def export_history():
    return PlainTextResponse(
        engine.memory.as_markdown(),
        headers={"Content-Disposition": 'attachment; filename="documind-chat.md"'},
    )


@app.delete("/api/history")
def clear_history():
    engine.clear_history()
    return {"cleared": True}


@app.post("/api/reset")
def reset():
    library.reset()
    engine.clear_history()
    sync_engine()
    return {"reset": True, "stats": library.stats()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
