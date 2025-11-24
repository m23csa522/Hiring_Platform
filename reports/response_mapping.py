from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path

# Data processing
import pandas as pd
from pypdf import PdfReader
import json

# LangChain & LangGraph imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import re
from django.conf import settings

load_dotenv()

app = FastAPI(title="Multi-Document RAG Interview Evaluator")

# ============== MODELS ==============
class QuestionResponse(BaseModel):
    question: str
    user_answer: str

class QAPair(BaseModel):
    """Model for question-answer pair identified from transcript"""
    question: str
    answer: str
    timestamp: Optional[str] = None
    confidence: float = 0.9

class TranscriptInput(BaseModel):
    """Model for receiving interview transcript"""
    transcript_text: str
    candidate_id: Optional[str] = None
    interview_id: Optional[str] = None

class QAExtractionResult(BaseModel):
    """Model for QA extraction results"""
    qa_pairs: List[QAPair]
    total_pairs: int
    processing_time: float
    transcript_summary: Optional[str] = None

class EvaluationResult(BaseModel):
    question: str
    user_answer: str
    score: int
    feedback: str
    correct_answer: str
    matches_expected: bool
    sources_used: List[str]

class DocumentStatus(BaseModel):
    filename: str
    doc_type: str
    chunks: int
    status: str

class ProcessingStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentStatus]

# ============== STATE ==============
class InterviewState(TypedDict):
    question: str
    user_answer: str
    retrieved_context: str
    evaluation: str
    score: int
    sources: List[str]
    messages: Annotated[list, operator.add]

# ============== GLOBAL VARIABLES ==============
vector_db = None
llm = None
embeddings = None
document_metadata = {}  # Track doc sources
processing_stats = {
    "total_documents": 0,
    "total_chunks": 0,
    "documents": []
}

media_root = Path(__file__).resolve().parent.parent

print(f"Media root set to: {media_root}")

UPLOAD_DIR = media_root / "scripts"/"boiler_plate_questions"
VECTOR_DB_PATH = media_root / "vector_db"

Path(UPLOAD_DIR).mkdir(exist_ok=True)

# ============== DOCUMENT LOADERS ==============
def load_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return ""

def load_excel(file_path: str) -> str:
    """Extract text from Excel (all sheets)"""
    try:
        excel_file = pd.ExcelFile(file_path)
        text = ""
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text += f"\n=== Sheet: {sheet_name} ===\n"
            text += df.to_string()
            text += "\n"
        return text
    except Exception as e:
        print(f"Error loading Excel {file_path}: {e}")
        return ""

def load_csv(file_path: str) -> str:
    """Extract text from CSV"""
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return ""

def load_txt(file_path: str) -> str:
    """Load plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading TXT {file_path}: {e}")
        return ""

def load_json(file_path: str) -> str:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return ""

def load_markdown(file_path: str) -> str:
    """Load Markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading Markdown {file_path}: {e}")
        return ""

# ============== DOCUMENT PROCESSOR ==============
def process_document(file_path: str, filename: str) -> tuple[str, int]:
    """Process document based on file type"""
    file_ext = Path(file_path).suffix.lower()
    doc_type = "unknown"
    text = ""
    
    if file_ext == ".pdf":
        doc_type = "PDF"
        text = load_pdf(file_path)
    elif file_ext in [".xlsx", ".xls"]:
        doc_type = "Excel"
        text = load_excel(file_path)
    elif file_ext == ".csv":
        doc_type = "CSV"
        text = load_csv(file_path)
    elif file_ext == ".txt":
        doc_type = "Text"
        text = load_txt(file_path)
    elif file_ext == ".json":
        doc_type = "JSON"
        text = load_json(file_path)
    elif file_ext in [".md", ".markdown"]:
        doc_type = "Markdown"
        text = load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    if not text:
        raise ValueError(f"Failed to extract content from {filename}")
    
    # Create documents with metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(
        [Document(
            page_content=text,
            metadata={
                "source": filename,
                "type": doc_type,
                "file_path": file_path
            }
        )]
    )
    
    return split_docs, len(split_docs), doc_type

# ============== RAG INITIALIZATION ==============
def initialize_rag():
    """Initialize embeddings and LLM"""
    global vector_db, llm, embeddings
    
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="text-embedding-ada-002"
    )
    
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="gpt-4.1",
        temperature=0.7
    )

#     embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",      # or "text-embedding-3-large"
#     api_key=os.getenv("OPENAI_API_KEY")  # standard OpenAI key
# )

#     # Chat LLM (non-Azure)
#     llm = ChatOpenAI(
#         model="gpt-5.1",                     # or "gpt-4.1", etc.
#         temperature=0.7,
#         api_key=os.getenv("OPENAI_API_KEY")
#     )
    print("✓ Azure OpenAI Embeddings and LLM initialized")

def load_all_documents():
    """Load all documents from UPLOAD_DIR into vector DB"""
    global vector_db, document_metadata, processing_stats
    
    all_docs = []
    processing_stats = {"total_documents": 0, "total_chunks": 0, "documents": []}
    
    if not Path(UPLOAD_DIR).exists():
        print("No documents found in upload directory")
        return
    
    supported_extensions = {".pdf", ".xlsx", ".xls", ".csv", ".txt", ".json", ".md", ".markdown"}
    
    for file_path in Path(UPLOAD_DIR).glob("*"):
        if file_path.suffix.lower() not in supported_extensions:
            continue
        
        try:
            print(f"Processing: {file_path.name}")
            docs, chunk_count, doc_type = process_document(str(file_path), file_path.name)
            all_docs.extend(docs)
            
            processing_stats["documents"].append({
                "filename": file_path.name,
                "doc_type": doc_type,
                "chunks": chunk_count,
                "status": "loaded"
            })
            processing_stats["total_documents"] += 1
            processing_stats["total_chunks"] += chunk_count
            
            print(f"✓ Loaded {file_path.name} ({chunk_count} chunks)")
        
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            processing_stats["documents"].append({
                "filename": file_path.name,
                "doc_type": "unknown",
                "chunks": 0,
                "status": f"error: {str(e)}"
            })
    
    if all_docs:
        vector_db = FAISS.from_documents(all_docs, embeddings)
        print(f"\n✓ Vector DB created with {len(all_docs)} total chunks")
    else:
        print("⚠ No documents loaded")

# ============== RETRIEVAL ==============
def retrieve_relevant_context(question: str, top_k: int = 5) -> tuple[str, List[str]]:
    """Retrieve context and track sources"""
    if not vector_db:
        return "", []
    
    results = vector_db.similarity_search_with_score(question, k=top_k)
    context_parts = []
    sources = set()
    
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        sources.add(source)
        doc_type = doc.metadata.get("type", "")
        context_parts.append(
            f"[Source: {source} ({doc_type})]\n{doc.page_content}"
        )
    
    context = "\n---\n".join(context_parts)
    return context, list(sources)

# ============== LANGGRAPH AGENT ==============
def create_evaluation_agent():
    """Create LangGraph agent for evaluation"""
    workflow = StateGraph(InterviewState)
    
    def retrieve_node(state: InterviewState) -> InterviewState:
        """Retrieve relevant context from vector DB"""
        context, sources = retrieve_relevant_context(state["question"])
        state["retrieved_context"] = context
        state["sources"] = sources
        state["messages"].append(
            HumanMessage(content=f"Retrieved context from {len(sources)} sources")
        )
        return state
    
    def evaluate_node(state: InterviewState) -> InterviewState:
        """Evaluate user answer against retrieved context"""
        prompt = f"""You are an expert interviewer. Evaluate the user's answer based on the provided reference materials.

Question: {state['question']}
User Answer: {state['user_answer']}

Reference Context (from multiple sources):
{state['retrieved_context']}

Provide evaluation in this exact format:
SCORE: [0-100]
MATCH: [YES/NO]
CORRECT_ANSWER: [Brief correct answer based on sources]
FEEDBACK: [Detailed constructive feedback]

Scoring guidelines:
- 90-100: Excellent, comprehensive answer
- 75-89: Good, covers main points
- 60-74: Acceptable, some gaps
- 40-59: Partial, missing key elements
- 0-39: Incomplete or incorrect

Be fair and base evaluation on reference materials provided."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        eval_text = response.content
        
        # Parse response
        score = 70
        match = "NO"
        correct_answer = ""
        feedback = ""
        
        for line in eval_text.split("\n"):
            if "SCORE:" in line:
                try:
                    score = int(line.split("SCORE:")[1].strip().split()[0])
                except:
                    pass
            elif "MATCH:" in line:
                match = "YES" if "YES" in line else "NO"
            elif "CORRECT_ANSWER:" in line:
                correct_answer = line.split("CORRECT_ANSWER:")[1].strip()
            elif "FEEDBACK:" in line:
                feedback = line.split("FEEDBACK:")[1].strip()
        
        state["evaluation"] = eval_text
        state["score"] = score
        state["messages"].append(
            AIMessage(content=f"Evaluation complete. Score: {score}")
        )
        return state
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_edge("retrieve", "evaluate")
    workflow.set_entry_point("retrieve")
    workflow.set_finish_point("evaluate")
    
    return workflow.compile()

evaluator_agent = None

# ============== STARTUP ==============
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global evaluator_agent
    initialize_rag()
    load_all_documents()
    evaluator_agent = create_evaluation_agent()
    print("✓ Server started")

# ============== ENDPOINTS ==============
@app.get("/")
async def root():
    return {
        "message": "Multi-Document RAG Interview Evaluator",
        "endpoints": {
            "evaluate": "POST /evaluate",
            "upload": "POST /upload",
            "documents": "GET /documents",
            "reload": "POST /reload-documents",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_db_ready": vector_db is not None,
        "stats": processing_stats
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a document"""
    try:
        file_path = Path(UPLOAD_DIR) / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reload documents in background
        if background_tasks:
            background_tasks.add_task(load_all_documents)
        
        return {
            "message": f"File {file.filename} uploaded successfully",
            "status": "processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/documents", response_model=ProcessingStats)
async def get_documents():
    """Get all loaded documents"""
    return processing_stats

@app.post("/reload-documents")
async def reload_documents():
    """Manually reload all documents"""
    try:
        load_all_documents()
        return {
            "message": "Documents reloaded",
            "stats": processing_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_answer(response: QuestionResponse):
    """Evaluate user's answer"""
    try:
        if not response.question or not response.user_answer:
            raise HTTPException(
                status_code=400,
                detail="Question and user_answer are required"
            )
        
        if not vector_db:
            raise HTTPException(
                status_code=503,
                detail="Vector database not initialized. Upload documents first."
            )
        
        initial_state = {
            "question": response.question,
            "user_answer": response.user_answer,
            "retrieved_context": "",
            "evaluation": "",
            "score": 0,
            "sources": [],
            "messages": []
        }
        
        final_state = evaluator_agent.invoke(initial_state)
        
        eval_text = final_state["evaluation"]
        score = final_state["score"]
        correct_answer = ""
        feedback = ""
        matches = False
        
        for line in eval_text.split("\n"):
            if "MATCH:" in line:
                matches = "YES" in line
            elif "CORRECT_ANSWER:" in line:
                correct_answer = line.split("CORRECT_ANSWER:")[1].strip()
            elif "FEEDBACK:" in line:
                feedback = line.split("FEEDBACK:")[1].strip()
        
        return EvaluationResult(
            question=response.question,
            user_answer=response.user_answer,
            score=score,
            feedback=feedback,
            correct_answer=correct_answer,
            matches_expected=matches,
            sources_used=final_state["sources"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== TRANSCRIPT ANALYSIS ==============
def extract_qa_pairs_from_transcript(transcript_text: str) -> List[QAPair]:
    """
    Extract question-answer pairs from interview transcript using LLM.
    Assumes interviewer asks questions and candidate provides answers.
    """
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    prompt = f"""You are an expert at analyzing interview transcripts. Your task is to identify all question-answer pairs from the following interview transcript.

For each Q&A pair:
1. Identify the interviewer's question
2. Extract the candidate's complete answer
3. Ensure each pair is properly separated

Format your response as a JSON array of objects with exactly this structure:
[
  {{"question": "What is your experience with Python?", "answer": "I have 5 years of Python experience..."}},
  {{"question": "Tell us about your last project", "answer": "My last project was a machine learning..."}}
]

IMPORTANT: Return ONLY the JSON array, nothing else. No markdown, no explanations.

Transcript:
---
{transcript_text}
---

Now extract all Q&A pairs:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Clean up response if wrapped in markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        qa_data = json.loads(response_text)
        
        qa_pairs = []
        for item in qa_data:
            qa_pairs.append(QAPair(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                timestamp=item.get("timestamp"),
                confidence=item.get("confidence", 0.9)
            ))
        
        return qa_pairs
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response: {response_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response: {str(e)}"
        )
    except Exception as e:
        print(f"Error extracting QA pairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-transcript", response_model=QAExtractionResult)
async def extract_transcript(transcript_input: TranscriptInput):
    """
    API endpoint to extract question-answer pairs from interview transcript.
    
    Request body:
    {
        "transcript_text": "Interviewer: What is your experience? Candidate: I have...",
        "candidate_id": "optional-candidate-id",
        "interview_id": "optional-interview-id"
    }
    
    Response:
    {
        "qa_pairs": [
            {"question": "...", "answer": "...", "confidence": 0.9},
            ...
        ],
        "total_pairs": 5,
        "processing_time": 2.34
    }
    """
    import time
    
    try:
        if not transcript_input.transcript_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Transcript text cannot be empty"
            )
        
        start_time = time.time()
        
        # Extract QA pairs using LLM
        qa_pairs = extract_qa_pairs_from_transcript(transcript_input.transcript_text)
        
        processing_time = time.time() - start_time
        
        result = QAExtractionResult(
            qa_pairs=qa_pairs,
            total_pairs=len(qa_pairs),
            processing_time=round(processing_time, 2),
            transcript_summary=f"Extracted {len(qa_pairs)} Q&A pairs from interview transcript"
        )
        
        print(f"Extracted {len(qa_pairs)} Q&A pairs in {processing_time:.2f}s")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in extract_transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-transcript-batch")
async def extract_transcript_batch(transcripts: List[TranscriptInput]):
    """
    Extract Q&A pairs from multiple transcripts in batch.
    
    Request body:
    [
        {"transcript_text": "Interviewer: ... Candidate: ..."},
        {"transcript_text": "Interviewer: ... Candidate: ..."}
    ]
    
    Response:
    [
        {"qa_pairs": [...], "total_pairs": 5, ...},
        {"qa_pairs": [...], "total_pairs": 3, ...}
    ]
    """
    try:
        if not transcripts:
            raise HTTPException(
                status_code=400,
                detail="At least one transcript is required"
            )
        
        results = []
        for transcript_input in transcripts:
            try:
                result = await extract_transcript(transcript_input)
                results.append(result.dict())
            except HTTPException as e:
                results.append({
                    "error": e.detail,
                    "qa_pairs": [],
                    "total_pairs": 0
                })
        
        return {
            "total_transcripts": len(transcripts),
            "successful": sum(1 for r in results if "error" not in r),
            "results": results
        }
    
    except Exception as e:
        print(f"Error in batch extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)
