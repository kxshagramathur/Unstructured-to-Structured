import os
import sys
import json
import shutil
import tempfile
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions

import boto3
from botocore.exceptions import ClientError


# ---------- FastAPI Setup ----------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- AWS Bedrock Setup ----------
client = boto3.client("bedrock-runtime", region_name="ap-south-1")
model_id = "apac.amazon.nova-lite-v1:0"


# ---------- Helper: Build Improved Prompt ----------
def build_prompt(doc_content: str, key_fields: List[str], table_columns: List[str]) -> str:
    return f"""
You are an information extraction assistant that extracts structured data from documents.

Document Content (Markdown format):
--------------------
{doc_content}
--------------------

EXTRACTION REQUIREMENTS:

1. KEY-VALUE EXTRACTION:
   Extract the following fields as a clean JSON object:
   Fields: {", ".join(key_fields)}
   
   Rules:
   - Use exact field names as JSON keys
   - If a field is not found, set value to null
   - Return clean, properly formatted JSON
   
   Example format:
   {{
     "Invoice Number": "INV-12345",
     "Invoice Date": "2023-08-30",
     "Total Amount": "₹15,000"
   }}

2. TABLE EXTRACTION:
   Extract ONLY the following specific columns from tables:
   Required Columns: {", ".join(table_columns)}
   
   CRITICAL RULES:
   - Find tables that contain columns matching these keywords
   - Extract ONLY the specified columns, ignore all other columns
   - Match column names intelligently but flexibly:
     * "Batch No." matches "Batch No", "Batch Number", "Batch", etc.
     * "Exp.Dt" matches "Exp Date", "Expiry Date", "Expiration", etc.
     * "Quantity" matches "Qty", "Quan", "Amount", etc.
   - Output as clean CSV with ONLY the requested columns
   - Use the original column names from the document as headers
   - If a requested column is not found, skip it entirely
   
   Example format (if requesting "Batch No." and "Exp.Dt"):
   Batch No.,Exp.Dt
   B001,2024-12-31
   B002,2025-01-15

IMPORTANT FORMATTING RULES:
- Return ONLY the JSON object and CSV data
- CSV must contain ONLY the requested columns
- Separate JSON and CSV with a blank line
- No additional text, explanations, or markdown formatting
- Ensure JSON is valid and CSV is properly comma-separated
"""


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


# In-memory storage for processed documents
processed_documents = {}

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Step 1: Process document with OCR (Docling) - happens on file upload"""
    tmp_dir = None
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            return JSONResponse(
                content={"error": "Unsupported file type. Please upload PDF, JPG, or PNG files."},
                status_code=400
            )

        # Save uploaded file temporarily
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ---------- OCR Processing with Docling ----------
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        doc_content = result.document.export_to_markdown()

        # Generate unique document ID
        import hashlib
        import time
        doc_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()
        
        # Store processed document in memory
        processed_documents[doc_id] = {
            "filename": file.filename,
            "content": doc_content,
            "processed_at": time.time()
        }

        # Clean up old documents (older than 1 hour)
        cleanup_old_documents()

        return JSONResponse(content={
            "document_id": doc_id,
            "filename": file.filename,
            "preview": doc_content[:300] + "..." if len(doc_content) > 300 else doc_content,
            "status": "processed"
        })

    except Exception as e:
        return JSONResponse(
            content={"error": f"Document processing failed: {str(e)}"},
            status_code=500
        )
    
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/extract")
async def extract_data(
    document_id: str = Form(...),
    key_values: str = Form(...),           # comma-separated field names
    table_columns: str = Form(...)         # newline-separated column keywords
):
    """Step 2: Extract data using LLM - happens when user clicks extract"""
    
    try:
        # Get processed document
        if document_id not in processed_documents:
            return JSONResponse(
                content={"error": "Document not found. Please upload the document again."},
                status_code=404
            )

        doc_data = processed_documents[document_id]
        doc_content = doc_data["content"]

        # ---------- Parse input parameters ----------
        key_fields = [k.strip() for k in key_values.split(",") if k.strip()]
        table_column_keywords = [k.strip() for k in table_columns.split("\n") if k.strip()]
        
        if not key_fields and not table_column_keywords:
            return JSONResponse(
                content={"error": "Please specify at least one key field or table column keyword."},
                status_code=400
            )

        # ---------- Build and send prompt to LLM ----------
        prompt = build_prompt(doc_content, key_fields, table_column_keywords)

        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={
                "maxTokens": 2048,
                "temperature": 0.1,
                "topP": 0.9
            },
        )

        response_text = response["output"]["message"]["content"][0]["text"]

        return JSONResponse(content={
            "result": response_text,
            "document_info": {
                "filename": doc_data["filename"],
                "processed_at": doc_data["processed_at"]
            }
        })

    except ClientError as e:
        return JSONResponse(
            content={"error": f"AWS Bedrock error: {str(e)}"},
            status_code=500
        )
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Extraction failed: {str(e)}"},
            status_code=500
        )


def cleanup_old_documents():
    """Clean up documents older than 1 hour to prevent memory leaks"""
    import time
    current_time = time.time()
    expired_docs = [
        doc_id for doc_id, doc_data in processed_documents.items()
        if current_time - doc_data["processed_at"] > 3600  # 1 hour
    ]
    
    for doc_id in expired_docs:
        del processed_documents[doc_id]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Document extraction service is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)