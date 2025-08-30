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


# ---------- Helper: Build Prompt ----------
def build_prompt(doc_content: str, key_fields: List[str], tables_spec: Dict[str, List[str]]) -> str:
    return f"""
You are an information extraction assistant.  
Your job is to read the following document (in Markdown format) and extract structured information according to the user-defined input schema.

Document:
--------------------
{doc_content}
--------------------

Extraction Instructions:
1. Key–Value Sections:
   - Extract the following fields as JSON.
   - Use the exact field names as keys.
   - If a field is missing, return its value as null.

   Fields: {", ".join(key_fields)}

   Example output:
   {{
     "Invoice Number": "BL1498",
     "Invoice Date": "2025-08-30"
   }}

2. Table Sections:
   - Extract tables according to the specified column names.
   - Each column must appear exactly as given.
   - Output tables in **CSV format** (comma-separated, no markdown).

   Tables to extract:
   {json.dumps(tables_spec)}

   Example output for a table:
Item Name,Qty,Rate,Amount
Widget A,10,50,500
Widget B,5,40,200

sql
Copy code

Formatting Rules:
- Return only JSON and CSV sections as specified.
- Do not add commentary, explanations, or extra formatting.
"""


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
 return templates.TemplateResponse("index.html", {"request": request})


@app.post("/extract")
async def extract_data(
 request: Request,
 file: UploadFile = File(...),
 key_values: str = Form(...),     # comma-separated field names
 tables: str = Form(...)          # JSON string for tables { "TableName": ["Col1","Col2"] }
):
 try:
     # Save uploaded file temporarily
     tmp_dir = tempfile.mkdtemp()
     tmp_path = os.path.join(tmp_dir, file.filename)
     with open(tmp_path, "wb") as f:
         shutil.copyfileobj(file.file, f)

     # ---------- Step 1: OCR with Docling ----------
     converter = DocumentConverter()
     result = converter.convert(tmp_path)
     doc_content = result.document.export_to_markdown()

     # ---------- Step 2: Build Prompt ----------
     key_fields = [k.strip() for k in key_values.split(",") if k.strip()]
     tables_spec = json.loads(tables) if tables else {}
     prompt = build_prompt(doc_content, key_fields, tables_spec)

     # ---------- Step 3: Call Bedrock LLM ----------
     conversation = [
         {
             "role": "user",
             "content": [{"text": prompt}],
         }
     ]

     response = client.converse(
         modelId=model_id,
         messages=conversation,
         inferenceConfig={"maxTokens": 1024, "temperature": 0.3, "topP": 0.9},
     )

     response_text = response["output"]["message"]["content"][0]["text"]

     return JSONResponse(content={"result": response_text})

 except (ClientError, Exception) as e:
     return JSONResponse(content={"error": str(e)}, status_code=500)
 finally:
     shutil.rmtree(tmp_dir, ignore_errors=True)