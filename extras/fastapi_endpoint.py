#!/usr/bin/env python3
import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our data processing modules
import sys
sys.path.append('..')
from data_cleaner import EMRDataCleaner, NormalizedClaim
from main import ResubmissionAnalyzer, ResubmissionCandidate


# Pydantic models
class ProcessingRequest(BaseModel):
    source_system: str = Field(..., description="Source system identifier")
    reference_date: str = Field(default="2025-07-30", description="Reference date")
    auto_classify: bool = Field(default=True, description="Use LLM classification")


class ProcessingResponse(BaseModel):
    request_id: str
    status: str
    message: str
    total_claims: int
    eligible_claims: int
    processing_time: float
    timestamp: str


class EMRDataAPI:
    def __init__(self):
        self.app = FastAPI(
            title="EMR Data Processing API",
            description="API for processing EMR data and identifying resubmission candidates",
            version="1.0.0"
        )
        
        self.processing_requests: Dict[str, Dict[str, Any]] = {}
        self.request_counter = 0
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.post("/upload", response_model=ProcessingResponse)
        async def upload_dataset(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            request: ProcessingRequest = None
        ):
            try:
                if not file.filename.endswith(('.csv', '.json')):
                    raise HTTPException(
                        status_code=400, 
                        detail="Only CSV and JSON files are supported"
                    )
                
                request_id = f"req_{self.request_counter:06d}"
                self.request_counter += 1
                
                self.processing_requests[request_id] = {
                    "status": "processing",
                    "progress": 0.0,
                    "message": "File uploaded, starting processing...",
                    "timestamp": datetime.now().isoformat(),
                    "file_info": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": 0
                    },
                    "request_data": request.dict() if request else {}
                }
                
                background_tasks.add_task(
                    self._process_uploaded_file,
                    request_id,
                    file,
                    request or ProcessingRequest(source_system="auto")
                )
                
                return ProcessingResponse(
                    request_id=request_id,
                    status="accepted",
                    message="File uploaded successfully. Processing started in background.",
                    total_claims=0,
                    eligible_claims=0,
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.logger.error(f"Error in upload endpoint: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status/{request_id}")
        async def get_processing_status(request_id: str):
            if request_id not in self.processing_requests:
                raise HTTPException(status_code=404, detail="Request ID not found")
            
            request_info = self.processing_requests[request_id]
            return {
                "request_id": request_id,
                "status": request_info["status"],
                "progress": request_info["progress"],
                "message": request_info["message"],
                "timestamp": request_info["timestamp"]
            }
        
        @self.app.get("/results/{request_id}")
        async def get_processing_results(request_id: str):
            if request_id not in self.processing_requests:
                raise HTTPException(status_code=404, detail="Request ID not found")
            
            request_info = self.processing_requests[request_id]
            
            if request_info["status"] != "completed":
                raise HTTPException(
                    status_code=400, 
                    detail="Processing not completed yet"
                )
            
            return JSONResponse(content=request_info.get("results", {}))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def _process_uploaded_file(self, request_id: str, file: UploadFile, request: ProcessingRequest):
        try:
            self.processing_requests[request_id]["status"] = "processing"
            self.processing_requests[request_id]["progress"] = 10.0
            self.processing_requests[request_id]["message"] = "Reading uploaded file..."
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                file_path = temp_path / file.filename
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                self.processing_requests[request_id]["file_info"]["size"] = file_path.stat().st_size
                self.processing_requests[request_id]["progress"] = 20.0
                self.processing_requests[request_id]["message"] = "File saved, starting data processing..."
                
                start_time = datetime.now()
                data_cleaner = EMRDataCleaner()
                
                if file.filename.endswith('.csv'):
                    normalized_claims = data_cleaner.ingest_csv_data(file_path)
                elif file.filename.endswith('.json'):
                    normalized_claims = data_cleaner.ingest_json_data(file_path)
                else:
                    raise ValueError("Unsupported file type")
                
                self.processing_requests[request_id]["progress"] = 60.0
                self.processing_requests[request_id]["message"] = f"Data normalized, analyzing {len(normalized_claims)} claims..."
                
                analyzer = ResubmissionAnalyzer(reference_date=request.reference_date)
                eligible_claims, analysis_metrics = analyzer.analyze_all_claims(normalized_claims)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                self.processing_requests[request_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "message": "Processing completed successfully",
                    "timestamp": datetime.now().isoformat(),
                    "results": {
                        "total_claims": len(normalized_claims),
                        "eligible_claims": len(eligible_claims),
                        "processing_time": processing_time,
                        "analysis_metrics": analysis_metrics,
                        "resubmission_candidates": [
                            {
                                "claim_id": c.claim_id,
                                "resubmission_reason": c.resubmission_reason,
                                "source_system": c.source_system,
                                "recommended_changes": c.recommended_changes
                            }
                            for c in eligible_claims
                        ]
                    }
                })
                
                self.logger.info(f"Processing completed for request {request_id}: {len(eligible_claims)} eligible claims")
                
        except Exception as e:
            self.logger.error(f"Error processing file for request {request_id}: {str(e)}")
            self.processing_requests[request_id].update({
                "status": "failed",
                "progress": 0.0,
                "message": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })


# Create and configure the FastAPI app
api = EMRDataAPI()
app = api.app


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
