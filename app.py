from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from pydantic import BaseModel
import uvicorn

# Importing the core AI processing functions and response schemas
from aiProcessor import process_image_pipeline, process_explanations_pipeline, process_quiz_pipeline, process_notes_pipeline
from schemas import ExplanationsRequest, NotesRequest, QuizRequest, SummaryResponse, SummaryWithIdResponse, ConceptExplanationResponse, NotesWithIdResponse, ErrorResponse, AllQuizFormatsResponse

# Importing file validation constants
from constants import MAX_FILE_SIZE, ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Study Helper API",
    description="An API that uses GPT-4o-mini to extract text from images and generate summaries, explanations, and study guidance.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# --- Image Processing Endpoint ---
@app.post("/api/process-image/", response_model=Union[SummaryWithIdResponse, ErrorResponse])
async def create_upload_file(file: UploadFile = File(...)):
    """
    This endpoint receives an image file, extracts text using GPT-4o-mini,
    generates a comprehensive summary, and returns it along with a text_id
    for future explanations generation.
    
    Validation rules:
    - Maximum file size: 10MB
    - Supported formats: PNG, JPEG, WEBP, non-animated GIF
    """
    print("Received image file:", file.filename)
    
    # Validating file content type
    if not file.content_type or file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: PNG, JPEG, WEBP, GIF. Received: {file.content_type}"
        )
    
    # Validating file extension (additional check)
    if file.filename:
        file_extension = file.filename.lower().split('.')[-1]
        if f".{file_extension}" not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    # Reading file content to check size
    image_bytes = await file.read()
    file_size = len(image_bytes)

    # Validating file size
    if file_size > MAX_FILE_SIZE:
        file_size_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed: 10MB. Your file: {file_size_mb:.2f}MB"
        )
    
    # Additional check for minimum file size (avoiding empty files)
    if file_size < 1024:  # Less than 1KB
        raise HTTPException(
            status_code=400,
            detail="File too small. Please upload a valid image file."
        )

    try:
        # File content is already read during validation, so we can use image_bytes directly
        # Calling the image processing pipeline that extracts text and generates summary
        result, text_id = process_image_pipeline(image_bytes)

        # Creating a response object that includes both summary and text_id
        response_with_id = SummaryWithIdResponse(
            summary=result.summary,
            text_id=text_id
        )

        # Returning the successful response with text_id for future explanations
        return response_with_id

    except Exception as e:
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)
        
        print(f"An error occurred in the image processing endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Explanations Generation Endpoint ---
@app.post("/api/generate-explanations/", response_model=Union[ConceptExplanationResponse, ErrorResponse])
async def generate_explanations(request: ExplanationsRequest):
    """
    This endpoint generates detailed concept explanations, study tips, and learning approaches
    from previously extracted text using the text_id obtained from the image processing endpoint.
    This avoids reprocessing the same image and provides faster responses.
    """
    try:
        # Calling the explanations pipeline using the stored extracted text
        result = process_explanations_pipeline(request.text_id)

        # Checking if the result is an ErrorResponse
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)

        # Returning the successful explanations response
        return result

    except Exception as e:
        # Catching any other unexpected errors during explanations generation
        print(f"An error occurred in the explanations generation endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Quiz Generation Endpoint ---
@app.post("/api/generate-quiz/", response_model=Union[AllQuizFormatsResponse, ErrorResponse])
async def generate_quiz(request: QuizRequest):
    """
    This endpoint generates quiz questions from previously extracted text using the text_id 
    obtained from the image processing endpoint. Creates 10 questions with answers and explanations
    based on the content of the processed study material.
    """
    try:
        # Calling the quiz generation pipeline using the stored extracted text
        result = process_quiz_pipeline(request.text_id)

        # Checking if the result is an ErrorResponse
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)

        # Returning the successful quiz response
        return result

    except Exception as e:
        # Catching any other unexpected errors during quiz generation
        print(f"An error occurred in the quiz generation endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Notes Generation Endpoint ---
@app.post("/api/generate-notes/", response_model=Union[NotesWithIdResponse, ErrorResponse])
async def generate_notes(request: NotesRequest):
    """
    This endpoint generates structured study notes from previously extracted text using the text_id 
    obtained from the image processing endpoint. Creates exactly 2 comprehensive study notes
    with metadata including difficulty, key points, and estimated reading time.
    """
    try:
        # Calling the notes generation pipeline using the stored extracted text
        result = process_notes_pipeline(request.text_id)

        # Checking if the result is an ErrorResponse
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)

        # Returning the successful notes response
        return result

    except Exception as e:
        # Catching any other unexpected errors during notes generation
        print(f"An error occurred in the notes generation endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "AI Study Helper API is running!"}


# --- Running the Application ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

