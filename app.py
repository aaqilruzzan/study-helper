from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from pydantic import BaseModel
import uvicorn

# Import the core AI processing function and schemas
from aiProcessor import process_image_pipeline, process_explanations_pipeline
from schemas import SummaryResponse, SummaryWithIdResponse, ConceptExplanationResponse, ErrorResponse

# Request model for explanations endpoint
class ExplanationsRequest(BaseModel):
    text_id: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Study Helper API",
    description="An API that uses GPT-4 Vision to generate summaries, explanations, and quizzes from images.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This is crucial for allowing your frontend application (running on a different domain)
# to make requests to this backend API.
# The wildcard "*" is permissive for development, but for production, you might
# want to restrict it to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# TODO:Add image type and size validations by referring to OPEN AI documentation
# --- API Endpoint Definition ---
@app.post("/api/process-image/", response_model=Union[SummaryWithIdResponse, ErrorResponse])
async def create_upload_file(file: UploadFile = File(...)):
    """
    This endpoint receives an image file, processes it using the AI,
    and returns the structured study helper content.
    """
    print("Received image fi:", file.filename)
    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the contents of the uploaded file into bytes
        image_bytes = await file.read()

        # Call the function that interacts with the GPT-4 API
        result, text_id = process_image_pipeline(image_bytes)

        # Check if the result is an ErrorResponse
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)

        # Create a new SummaryWithIdResponse with the text_id included
        response_with_id = SummaryWithIdResponse(
            summary=result.summary,
            text_id=text_id
        )
        
        # Return the successful validated response with text_id
        return response_with_id

    except Exception as e:
        # Catch any other unexpected errors during file processing or the API call
        print(f"An error occurred in the endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Explanations Endpoint ---
@app.post("/api/generate-explanations/", response_model=Union[ConceptExplanationResponse, ErrorResponse])
async def generate_explanations(request: ExplanationsRequest):
    """
    This endpoint generates concept explanations, study tips, and learning approaches
    from previously extracted text using the text_id.
    """
    try:
        # Call the function that generates explanations
        result = process_explanations_pipeline(request.text_id)

        # Check if the result is an ErrorResponse
        if isinstance(result, ErrorResponse):
            raise HTTPException(status_code=500, detail=result.error)

        # Return the successful validated response
        return result

    except Exception as e:
        # Catch any other unexpected errors
        print(f"An error occurred in the explanations endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "AI Study Helper API is running!"}


# --- Running the Application ---
# This block allows you to run the server directly using `python app.py`
# Uvicorn is a lightning-fast ASGI server with reload enabled for development.
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

