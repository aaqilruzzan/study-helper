import os
import base64
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Union, Tuple

# Importing Pydantic response schemas for structured AI outputs
from schemas import SummaryResponse, ConceptExplanationResponse, QuizResponse, ErrorResponse, AllQuizFormatsResponse, NotesResponse, NotesWithIdResponse, format_quiz_to_mcq,format_quiz_to_quickqa,format_quiz_to_flashcards

# Importing AI prompts from constants
from constants import IMAGE_PROCESSING_PROMPT, MEGA_PROMPT, EXPLANATIONS_PROMPT, QUIZ_PROMPT, NOTES_PROMPT

# Loading environment variables from a .env file
load_dotenv()

# --- OpenAI Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the OpenAI client for GPT-4o-mini API calls
client = OpenAI(api_key=API_KEY)

# In-memory storage for extracted text content
# This allows reusing extracted text for multiple operations without reprocessing images
extracted_text_storage = {}


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract and comprehensively explain image content using GPT-4o-mini.
    
    This function acts as the first stage of the pipeline, where the AI:
    - Analyzes the uploaded image (notes, diagrams, textbook pages, etc.)
    - Extracts all text, equations, concepts, and visual elements
    - Provides detailed explanations as if teaching a student
    - Preserves technical terminology and mathematical notation
    - Returns structured error message if image is unreadable or irrelevant
    
    Args:
        image_bytes: Raw bytes of the uploaded image file
        
    Returns:
        str: Comprehensive text explanation of the image content or structured error message
    """
    try:
        # Encoding image to base64 format required by OpenAI Vision API
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,  # Low temperature for consistent, factual explanations
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": IMAGE_PROCESSING_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ],
        }
    ],
)

        explained_text = response.choices[0].message.content.strip()
        return explained_text
        
    except Exception as e:
        print(f"Error in extract_text_from_image: {e}")

        error_response = {
            "error": "IMAGE_PROCESSING_ERROR",
            "message": "Image cannot be processed due to technical issues. Please try again with a different image."
        }
        return json.dumps(error_response)



def generate_summary(extracted_text: str) -> Union[SummaryResponse, ErrorResponse]:
    """
    Second call: Using extracted OCR text and transforming it into a structured study guide.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": f"{MEGA_PROMPT}\n\nHere is the extracted text:\n{extracted_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "study_helper_response",
                    "schema": SummaryResponse.model_json_schema(),
                    "strict": True,
                },
            },
        )

        json_string = response.choices[0].message.content

        parsed_json = json.loads(json_string)
        validated_response = SummaryResponse(**parsed_json)
        return validated_response

    except ValidationError as ve:
        print(f"Validation error: {ve}")
        return ErrorResponse(error="The AI response format is invalid. Please try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ErrorResponse(error="An error occurred while generating the study guide.")


def generate_explanations(extracted_text: str) -> Union[ConceptExplanationResponse, ErrorResponse]:
    """
    Generating concept explanations, study tips, and learning approaches from extracted text.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": f"{EXPLANATIONS_PROMPT}\n\nHere is the study material content:\n{extracted_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "concept_explanation_response",
                    "schema": ConceptExplanationResponse.model_json_schema(),
                    "strict": True,
                },
            },
        )

        json_string = response.choices[0].message.content

        parsed_json = json.loads(json_string)
        validated_response = ConceptExplanationResponse(**parsed_json)
        return validated_response

    except ValidationError as ve:
        print(f"Validation error: {ve}")
        return ErrorResponse(error="The AI response format is invalid. Please try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ErrorResponse(error="An error occurred while generating explanations.")


def generate_notes(extracted_text: str) -> Union[NotesResponse, ErrorResponse]:
    """
    Generating structured study notes from extracted text content using GPT-4o-mini with JSON mode.
    
    This function uses the NotesResponse Pydantic model to ensure structured output
    and validate the AI's response format for notes generation.
    
    Args:
        extracted_text: The extracted text content from the image processing stage
        
    Returns:
        NotesResponse: Contains exactly 2 structured study notes with metadata
        ErrorResponse: If generation fails or validation errors occur
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": f"{NOTES_PROMPT}\n\nHere is the study material content:\n{extracted_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "notes_response",
                    "schema": NotesResponse.model_json_schema(),
                    "strict": True,
                },
            },
        )

        json_string = response.choices[0].message.content

        parsed_json = json.loads(json_string)
        validated_response = NotesResponse(**parsed_json)
        return validated_response

    except ValidationError as ve:
        print(f"Validation error: {ve}")
        return ErrorResponse(error="The AI response format is invalid. Please try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ErrorResponse(error="An error occurred while generating notes.")


def process_image_pipeline(image_bytes: bytes) -> Tuple[Union[SummaryResponse, ErrorResponse], str]:
    """
    Complete pipeline: OCR extraction → Summary generation.
    Returns tuple of (response, text_id) for later use.
    Handles image processing errors and returns appropriate error responses.
    """
    extracted_text = extract_text_from_image(image_bytes)
    print("Extracted text from OCR:\n", extracted_text[:300], "...")  # Preview first 300 chars
    
    # Checking if the extracted text is an error response
    try:
        parsed_response = json.loads(extracted_text)
        if parsed_response.get("error") == "IMAGE_PROCESSING_ERROR":
            return ErrorResponse(error=parsed_response.get("message", "Image processing failed")), ""
    except json.JSONDecodeError:
        # Not JSON, so it's normal text content - continue processing
        pass
    
    # Generating a unique ID for this extracted text
    text_id = hashlib.md5(extracted_text.encode()).hexdigest()[:16]
    extracted_text_storage[text_id] = extracted_text
    
    summary_response = generate_summary(extracted_text)
    return summary_response, text_id


def process_explanations_pipeline(text_id: str) -> Union[ConceptExplanationResponse, ErrorResponse]:
    """
    Pipeline for generating explanations from already extracted text using text_id.
    """
    if text_id not in extracted_text_storage:
        return ErrorResponse(error="Text ID not found. Please process the image first.")
    
    extracted_text = extracted_text_storage[text_id]
    return generate_explanations(extracted_text)


def process_notes_pipeline(text_id: str) -> Union[NotesWithIdResponse, ErrorResponse]:
    """
    Pipeline for generating structured notes from already extracted text using text_id.
    Returns the notes with the text_id as the id field.
    """
    if text_id not in extracted_text_storage:
        return ErrorResponse(error="Text ID not found. Please process the image first.")
    
    extracted_text = extracted_text_storage[text_id]
    notes_result = generate_notes(extracted_text)
    
    if isinstance(notes_result, ErrorResponse):
        return notes_result
    
    notes_with_id = NotesWithIdResponse(
        id=text_id,
        notes=notes_result.notes
    )
    
    return notes_with_id


def generate_quiz(extracted_text: str) -> Union[QuizResponse, ErrorResponse]:
    """
    Generating quiz questions from extracted text content using GPT-4o-mini with JSON mode.

    This function uses the QuizResponse Pydantic model to ensure structured output
    and validate the AI's response format for quiz generation.
    
    Args:
        text: The extracted text content from the image processing stage
        
    Returns:
        QuizResponse: Contains list of 10 quiz questions with answers and explanations
        ErrorResponse: If generation fails or validation errors occur
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": f"{QUIZ_PROMPT}\n\nHere is the study material content:\n{extracted_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "quiz_response",
                    "schema": QuizResponse.model_json_schema(),
                    "strict": True,
                },
            },
        )
        
        # Parse and validate using Pydantic model
        response_data = json.loads(response.choices[0].message.content)
        return QuizResponse(**response_data)
        
    except ValidationError as ve:
        print(f"Validation error: {ve}")
        return ErrorResponse(error="The AI response format is invalid. Please try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ErrorResponse(error="An error occurred while generating quiz questions.")


def process_quiz_pipeline(text_id: str) -> Union[AllQuizFormatsResponse, ErrorResponse]:
    """
    Pipeline for generating quiz questions in all formats from already extracted text using text_id.
    Returns MCQ, QuickQA, and Flashcards formats in one combined response.
    """
    if text_id not in extracted_text_storage:
        return ErrorResponse(error="Text ID not found. Please process the image first.")
    
    extracted_text = extracted_text_storage[text_id]
    quizResponse = generate_quiz(extracted_text)
    
    if isinstance(quizResponse, ErrorResponse):
        return quizResponse
    
    try:
        # Generating all three formats
        mcqResponse = format_quiz_to_mcq(quizResponse)
        quickQAResponse = format_quiz_to_quickqa(quizResponse)
        flashcardsResponse = format_quiz_to_flashcards(quizResponse)

        # Combining all formats into one response
        combined_response = AllQuizFormatsResponse(
            MCQ=mcqResponse.MCQ,
            QuickQA=quickQAResponse.QuickQA,
            Flashcards=flashcardsResponse.Flashcards
        )
        
        return combined_response
        
    except Exception as e:
        print(f"Error formatting quiz responses: {e}")
        return ErrorResponse(error="An error occurred while formatting the quiz into different formats.")


