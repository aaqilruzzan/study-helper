import os
import base64
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Union, Tuple

# Import our Pydantic response schemas for structured AI outputs
from schemas import SummaryResponse, ConceptExplanationResponse, QuizResponse, ErrorResponse, AllQuizFormatsResponse, NotesResponse, NotesWithIdResponse, format_quiz_to_mcq,format_quiz_to_quickqa,format_quiz_to_flashcards

# Load environment variables from a .env file
# Ensure you have created a .env file with your OPENAI_API_KEY
load_dotenv()

# --- OpenAI Configuration ---
# Using environment variables for API keys follows security best practices
# and prevents accidental exposure of sensitive credentials in source code
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the OpenAI client for GPT-4 Vision API calls
client = OpenAI(api_key=API_KEY)

# In-memory storage for extracted text content (use Redis/database in production)
# This allows reusing extracted text for multiple operations without reprocessing images
extracted_text_storage = {}

# --- AI Prompts for Different Processing Stages ---

# Summary Generation Prompt - Used for initial text extraction and summarization
MEGA_PROMPT = """
You are a friendly and encouraging AI tutor. A student has uploaded an image of their study material to help them prepare for a test. Your goal is to transform the extracted content into a powerful and easy-to-understand study guide.

Return your response as a valid JSON object matching the required schema.
"""

# Explanations Generation Prompt - Used for detailed concept breakdowns and study guidance
EXPLANATIONS_PROMPT = """
You are an expert AI tutor. Based on the provided study material content, generate detailed explanations and study guidance.

Your task:
1. **Explanations**: Identify up to 5 key concepts from the content and explain them in simple, easy-to-understand language
2. **Study Tips**: Provide 4 practical study techniques specifically tailored to this content 
3. **Learning Approaches**: Suggest 4 specific approaches for different learning styles (Visual, Kinesthetic, Auditory, Reading/Writing)

Focus on being practical and actionable. Make the explanations clear enough for any student to understand.

Return your response as a valid JSON object matching the required schema.
"""

# Quiz Generation Prompt - Used for creating quiz questions from extracted text
QUIZ_PROMPT = """
You are a patient teacher whose goal is to help the student gain a clear and good overall understanding of the text. 

Task: Create a pool of 10 high-quality questions based ONLY on the provided text. These 10 questions should attempt to cover all key concepts and flows within the text so that the student gets a complete picture.

Return your response as a valid JSON object matching the required schema.
"""


ADDITIONAL_CONTENT_PROMPT = """
You are a patient teacher whose goal is to help the student gain a clear and good overall understanding of the text. 

Task: Create 2 high-quality notes based ONLY on the provided text. These 2 notes should attempt to cover some key concepts and flows within the text so that the student can review them later effectively for important memory retention.

Return your response as a valid JSON object matching the required schema.
"""


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract and comprehensively explain image content using GPT-4 Vision.
    
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
        # Encode image to base64 format required by OpenAI Vision API
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
                    "text": (
                        "Look at this image and explain everything it contains as if you are "
                        "teaching it to a student. Do not just summarize or list topics—break it down "
                        "step by step, clearly explaining concepts, definitions, equations, diagrams, "
                        "and examples exactly as they appear in the image. Preserve all concepts, "
                        "technical terms, details, and equations. "
                        "Avoid outside knowledge—only explain what is in the image itself. "
                        "Your output should feel like a teacher walking through the material, "
                        "not a summary.\n\n"
                        "⚠️ Important: Only if the image cannot be processed at all "
                        "(because of lack of visibility, unreadable quality, or irrelevant non-study content), "
                        "then respond with exactly this JSON structure and nothing else:\n"
                        '{"error": "IMAGE_PROCESSING_ERROR", "message": "Image cannot be processed due to lack of visibility, poor image quality, or irrelevant content that is not study material. Please try again with a clearer image of study materials."}'
                    )
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
        # Return a structured error message if something goes wrong
        error_response = {
            "error": "IMAGE_PROCESSING_ERROR",
            "message": "Image cannot be processed due to technical issues. Please try again with a different image."
        }
        return json.dumps(error_response)



def generate_summary(extracted_text: str) -> Union[SummaryResponse, ErrorResponse]:
    """
    Second call: Use extracted OCR text and transform it into a structured study guide.
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
    Generate concept explanations, study tips, and learning approaches from extracted text.
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
    Generate structured study notes from extracted text content using GPT-4 with JSON mode.
    
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
                    "content": f"{ADDITIONAL_CONTENT_PROMPT}\n\nHere is the study material content:\n{extracted_text}"
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
    
    # Check if the extracted text is an error response
    try:
        parsed_response = json.loads(extracted_text)
        if parsed_response.get("error") == "IMAGE_PROCESSING_ERROR":
            # Return the error message from the image processing
            return ErrorResponse(error=parsed_response.get("message", "Image processing failed")), ""
    except json.JSONDecodeError:
        # Not JSON, so it's normal text content - continue processing
        pass
    
    # Generate a unique ID for this extracted text
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
    
    # Check if notes generation failed
    if isinstance(notes_result, ErrorResponse):
        return notes_result
    
    # Wrap the notes with the text_id as id
    notes_with_id = NotesWithIdResponse(
        id=text_id,
        notes=notes_result.notes
    )
    
    return notes_with_id


def generate_quiz(extracted_text: str) -> Union[QuizResponse, ErrorResponse]:
    """
    Generate quiz questions from extracted text content using GPT-4 with JSON mode.
    
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
    
    # Check if quiz generation failed
    if isinstance(quizResponse, ErrorResponse):
        return quizResponse
    
    try:
        # Generate all three formats
        mcqResponse = format_quiz_to_mcq(quizResponse)
        quickQAResponse = format_quiz_to_quickqa(quizResponse)
        flashcardsResponse = format_quiz_to_flashcards(quizResponse)
        
        # Combine all formats into one response
        combined_response = AllQuizFormatsResponse(
            MCQ=mcqResponse.MCQ,
            QuickQA=quickQAResponse.QuickQA,
            Flashcards=flashcardsResponse.Flashcards
        )
        
        return combined_response
        
    except Exception as e:
        print(f"Error formatting quiz responses: {e}")
        return ErrorResponse(error="An error occurred while formatting the quiz into different formats.")


