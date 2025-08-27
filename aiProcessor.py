import os
import base64
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Union, Tuple

# Import our Pydantic response schemas for structured AI outputs
from schemas import SummaryResponse, ConceptExplanationResponse, ErrorResponse

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

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract and comprehensively explain image content using GPT-4 Vision.
    
    This function acts as the first stage of the pipeline, where the AI:
    - Analyzes the uploaded image (notes, diagrams, textbook pages, etc.)
    - Extracts all text, equations, concepts, and visual elements
    - Provides detailed explanations as if teaching a student
    - Preserves technical terminology and mathematical notation
    
    Args:
        image_bytes: Raw bytes of the uploaded image file
        
    Returns:
        str: Comprehensive text explanation of the image content
    """
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
                            "and examples exactly as they appear in the image. Preserve all concepts, technical terms, "
                            "details and equations. "
                            "Avoid outside knowledge—only explain what is in the image itself. "
                            "Your output should feel like a teacher walking through the material, "
                            "not a summary."
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


def process_image_pipeline(image_bytes: bytes) -> Tuple[Union[SummaryResponse, ErrorResponse], str]:
    """
    Complete pipeline: OCR extraction → Summary generation.
    Returns tuple of (response, text_id) for later use.
    """
    extracted_text = extract_text_from_image(image_bytes)
    print("Extracted text from OCR:\n", extracted_text[:300], "...")  # Preview first 300 chars
    
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


