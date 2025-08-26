import os
import base64
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Union, Tuple

# Import our Pydantic schemas
from schemas import SummaryResponse, ConceptExplanationResponse, ErrorResponse

# Load environment variables from a .env file
# Make sure to create a .env file with your OPENAI_API_KEY
load_dotenv()

# --- Configuration ---
# It's best practice to use environment variables for API keys
# to avoid hardcoding them in your source code.
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# Simple in-memory storage for extracted text (in production, use a proper database)
extracted_text_storage = {}

# --- The Mega Prompt ---
# Updated prompt for JSON mode - cleaner and more direct
MEGA_PROMPT = """
You are a friendly and encouraging AI tutor. A student has uploaded an image of their study material to help them prepare for a test. Your goal is to transform the extracted content into a powerful and easy-to-understand study guide.


Return your response as a valid JSON object matching the required schema.
"""

# --- Explanations Prompt ---
# Prompt specifically for generating concept explanations, study tips, and learning approaches
EXPLANATIONS_PROMPT = """
You are an expert AI tutor. Based on the provided study material content, generate detailed explanations and study guidance.

Focus on being practical and actionable. Make the explanations clear enough for any student to understand.

Return your response as a valid JSON object matching the required schema.
"""

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract and explain the image content using GPT-4o-mini.
    The model should act like a teacher, breaking down and explaining
    all details in the image clearly and thoroughly.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,  # allow slightly richer explanations
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


