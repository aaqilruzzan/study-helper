"""
Pydantic schemas for the AI Study Helper API.

This module defines the data models used for:
1. LLM response validation (ensuring AI outputs match expected structure)
2. API response formatting (what clients receive)
3. Request validation (ensuring valid inputs)

Uses Pydantic v2 with ConfigDict for strict schema validation required by OpenAI's structured outputs.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class ConceptExplanation(BaseModel):
    """
    Individual concept explanation model used within larger response objects.
    
    This represents a single key concept and its simplified explanation,
    designed to help students understand complex topics.
    """
    model_config = ConfigDict(extra='forbid')  # Prevents additional fields for strict OpenAI compliance
    
    concept: str = Field(..., description="The key concept being explained. Example: 'Photosynthesis' or 'Newton's First Law'. Use plain text only - no LaTeX (//, \\\\), no bold markdown (**text**).")
    explanation: str = Field(..., description="Simple explanation of the concept in easy-to-understand language. Example: 'The process by which plants convert sunlight into energy' or 'An object at rest stays at rest unless acted upon by a force'. Use plain text only - no LaTeX (//, \\\\), no bold markdown (**text**).")


# TODO: Ensure formula formatting - remove LaTeX //s and bold **s from AI responses
class SummaryResponse(BaseModel):
    """
    LLM response model for summary generation (used by OpenAI JSON mode).
    
    This schema is sent to the LLM to ensure consistent response structure.
    Contains only fields that the AI should generate - no metadata.
    """
    model_config = ConfigDict(extra='forbid')  # Required for OpenAI structured outputs
    
    summary: str = Field(
        ...,
        description=(
            "Act as a teacher and fully explain the content of the text in clear, simple language. "
            "Do not just summarize or point out topics—break down every concept, definition, equation, "
            "and example step by step. Preserve all technical details from the text while making it "
            "easy to understand, as if explaining to a student. "
            "Keep the output to a maximum of 100 words. "
            "IMPORTANT: Use plain text only - no LaTeX formatting (//, \\\\), no bold markdown (**text**), "
        )
    )


class SummaryWithIdResponse(BaseModel):
    """
    API response model that wraps the LLM summary with additional metadata.
    
    This is what the client receives - includes both the AI-generated summary
    and a text_id for future operations (explanations generation).
    """
    model_config = ConfigDict(extra='forbid')
    
    summary: str = Field(..., description="The generated summary from the LLM")
    text_id: str = Field(..., description="Unique identifier for the extracted text, used for generating explanations later")


class ConceptExplanationResponse(BaseModel):
    """
    LLM response model for detailed explanations and study guidance (used by OpenAI JSON mode).
    
    This schema is used when generating concept explanations, study tips, and learning approaches
    from previously extracted text. Provides comprehensive study assistance beyond just summaries.
    """
    model_config = ConfigDict(extra='forbid')  # Required for OpenAI structured outputs
    
    explanations: List[ConceptExplanation] = Field(
        ..., 
        description="List of key concepts and their explanations. Each should have a 'concept' (the key term) and 'explanation' (simple definition). Use plain text only - no LaTeX or markdown formatting.",
        min_length=1,
        max_length=5
    )
    studyTips: List[str] = Field(
        ..., 
        description="List of 4 practical study techniques specific to the content. Format examples: 'Use active recall techniques', 'Create concept maps', 'Practice spaced repetition', 'Teach concepts to others' Limit each output to 4 words. Use plain text only - no formatting symbols.",
        min_length=4,
        max_length=4
    )
    learningApproaches: List[str] = Field(
        ..., 
        description="List of 4 learning approaches for different learning styles. Format examples: 'Visual learners: Use diagrams', 'Kinesthetic: Practice exercises', 'Auditory: Discuss concepts', 'Reading/Writing: Take notes' Limit each output to 4 words. Use plain text only - no formatting symbols.",
        min_length=4,
        max_length=4
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API endpoints.
    
    Used when processing fails due to invalid inputs, AI errors, or system issues.
    Provides consistent error messaging across the entire API.
    """
    model_config = ConfigDict(extra='forbid')
    
    error: str = Field(..., description="Error message describing what went wrong")
