from pydantic import BaseModel, Field, ConfigDict
from typing import List


class ConceptExplanation(BaseModel):
    """Model for individual concept explanations."""
    model_config = ConfigDict(extra='forbid')
    
    concept: str = Field(..., description="The key concept being explained. Example: 'Photosynthesis' or 'Newton's First Law'")
    explanation: str = Field(..., description="Simple explanation of the concept in easy-to-understand language. Example: 'The process by which plants convert sunlight into energy' or 'An object at rest stays at rest unless acted upon by a force'")

#TODO: Ensure formula formatting remove the //s and remove the bold **s
class SummaryResponse(BaseModel):
    """Main response model for the study helper API."""
    model_config = ConfigDict(extra='forbid')
    
    summary: str = Field(
    ...,
    description=(
        "Act as a teacher and fully explain the content of the text in clear, simple language. "
        "Do not just summarize or point out topics—break down every concept, definition, equation, "
        "and example step by step. Preserve all technical details from the text while making it "
        "easy to understand, as if explaining to a student. "
        "Keep the output to a maximum of 100 words."
    )
)


class SummaryWithIdResponse(BaseModel):
    """API response model that includes both summary and text_id."""
    model_config = ConfigDict(extra='forbid')
    
    summary: str = Field(..., description="The generated summary from the LLM")
    text_id: str = Field(..., description="Unique identifier for the extracted text, used for generating explanations later")


class ConceptExplanationResponse(BaseModel):
    """Response model for concept explanations, study tips, and learning approaches."""
    model_config = ConfigDict(extra='forbid')
    
    explanations: List[ConceptExplanation] = Field(
        ..., 
        description="List of key concepts and their explanations. Each should have a 'concept' (the key term) and 'explanation' (simple definition).",
        min_length=1,
        max_length=5
    )
    studyTips: List[str] = Field(
        ..., 
        description="List of 4 practical study techniques specific to the content. Format examples: 'Use active recall techniques', 'Create concept maps', 'Practice spaced repetition', 'Teach concepts to others' Limit each output to 4 words",
        min_length=4,
        max_length=4
    )
    learningApproaches: List[str] = Field(
        ..., 
        description="List of 4 learning approaches for different learning styles. Format examples: 'Visual learners: Use diagrams', 'Kinesthetic: Practice exercises', 'Auditory: Discuss concepts', 'Reading/Writing: Take notes' Limit each output to 4 words",
        min_length=4,
        max_length=4
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    """Model for error responses."""
    error: str = Field(..., description="Error message describing what went wrong")
