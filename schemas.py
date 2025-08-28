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


class QuizQuestion(BaseModel):
    """
    Individual quiz question model.
    
    Represents a single question with its answer and explanation.
    """
    model_config = ConfigDict(extra='forbid')
    
    id: int = Field(..., description="Unique identifier for the question (1-10)")
    question: str = Field(..., description="The quiz question text. Must be descriptive and self-contained. Base questions ONLY on the provided text content - do not add outside knowledge. Vary question difficulty from basic recall to application/analysis. Maximum output is 10 words.")
    answer: str = Field(..., description="Short correct answer, maximum 6 words.")
    explanation: str = Field(..., description="Explanation of the correct answer, maximum 20 words.")
    incorrect_answers: List[str] = Field(..., description="Array of 3 plausible but incorrect answers, each maximum 6 words. These should be believable distractors related to the topic.", min_length=3, max_length=3)
    other_correct_options: List[str] = Field(..., description="Array of 3 alternative ways to phrase the correct answer, each maximum 6 words. These are acceptable alternative correct answers.", min_length=3, max_length=3)


class QuizResponse(BaseModel):
    """
    LLM response model for quiz generation (used by OpenAI JSON mode).
    
    Contains a list of 10 high-quality questions based on the extracted text.
    """
    model_config = ConfigDict(extra='forbid')  # Required for OpenAI structured outputs
    
    questions: List[QuizQuestion] = Field(
        ...,
        description="List of exactly 10 quiz questions based on the provided text. . Use plain text only - no LaTeX formatting (//, \\\\), no bold markdown (**text**).",
        min_length=10,
        max_length=10
    )


# --- Formatted Quiz Output Models ---

class MCQAnswer(BaseModel):
    """
    Individual answer option for MCQ format.
    """
    model_config = ConfigDict(extra='forbid')
    
    answer: str = Field(..., description="The answer option text")
    correct: bool = Field(..., description="Whether this answer is correct")


class MCQQuestion(BaseModel):
    """
    Multiple Choice Question format.
    """
    model_config = ConfigDict(extra='forbid')
    
    question: str = Field(..., description="The question text")
    answers: List[MCQAnswer] = Field(..., description="List of answer options with correct/incorrect flags")
    explanation: str = Field(..., description="Explanation of the correct answer")


class MCQResponse(BaseModel):
    """
    Response model for MCQ formatted quiz.
    """
    model_config = ConfigDict(extra='forbid')
    
    MCQ: List[MCQQuestion] = Field(..., description="List of multiple choice questions")


class QuickQAQuestion(BaseModel):
    """
    Quick Q&A Question format.
    """
    model_config = ConfigDict(extra='forbid')
    
    question: str = Field(..., description="The question text")
    correct_answer: str = Field(..., description="The correct answer")
    explanation: str = Field(..., description="Explanation of the correct answer")
    other_correct_options: List[str] = Field(..., description="Alternative correct answer phrasings")


class QuickQAResponse(BaseModel):
    """
    Response model for QuickQA formatted quiz.
    """
    model_config = ConfigDict(extra='forbid')
    
    QuickQA: List[QuickQAQuestion] = Field(..., description="List of quick Q&A questions")


class FlashcardQuestion(BaseModel):
    """
    Flashcard Question format.
    """
    model_config = ConfigDict(extra='forbid')
    
    question: str = Field(..., description="The question text")
    correctanswer: str = Field(..., description="The correct answer")
    explanation: str = Field(..., description="Explanation of the correct answer")


class FlashcardsResponse(BaseModel):
    """
    Response model for Flashcards formatted quiz.
    """
    model_config = ConfigDict(extra='forbid')
    
    Flashcards: List[FlashcardQuestion] = Field(..., description="List of flashcard questions")


class AllQuizFormatsResponse(BaseModel):
    """
    Combined response model containing all quiz formats.
    """
    model_config = ConfigDict(extra='forbid')
    
    MCQ: List[MCQQuestion] = Field(..., description="List of multiple choice questions")
    QuickQA: List[QuickQAQuestion] = Field(..., description="List of quick Q&A questions")
    Flashcards: List[FlashcardQuestion] = Field(..., description="List of flashcard questions")


# --- Notes Generation Models ---

class Note(BaseModel):
    """
    Individual note model for study notes generation.
    """
    model_config = ConfigDict(extra='forbid')
    
    title: str = Field(..., description="Title of the note section. Maximum output is 3 words.")
    subject: str = Field(..., description="Subject or topic area of the note. Maximum output is 3 words.")
    description: str = Field(..., description="Brief description of what the note covers. Maximum output is 17 words.")
    content: str = Field(..., description="The main content of the note, Maximum output is 100 words.")
    keyPoints: List[str] = Field(..., description="List of key points or concepts covered in this note. Per output maximum is 5 words", min_length=3, max_length=6)
    difficulty: str = Field(..., description="Difficulty level: 'Beginner', 'Intermediate', or 'Advanced'")
    estimatedTime: str = Field(..., description="Estimated reading time (e.g., '15 min read')")
    lastUpdated: str = Field(..., description="When the note was last updated (e.g., '2 days ago')")


class NotesResponse(BaseModel):
    """
    Response model for notes generation containing 2 generated notes.
    """
    model_config = ConfigDict(extra='forbid')
    
    notes: List[Note] = Field(..., description="List of exactly 2 generated study notes. Use plain text only - no LaTeX formatting (//, \\\\), no bold markdown (**text**).", min_length=2, max_length=2)


class NotesWithIdResponse(BaseModel):
    """
    API response model that wraps the notes with the text_id as id.
    """
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="The text_id used to generate these notes")
    notes: List[Note] = Field(..., description="List of exactly 2 generated study notes", min_length=2, max_length=2)
    

# --- Quiz Formatting Functions ---

def format_quiz_to_mcq(quiz_response: QuizResponse) -> MCQResponse:
    """
    Convert QuizResponse to MCQ format.
    
    Takes the raw quiz data and formats it into multiple choice questions
    where each question has 4 options (1 correct + 3 incorrect).
    
    Args:
        quiz_response: The original quiz response from the AI
        
    Returns:
        MCQResponse: Formatted multiple choice questions
    """
    import random
    
    mcq_questions = []
    
    for q in quiz_response.questions:
        # Create answer options: 1 correct + 3 incorrect
        answers = []
        
        # Add the correct answer
        answers.append(MCQAnswer(answer=q.answer, correct=True))
        
        # Add the incorrect answers
        for incorrect in q.incorrect_answers:
            answers.append(MCQAnswer(answer=incorrect, correct=False))
        
        # Shuffle the answers so correct answer isn't always first
        random.shuffle(answers)
        
        # Create MCQ question
        mcq_question = MCQQuestion(
            question=q.question,
            answers=answers,
            explanation=q.explanation
        )
        
        mcq_questions.append(mcq_question)
    
    return MCQResponse(MCQ=mcq_questions)


def format_quiz_to_quickqa(quiz_response: QuizResponse) -> QuickQAResponse:
    """
    Convert QuizResponse to QuickQA format.
    
    Takes the raw quiz data and formats it into quick Q&A structure
    with question, correct answer, explanation, and other correct options.
    
    Args:
        quiz_response: The original quiz response from the AI
        
    Returns:
        QuickQAResponse: Formatted quick Q&A questions
    """
    quickqa_questions = []
    
    for q in quiz_response.questions:
        quickqa_question = QuickQAQuestion(
            question=q.question,
            correct_answer=q.answer,
            explanation=q.explanation,
            other_correct_options=q.other_correct_options
        )
        
        quickqa_questions.append(quickqa_question)
    
    return QuickQAResponse(QuickQA=quickqa_questions)


def format_quiz_to_flashcards(quiz_response: QuizResponse) -> FlashcardsResponse:
    """
    Convert QuizResponse to Flashcards format.
    
    Takes the raw quiz data and formats it into flashcard structure
    with question, correct answer, and explanation.
    
    Args:
        quiz_response: The original quiz response from the AI
        
    Returns:
        FlashcardsResponse: Formatted flashcard questions
    """
    flashcard_questions = []
    
    for q in quiz_response.questions:
        flashcard_question = FlashcardQuestion(
            question=q.question,
            correctanswer=q.answer,
            explanation=q.explanation
        )
        
        flashcard_questions.append(flashcard_question)
    
    return FlashcardsResponse(Flashcards=flashcard_questions)
