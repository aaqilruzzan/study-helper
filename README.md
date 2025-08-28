# AI Study Helper API

A FastAPI-based service that uses GPT-4 Vision to extract text from study material images and generate comprehensive summaries, explanations, and study guidance.

## Quick Start for Frontend Developers

### Base URL

```
http://localhost:8000 - local
https://study-helper-5bl7.onrender.com - hosted

```

### Health Check

```http
GET /
```

**Response:**

```json
{
  "status": "AI Study Helper API is running!"
}
```

---

## API Endpoints

### 1. Process Image → Generate Summary

**Endpoint:** `POST /api/process-image/`

**Description:** Upload an image of study materials (notes, textbook pages, diagrams) to extract text and generate a comprehensive summary.

**Request:**

- **Method:** POST
- **Content-Type:** `multipart/form-data`
- **Body:** Form data with image file

**Request Example (JavaScript):**

```javascript
const formData = new FormData();
formData.append("file", imageFile); // imageFile is a File object

const response = await fetch("http://localhost:8000/api/process-image/", {
  method: "POST",
  body: formData,
});

const result = await response.json();
```

**Success Response (200):**

```json
{
  "summary": "This content explains the fundamental principles of photosynthesis, including the role of chlorophyll in capturing light energy and converting carbon dioxide and water into glucose. The process occurs in two main stages: light-dependent reactions in the thylakoids and light-independent reactions in the stroma.",
  "text_id": "abc123def456"
}
```

**Error Response (400/500):**

```json
{
  "error": "File provided is not an image."
}
```

---

### 2. Generate Detailed Explanations

**Endpoint:** `POST /api/generate-explanations/`

**Description:** Generate detailed concept explanations, study tips, and learning approaches using the `text_id` from the previous request.

**Request:**

- **Method:** POST
- **Content-Type:** `application/json`
- **Body:** JSON with text_id

**Request Example (JavaScript):**

```javascript
const response = await fetch(
  "http://localhost:8000/api/generate-explanations/",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text_id: "abc123def456", // Use text_id from previous response
    }),
  }
);

const result = await response.json();
```

**Success Response (200):**

```json
{
  "explanations": [
    {
      "concept": "Photosynthesis",
      "explanation": "The process by which plants convert sunlight into energy using chlorophyll"
    },
    {
      "concept": "Chlorophyll",
      "explanation": "The green pigment in plants that captures light energy for photosynthesis"
    },
    {
      "concept": "Light-dependent reactions",
      "explanation": "The first stage of photosynthesis that occurs in thylakoids and requires direct sunlight"
    }
  ],
  "studyTips": [
    "Use active recall techniques",
    "Create concept maps",
    "Practice spaced repetition",
    "Teach concepts to others"
  ],
  "learningApproaches": [
    "Visual learners: Use diagrams",
    "Kinesthetic: Practice exercises",
    "Auditory: Discuss concepts",
    "Reading/Writing: Take notes"
  ]
}
```

**Error Response (500):**

```json
{
  "error": "Text ID not found. Please process the image first."
}
```

---

### 3. Generate Quiz Questions (All Formats)

**Endpoint:** `POST /api/generate-quiz/`

**Description:** Generate quiz questions in multiple formats (MCQ, QuickQA, and Flashcards) from previously extracted text using the text_id from the image processing step.

**Request:**

- **Method:** POST
- **Content-Type:** `application/json`
- **Body:** JSON with text_id

**Request Example (JavaScript):**

```javascript
const response = await fetch("/api/generate-quiz/", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text_id: "abc123def456", // Use text_id from previous response
  }),
});

const result = await response.json();
```

**Success Response (200):**

```json
{
  "MCQ": [
    {
      "question": "What is the primary function of chlorophyll in photosynthesis?",
      "answers": [
        {
          "answer": "Absorb light energy",
          "correct": true
        },
        {
          "answer": "Store water",
          "correct": false
        },
        {
          "answer": "Release oxygen",
          "correct": false
        },
        {
          "answer": "Break down glucose",
          "correct": false
        }
      ],
      "explanation": "Chlorophyll captures sunlight and converts it into chemical energy for the plant to use."
    }
  ],
  "QuickQA": [
    {
      "question": "What is the primary function of chlorophyll in photosynthesis?",
      "correct_answer": "Absorb light energy",
      "explanation": "Chlorophyll captures sunlight and converts it into chemical energy for the plant to use.",
      "other_correct_options": [
        "Capture sunlight",
        "Convert light energy",
        "Harvest light"
      ]
    }
  ],
  "Flashcards": [
    {
      "question": "What is the primary function of chlorophyll in photosynthesis?",
      "correctanswer": "Absorb light energy",
      "explanation": "Chlorophyll captures sunlight and converts it into chemical energy for the plant to use."
    }
  ]
}
```

**Note:** The response includes exactly 10 questions in each format with varying difficulty levels from basic recall to application/analysis.

**Error Response (500):**

```json
{
  "error": "Text ID not found. Please process the image first."
}
```

---

### 4. Generate Study Notes

**Endpoint:** `POST /api/generate-notes/`

**Description:** Generate structured study notes from previously extracted text using the text_id from the image processing step. Creates exactly 2 comprehensive study notes with metadata including difficulty, key points, and estimated reading time.

**Request:**

- **Method:** POST
- **Content-Type:** `application/json`
- **Body:** JSON with text_id

**Request Example (JavaScript):**

```javascript
const response = await fetch("/api/generate-notes/", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text_id: "abc123def456", // Use text_id from previous response
  }),
});

const result = await response.json();
```

**Success Response (200):**

```json
{
  "id": "abc123def456",
  "notes": [
    {
      "title": "Mathematical Foundations",
      "subject": "Calculus",
      "description": "Core concepts of differential and integral calculus explained through visual examples and real-world applications.",
      "content": "Calculus is a branch of mathematics that deals with rates of change and areas under curves. The fundamental concepts include derivatives, which measure how functions change at specific points, and integrals, which calculate accumulated quantities like areas and volumes. The Fundamental Theorem of Calculus elegantly connects these two concepts, showing that differentiation and integration are inverse operations. These mathematical tools are essential for understanding motion, optimization problems, and modeling real-world phenomena in physics, engineering, and economics.",
      "keyPoints": [
        "Derivatives measure rates of change",
        "Integrals calculate areas under curves",
        "Fundamental theorem connects both concepts",
        "Applications in physics and engineering"
      ],
      "difficulty": "Intermediate",
      "estimatedTime": "15 min read",
      "lastUpdated": "2 days ago"
    },
    {
      "title": "Problem Solving Techniques",
      "subject": "Mathematics",
      "description": "Systematic approaches to solving complex mathematical problems using step-by-step methodologies.",
      "content": "Effective problem solving in mathematics requires a structured approach that breaks complex problems into manageable steps. Start by carefully reading and understanding what the problem is asking, then identify the given information and what needs to be found. Look for patterns, relationships, and connections to previously learned concepts. Use visual representations like diagrams, graphs, or tables to organize information. Apply appropriate mathematical methods and formulas, then verify your solution by checking if it makes sense in the context of the original problem.",
      "keyPoints": [
        "Break complex problems into steps",
        "Identify patterns and relationships",
        "Use visual representations effectively",
        "Verify solutions through substitution"
      ],
      "difficulty": "Advanced",
      "estimatedTime": "20 min read",
      "lastUpdated": "1 day ago"
    }
  ]
}
```

**Note:** The response includes exactly 2 structured study notes with comprehensive metadata for effective learning organization.

**Error Response (500):**

```json
{
  "error": "Text ID not found. Please process the image first."
}
```

---
