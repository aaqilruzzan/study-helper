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
