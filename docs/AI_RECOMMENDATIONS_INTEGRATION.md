# AI Recommendations Integration Guide

This document explains how the AI recommendations feature is integrated between the frontend and backend.

## System Architecture

```
Frontend (ai.html + ai.js) → Flask API (/ai_recommend) → ML Model → JSON Response → Frontend Display
```

## Backend Implementation

### New API Endpoint: `/ai_recommend`

The Flask server now includes a new endpoint at `POST /ai_recommend` that:

1. Receives form data from the frontend
2. Maps the form fields to a user profile compatible with the ML model
3. Creates a temporary user in the dataset
4. Uses the ML model to generate recommendations
5. Returns the top 3 internship recommendations as JSON

### Key Files Modified

1. **[api_server.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/api_server.py)** - Added the `/ai_recommend` endpoint
2. **[frontend/ai.js](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.js)** - Updated to send form data to backend and display results
3. **[frontend/ai.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.html)** - Removed incorrect form action

## How It Works

### 1. Frontend Form Submission

When the user fills out the form in [ai.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.html) and clicks "Get AI Recommendations":

1. JavaScript in [ai.js](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.js) collects all form data
2. Form validation is performed
3. Data is sent as JSON to `http://localhost:5000/ai_recommend`

### 2. Backend Processing

The Flask server at `/ai_recommend`:

1. Receives the JSON form data
2. Maps form fields to user profile attributes:
   - Education field mapping
   - Skills parsing
   - Domain and location preferences
   - Enrollment status conversion
3. Creates a temporary user in the dataset
4. Calls the ML model to get recommendations
5. Returns results as JSON

### 3. Response Handling

The frontend receives the JSON response containing:

```json
{
  "user_profile": {
    "name": "John Doe",
    "education": "BCA",
    "skills": "Python, Machine Learning, SQL",
    "preferred_domain": "Data Science",
    "preferred_location": "Bangalore",
    // ... other fields
  },
  "recommendations": [
    {
      "company": "Company Name",
      "role": "Intern Role",
      "domain": "Data Science",
      "location": "Bangalore",
      "type": "Full-time",
      "duration": "6 months",
      "stipend": "20000 INR",
      "similarity_score": 0.85,
      "reason": "This internship matches your preferred domain in Data Science..."
    }
    // ... up to 3 recommendations
  ]
}
```

### 4. Display Results

The JavaScript displays the recommendations in a modal popup with:
- User profile summary
- Top 3 internship recommendations
- Similarity scores
- Detailed reasons for each recommendation

## API Endpoints

### POST `/ai_recommend`

**Request Body:**
```json
{
  "name": "string",
  "citizenship": "string",
  "age": "integer",
  "eduMin": "string",
  "skills": "string",
  "domain": "string",
  "location": "string",
  "duration": "string",
  "edu": "string",
  "income": "string",
  "aadhaarLink": "string",
  "govtJob": "string"
}
```

**Response:**
```json
{
  "user_profile": {
    // Mapped user profile data
  },
  "recommendations": [
    {
      "company": "string",
      "role": "string",
      "domain": "string",
      "location": "string",
      "type": "string",
      "duration": "string",
      "stipend": "string",
      "similarity_score": "float",
      "reason": "string"
    }
  ],
  "total_recommendations": "integer",
  "model_type": "ml-based"
}
```

## Testing the Integration

### 1. Start the Backend Server

```bash
python api_server.py
```

The server will start on `http://localhost:5000`

### 2. Open the Frontend

Open [frontend/ai.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.html) in a web browser

### 3. Fill Out the Form

Complete all required fields in the form and click "Get AI Recommendations"

### 4. View Results

The recommendations will appear in a modal popup with detailed information

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure the Flask server is running and CORS is enabled
2. **ML Model Not Loaded**: Ensure the ML model files are in the correct location
3. **Network Errors**: Check that the frontend can reach `http://localhost:5000`

### Server Logs

Check the terminal where `api_server.py` is running for error messages and debugging information.

## Future Improvements

1. **Enhanced User Profile Mapping**: More sophisticated mapping of form fields to user attributes
2. **Real-time Validation**: Validate form data before submission
3. **Caching**: Cache ML model results for better performance
4. **Error Handling**: More detailed error messages for users
5. **Analytics**: Track which recommendations are most effective