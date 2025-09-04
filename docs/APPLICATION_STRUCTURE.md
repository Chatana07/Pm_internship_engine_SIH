# Application Structure

This document describes the organized structure of the Internship Matching System application.

## Directory Structure

```
ml for pm internship/
├── backend/
│   ├── api_server.py
│   ├── demo.py
│   ├── internship_matcher.py
│   └── validate.py
├── dataset/
│   ├── internship_dataset_50.csv
│   ├── internship_dataset_50.json
│   ├── temp_user_dataset.csv
│   └── user_profile_dataset_100.csv
├── docs/
│   ├── AI_RECOMMENDATIONS_INTEGRATION.md
│   ├── FINAL_SUMMARY.md
│   ├── FRONTEND_BACKEND_INTEGRATION_SUMMARY.md
│   ├── INTERACTIVE_MATCHER_README.md
│   ├── ML_README.md
│   ├── PROJECT_ORGANIZATION.md
│   ├── ml_model_summary.md
│   ├── model_comparison_report.json
│   └── validation_report.json
├── frontend/
│   ├── ai.css
│   ├── ai.html
│   ├── ai.js
│   ├── assests/
│   ├── index.html
│   ├── inship.css
│   ├── inship.html
│   ├── login.css
│   ├── login.js
│   ├── style.css
│   └── test.py
├── interactive/
│   ├── demo_interactive_matcher.py
│   ├── final_interactive_matcher.py
│   ├── interactive_ml_matcher.py
│   ├── json_based_matcher.py
│   └── test_interactive_matcher.py
├── ml_models/
│   ├── evaluate_models.py
│   ├── integrated_system.py
│   ├── internship_matcher_model.joblib
│   ├── ml_internship_matcher.py
│   └── simple_comparison.py
├── README.md
├── requirements.txt
├── start_application.py
└── SYSTEM_SUMMARY.md
```

## Directory Descriptions

### backend/
Contains all backend implementation files:
- **api_server.py**: Flask-based REST API server with new `/ai_recommend` endpoint
- **internship_matcher.py**: Original rule-based internship matcher
- **demo.py**: Interactive demo for testing and demonstration
- **validate.py**: Validation and testing framework

### dataset/
Contains all data files used by the system:
- **internship_dataset_50.csv**: Original internship data in CSV format
- **internship_dataset_50.json**: Internship data converted to JSON format for easier processing
- **user_profile_dataset_100.csv**: User profile data in CSV format
- **temp_user_dataset.csv**: Temporary user dataset (created during runtime)

### docs/
Contains documentation and reports:
- **AI_RECOMMENDATIONS_INTEGRATION.md**: Integration guide for AI recommendations
- **FINAL_SUMMARY.md**: Final project summary
- **FRONTEND_BACKEND_INTEGRATION_SUMMARY.md**: Frontend-backend integration summary
- **INTERACTIVE_MATCHER_README.md**: Documentation for interactive matcher programs
- **ML_README.md**: Documentation for ML-based matching system
- **PROJECT_ORGANIZATION.md**: Project structure documentation
- **ml_model_summary.md**: Detailed summary of ML model implementation
- **model_comparison_report.json**: Comparison between rule-based and ML-based approaches
- **validation_report.json**: System validation report

### frontend/
Contains all frontend files:
- **index.html**: Main page with navigation to AI recommendations
- **ai.html**: AI recommendations form
- **ai.js**: JavaScript for AI recommendations form
- **ai.css**: Styles for AI recommendations form
- **assests/**: Image and asset files
- **inship.html**: Internship information page
- **login.js**: Login functionality
- **style.css**: Main styles
- **login.css**: Login styles
- **inship.css**: Internship page styles

### interactive/
Contains interactive programs that ask for user input:
- **demo_interactive_matcher.py**: Demo version with sample user profiles
- **final_interactive_matcher.py**: Interactive program that asks for user input
- **interactive_ml_matcher.py**: Interactive ML matcher
- **json_based_matcher.py**: JSON-based matching implementation
- **test_interactive_matcher.py**: Test version with predefined user profiles

### ml_models/
Contains ML implementation files and trained models:
- **evaluate_models.py**: Comparison script between rule-based and ML-based approaches
- **integrated_system.py**: Integrated system combining both approaches
- **internship_matcher_model.joblib**: Pre-trained ML model saved with joblib
- **ml_internship_matcher.py**: ML-based internship matcher implementation
- **simple_comparison.py**: Simplified comparison script

## Usage Instructions

### Starting the Application

To start both the frontend and backend components:

```bash
python start_application.py
```

This script will:
1. Start the backend Flask server on `http://localhost:5000`
2. Open the frontend [index.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/index.html) in your default web browser
3. Display instructions for navigating to the AI recommendations

### Manual Startup

If you prefer to start components manually:

1. **Start Backend Server:**
   ```bash
   python backend/api_server.py
   ```

2. **Open Frontend:**
   Open [frontend/index.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/index.html) in a web browser

3. **Navigate to AI Recommendations:**
   Click on "AI Recommendations" in the quick links section

### Using the Application

1. From the main page ([index.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/index.html)), click on "AI Recommendations"
2. Fill out the form in [ai.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.html) with your information
3. Click "Get AI Recommendations"
4. View the top 3 internship recommendations in the modal that appears

## API Endpoints

### Backend API Server
Running on `http://localhost:5000`

Key endpoints:
- `POST /ai_recommend` - Get AI recommendations based on form data
- `POST /recommend` - Get recommendations for existing users
- `GET /health` - Check system health
- `GET /` - API documentation

## Data Flow

```
1. User fills form in ai.html
2. JavaScript in ai.js collects data
3. Data sent to Flask API /ai_recommend endpoint
4. API maps data to user profile
5. ML model generates recommendations
6. API returns JSON response
7. JavaScript displays results in modal
```

## Key Features

### Backend Features
- Flask REST API with CORS enabled
- Rule-based and ML-based matching algorithms
- Automatic ML model loading/training
- Temporary user management
- Comprehensive error handling

### Frontend Features
- Responsive form design
- Real-time form validation
- Loading state management
- Modal popup for results display
- User-friendly interface

### ML Model Features
- Content-based filtering with TF-IDF
- Cosine similarity matching
- Joblib model persistence
- Integration with Flask API
- High accuracy recommendations

## Testing Verification

The application structure has been verified with:
- Backend server startup
- Frontend page loading
- Form data submission
- ML model processing
- Results display

## Future Enhancements

1. **Enhanced Matching**: Incorporate more form fields into ML features
2. **Real-time Validation**: Validate form data before submission
3. **Caching**: Cache frequent requests for better performance
4. **Analytics**: Track recommendation effectiveness
5. **User Feedback**: Collect user ratings of recommendations