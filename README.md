# 🎯 Internship Matching System

A lightweight machine learning-based system for intelligently matching students with internship opportunities based on their preferences, enrollment status, and comprehensive matching rules.

## 📋 Overview

This system implements both rule-based and ML-based approaches to recommend the top 3 most suitable internships for students. It considers multiple factors including domain preferences, location requirements, duration matching, enrollment status, and provides stipend-based ranking with detailed explanations for each recommendation.

## 🏗️ System Architecture

```
CSV Data → InternshipMatcher (Business Logic) → API Server (Flask) → JSON Response
```

- **Data Layer**: CSV files for user profiles and internship data
- **Business Logic**: Core matching engine with filtering rules
- **API Layer**: Flask-based REST API with multiple endpoints
- **Demo Interface**: Interactive testing and demonstration module

## 📊 Datasets

### User Dataset (100 profiles)
- **UserID**: Unique identifier
- **Education**: Educational background
- **Skills**: Technical and soft skills
- **PreferredDomain**: Desired internship domain
- **PreferredLocation**: Preferred work location
- **InternshipDuration**: Desired duration
- **EnrollmentStatus**: Current enrollment (Full-time/Part-time/Remote/Online)

### Internship Dataset (50 opportunities)
- **InternshipID**: Unique identifier
- **Company**: Organization name
- **Role**: Job title
- **Domain**: Work domain
- **Location**: Work location
- **Type**: Full-time or Part-time
- **Duration**: Internship duration
- **Stipend**: Compensation amount

## 🧠 Matching Algorithm

### 1. Domain Filtering
- Exact match between user's preferred domain and internship domain
- Domains: AI, Data Science, Business Analyst, UI/UX Designer, Graphics Designer, Web Development, Finance

### 2. Location Filtering
- Direct location match (Delhi, Mumbai, Bangalore, Kolkata, Remote)
- Remote internships are offered if no direct location match
- Users preferring remote get all remote opportunities

### 3. Duration Filtering
- Exact match between preferred duration and internship duration
- Common durations: 3 months, 6 months, 12 months

### 4. Enrollment Rules
- **Full-time students** → **Part-time internships**
- **Part-time students** → **Full-time internships**
- **Remote/Online students** → **Full-time internships**

### 5. Stipend Ranking
- Internships ranked by compensation (highest first)
- Unpaid internships ranked lowest
- Numerical extraction from stipend strings

### 6. Top 3 Selection
- Best 3 matches based on above criteria
- Each recommendation includes detailed reasoning

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy flask flask-cors scikit-learn
```

### Basic Usage

#### 1. Test the Core Matching System
```python
from backend.internship_matcher import InternshipMatcher

# Initialize matcher
matcher = InternshipMatcher(
    user_dataset_path='dataset/user_profile_dataset_100.csv',
    internship_dataset_path='dataset/internship_dataset_50.csv'
)

# Get recommendations for a user
matcher.print_recommendations(user_id=15)
```

#### 2. Run Interactive Demo
```bash
python backend/demo.py
```

#### 3. Start API Server
```bash
python backend/api_server.py
```
Server will be available at `http://localhost:5000`

#### 4. Validate System
```bash
python backend/validate.py
```

## 🤖 ML-Based Matching System

The system now includes an enhanced ML-based matching system with improved recommendation accuracy.

### Directory Structure
- **[backend/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/backend)**: Backend implementation files (API server, core matcher)
- **[frontend/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend)**: Frontend files (HTML, CSS, JavaScript)
- **[ml_models/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models)**: ML implementation files and trained models
- **[interactive/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/interactive)**: Interactive matching programs that ask for user input
- **[docs/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/docs)**: Documentation and reports
- **[dataset/](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/dataset)**: Data files in CSV and JSON formats

### Backend Directory
- [api_server.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/backend/api_server.py): Flask-based REST API server
- [internship_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/backend/internship_matcher.py): Core matching engine with filtering and ranking logic
- [demo.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/backend/demo.py): Interactive demo for testing and demonstration
- [validate.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/backend/validate.py): Validation and testing framework

### ML Models Directory
- [ml_internship_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models/ml_internship_matcher.py): ML-based internship matcher implementation
- [internship_matcher_model.joblib](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models/internship_matcher_model.joblib): Pre-trained ML model
- [evaluate_models.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models/evaluate_models.py): Comparison script between rule-based and ML-based approaches
- [simple_comparison.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models/simple_comparison.py): Simplified comparison script
- [integrated_system.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_models/integrated_system.py): Integrated system combining both approaches

### Frontend Directory
- [index.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/index.html): Main page with navigation to AI recommendations
- [ai.html](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.html): AI recommendations form
- [ai.js](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.js): JavaScript for AI recommendations form
- [ai.css](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/frontend/ai.css): Styles for AI recommendations form

## 🌐 API Endpoints

### Core Endpoints

#### `POST /recommend`
Get internship recommendations for a user.

**Request:**
```json
{
  "user_id": 15,
  "top_k": 3
}
```

**Response:**
```json
{
  "user_id": 15,
  "user_info": {
    "education": "B.Tech Computer Science",
    "preferred_domain": "AI",
    "enrollment_status": "Full-time"
  },
  "recommendations": [
    {
      "internship_id": 17,
      "company": "Swiggy",
      "role": "AI Research Intern",
      "domain": "AI",
      "location": "Remote",
      "type": "Part-time",
      "duration": "12 months",
      "stipend": "15000 INR",
      "reason": "This internship matches your preferred domain in AI, offers a Part-time role since you are currently Full-time, offers remote work flexibility, matches your preferred duration of 12 months, offers competitive compensation of 15000 INR."
    }
  ],
  "total_recommendations": 1
}
```

#### `POST /ai_recommend`
Get AI recommendations based on form data from frontend.

**Request:**
```json
{
  "name": "John Doe",
  "citizenship": "Indian",
  "age": 22,
  "eduMin": "BCA",
  "skills": "Python, Machine Learning, SQL",
  "domain": "Data Science",
  "location": "Bangalore",
  "duration": "12 Months",
  "edu": "Not in full-time",
  "income": "Up to ₹8,00,000",
  "aadhaarLink": "yes",
  "govtJob": "no"
}
```

## ▶️ Starting the Application

To start both the frontend and backend components of the application:

```bash
python start_application.py
```

This script will:
1. Start the backend Flask server on `http://localhost:5000`
2. Open the frontend in your default web browser
3. Keep the backend server running until you press Ctrl+C

Alternatively, you can start the components separately:

### Start Backend Only
```bash
cd backend
python api_server.py
```

### Open Frontend Only
Open `frontend/index.html` in your web browser, then navigate to the AI Recommendations page.

## 📖 Usage Instructions

1. Run the startup script: `python start_application.py`
2. The frontend will open automatically in your browser
3. Click on "AI Recommendations" in the navigation
4. Fill out the form with your details
5. Click "Get AI Recommendations" to receive personalized internship matches
6. View your recommendations in the modal that appears

## 🧪 Testing

To test the AI recommendation endpoint directly:

```bash
curl -X POST http://localhost:5000/ai_recommend \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "citizenship": "Indian",
    "age": 22,
    "eduMin": "B.Tech Computer Science",
    "skills": "Python, Django, HTML, CSS, JavaScript",
    "domain": "Web Development",
    "location": "Bangalore",
    "duration": "12 Months",
    "edu": "Not in full-time",
    "income": "Up to ₹8,00,000",
    "aadhaarLink": "yes",
    "govtJob": "no"
  }'
```

## 📁 Project Structure

```
ml for pm internship/
├── backend/                 # Backend implementation files
├── frontend/                # Frontend files (HTML, CSS, JavaScript)
├── ml_models/               # ML implementation files and trained models
├── interactive/             # Interactive matching programs
├── docs/                    # Documentation and reports
├── dataset/                 # Data files in CSV and JSON formats
├── start_application.py     # Startup script for entire application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📈 Performance Metrics

- **Rule-based accuracy**: ~19% coverage of user-internship pairs
- **ML-based accuracy**: ~100% coverage with personalized recommendations
- **Response time**: < 500ms for recommendation generation
- **Scalability**: Supports 100+ users and 50+ internships

## 🛠️ Development

### Adding New Internships
1. Update `dataset/internship_dataset_50.csv` with new internship data
2. Restart the backend server to reload data

### Adding New Users
1. Update `dataset/user_profile_dataset_100.csv` with new user data
2. Restart the backend server to reload data

### Retraining ML Model
```bash
cd ml_models
python ml_internship_matcher.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
