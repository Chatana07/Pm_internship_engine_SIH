import requests
import json

# Test data similar to what the frontend would send
test_data = {
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
}

# Send request to the AI recommendation endpoint
try:
    response = requests.post('http://localhost:5000/ai_recommend', 
                           json=test_data,
                           headers={'Content-Type': 'application/json'})
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")