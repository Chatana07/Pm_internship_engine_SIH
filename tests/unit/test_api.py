import requests
import time

# Wait a moment for the server to start
time.sleep(5)

# Test data
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

try:
    # Test health endpoint
    health_response = requests.get('http://localhost:5000/health')
    print(f"Health Check - Status Code: {health_response.status_code}")
    print(f"Health Check - Response: {health_response.json()}")
    
    # Test AI recommendation endpoint
    ai_response = requests.post(
        'http://localhost:5000/ai_recommend',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nAI Recommendation - Status Code: {ai_response.status_code}")
    if ai_response.status_code == 200:
        print(f"AI Recommendation - Response: {ai_response.json()}")
    else:
        print(f"AI Recommendation - Error: {ai_response.text}")
        
except Exception as e:
    print(f"Error testing API: {e}")