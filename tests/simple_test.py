import urllib.request
import urllib.parse
import json

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

# Convert to JSON
data = json.dumps(test_data).encode('utf-8')

# Create request
req = urllib.request.Request(
    'http://localhost:5000/ai_recommend',
    data=data,
    headers={'Content-Type': 'application/json'}
)

try:
    # Send request
    response = urllib.request.urlopen(req)
    # Read response
    result = response.read().decode('utf-8')
    print(f"Status Code: {response.getcode()}")
    print(f"Response: {result}")
except Exception as e:
    print(f"Error: {e}")