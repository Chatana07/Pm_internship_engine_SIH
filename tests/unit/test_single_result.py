import urllib.request
import urllib.parse
import json

# Test data with very specific criteria that should match only one internship
test_data = {
    "name": "Test User",
    "citizenship": "Indian",
    "age": 22,
    "eduMin": "M.Tech AI",
    "skills": "Machine Learning, NLP, Python, TensorFlow",
    "domain": "AI",
    "location": "Remote",
    "duration": "3 months",
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
    response_data = json.loads(result)
    print(f"Total Recommendations: {response_data.get('total_recommendations', 0)}")
    print(f"Message: {response_data.get('message', '')}")
    
    # Print recommendations if any
    recommendations = response_data.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   Reason: {rec['reason']}")
            print()
    else:
        print("No recommendations found.")
        
except Exception as e:
    print(f"Error: {e}")