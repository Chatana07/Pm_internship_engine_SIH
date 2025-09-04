import urllib.request
import urllib.parse
import json

# Test data with AI as preferred domain
test_data = {
    "name": "Test User",
    "citizenship": "Indian",
    "age": 22,
    "eduMin": "B.Tech Computer Science",
    "skills": "Python, SQL",
    "domain": "AI",
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
    response_data = json.loads(result)
    print(f"Total Recommendations: {response_data.get('total_recommendations', 0)}")
    print(f"Message: {response_data.get('message', '')}")
    
    # Print recommendations if any
    recommendations = response_data.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        domains = set()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
            domains.add(rec['domain'])
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   Reason: {rec['reason']}")
            print()
        
        print(f"Domains in recommendations: {domains}")
        if len(domains) == 1 and list(domains)[0] == "AI":
            print("✅ Domain filtering is working correctly - all recommendations are from the AI domain")
        else:
            print("❌ Domain filtering issue - recommendations from other domains found")
    else:
        print("No recommendations found.")
        
except Exception as e:
    print(f"Error: {e}")