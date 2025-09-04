import urllib.request
import urllib.parse
import json
import time

def test_complete_system():
    """Test the complete system from frontend to backend with ML model."""
    
    print("=== Testing Complete System Integration ===\n")
    
    # Test data with AI domain preference and SQL skills
    test_data = {
        "name": "Integration Test User",
        "citizenship": "Indian",
        "age": 22,
        "eduMin": "B.Tech Computer Science",
        "skills": "Python, SQL, Machine Learning",
        "domain": "AI",
        "location": "Bangalore",
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
        print("Sending request to backend API...")
        # Send request
        response = urllib.request.urlopen(req)
        # Read response
        result = response.read().decode('utf-8')
        print(f"Status Code: {response.getcode()}")
        
        response_data = json.loads(result)
        print(f"Total Recommendations: {response_data.get('total_recommendations', 0)}")
        print(f"Message: {response_data.get('message', '')}")
        print(f"Model Type: {response_data.get('model_type', '')}")
        
        # Print recommendations if any
        recommendations = response_data.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
                print(f"   Location: {rec['location']}")
                print(f"   Score: {rec['similarity_score']:.3f}")
                print(f"   Reason: {rec['reason']}")
                print()
                
            # Verify first recommendation is from preferred domain
            first_rec = recommendations[0]
            if first_rec['domain'] == 'AI':
                print("✅ First recommendation correctly from preferred domain (AI)")
            else:
                print(f"❌ First recommendation not from preferred domain. Got: {first_rec['domain']}")
                
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the backend server is running on http://localhost:5000")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_complete_system()