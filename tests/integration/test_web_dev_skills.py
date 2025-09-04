import urllib.request
import urllib.parse
import json
import time

def test_web_dev_skills():
    """Test skills-based matching with Web Development as preferred domain."""
    
    print("=== Testing Web Development Skills Matching ===\n")
    
    # Test data with Web Development domain preference and AI-related skills
    test_data = {
        "name": "Web Dev Skills Test User",
        "citizenship": "Indian",
        "age": 22,
        "eduMin": "B.Tech Computer Science",
        "skills": "Python, Machine Learning, NLP, TensorFlow",  # AI skills
        "domain": "Web Development",  # Preferring Web Development
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
            domains = []
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
                print(f"   Location: {rec['location']}")
                print(f"   Score: {rec['similarity_score']:.3f}")
                print(f"   Reason: {rec['reason']}")
                print()
                domains.append(rec['domain'])
                
            # Verify first recommendation is from preferred domain
            first_rec = recommendations[0]
            if first_rec['domain'] == 'Web Development':
                print("✅ First recommendation correctly from preferred domain (Web Development)")
            else:
                print(f"❌ First recommendation not from preferred domain. Got: {first_rec['domain']}")
                
            # Check if subsequent recommendations include other domains based on skills
            if len(domains) > 1:
                other_domains = domains[1:]
                print(f"Subsequent recommendation domains: {other_domains}")
                if any(domain != 'Web Development' for domain in other_domains):
                    print("✅ Subsequent recommendations include other domains based on skills")
                else:
                    print("ℹ️  All recommendations from Web Development domain")
                
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the backend server is running on http://localhost:5000")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_web_dev_skills()