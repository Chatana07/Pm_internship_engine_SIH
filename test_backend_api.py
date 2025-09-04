import urllib.request
import urllib.parse
import json
import time

def test_backend_api():
    """Test the backend API to verify it's using the updated ML model features."""
    
    print("=== Testing Backend API Integration ===\n")
    
    # Test data with AI domain preference and SQL skills
    test_data = {
        "name": "API Test User",
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
            if first_rec['domain'] == 'AI':
                print("✅ First recommendation correctly from preferred domain (AI)")
            else:
                print(f"❌ First recommendation not from preferred domain. Got: {first_rec['domain']}")
                
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the backend server is running on http://localhost:5000")
        
    print("\n" + "="*50 + "\n")
    
    # Test 2: User with Web Development domain preference but AI skills
    print("Test 2: User with Web Development domain preference but AI skills")
    test_data_2 = {
        "name": "API Test User 2",
        "citizenship": "Indian",
        "age": 22,
        "eduMin": "B.Tech Computer Science",
        "skills": "Python, SQL, Machine Learning",  # AI skills
        "domain": "Web Development",                # But preferring Web Development
        "location": "Bangalore",
        "duration": "3 months",
        "edu": "Not in full-time",
        "income": "Up to ₹8,00,000",
        "aadhaarLink": "yes",
        "govtJob": "no"
    }
    
    # Convert to JSON
    data_2 = json.dumps(test_data_2).encode('utf-8')
    
    # Create request
    req_2 = urllib.request.Request(
        'http://localhost:5000/ai_recommend',
        data=data_2,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        print("Sending request to backend API...")
        # Send request
        response = urllib.request.urlopen(req_2)
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
                print("ℹ️  Subsequent recommendations may include other domains based on skills")
                
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the backend server is running on http://localhost:5000")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_backend_api()