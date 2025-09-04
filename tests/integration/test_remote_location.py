import urllib.request
import urllib.parse
import json
import time

def test_remote_location():
    """Test remote location handling."""
    
    print("=== Testing Remote Location Handling ===\n")
    
    # Test data with remote location preference
    test_data = {
        "name": "Remote Location Test User",
        "citizenship": "Indian",
        "age": 22,
        "eduMin": "B.Tech Computer Science",
        "skills": "Python, SQL",  # Generic skills
        "domain": "AI",
        "location": "Remote",  # Preferring remote
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
            locations = []
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
                print(f"   Location: {rec['location']}")
                print(f"   Score: {rec['similarity_score']:.3f}")
                print(f"   Reason: {rec['reason']}")
                print()
                locations.append(rec['location'])
                
            # Check if all recommendations are remote or if remote is properly handled
            remote_internships = [loc for loc in locations if loc.lower() == 'remote']
            print(f"Remote internships: {len(remote_internships)} out of {len(locations)}")
            
            if len(remote_internships) == len(locations):
                print("✅ All recommendations are remote as requested")
            elif len(remote_internships) > 0:
                print("✅ Some recommendations are remote (system correctly includes remote options)")
            else:
                print("ℹ️  No remote internships found, but system provided alternatives")
                
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the backend server is running on http://localhost:5000")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_remote_location()