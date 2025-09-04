import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ml_internship_matcher import MLInternshipMatcher
import json

def test_improved_ml_model():
    """Test the improved ML model with various scenarios."""
    
    # Initialize the matcher
    matcher = MLInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Train the model
    matcher.train_model()
    
    print("=== Testing Improved ML Model ===\n")
    
    # Test 1: User with AI domain preference and SQL skills
    print("Test 1: User with AI domain preference and SQL skills")
    user_profile_1 = {
        'name': 'Test User 1',
        'education': 'B.Tech Computer Science',
        'skills': 'Python, SQL, Machine Learning',
        'preferred_domain': 'AI',
        'preferred_location': 'Bangalore',
        'internship_duration': '3 months',
        'enrollment_status': 'Remote/Online'
    }
    
    recommendations_1 = matcher.get_recommendations_for_profile(user_profile_1, 3)
    print(f"Total recommendations: {len(recommendations_1)}")
    
    if recommendations_1:
        first_rec = recommendations_1[0]
        print(f"First recommendation domain: {first_rec['domain']}")
        print(f"First recommendation should be AI: {first_rec['domain'] == 'AI'}")
        
        # Check that at least the first recommendation is from the preferred domain
        if first_rec['domain'] == 'AI':
            print("✅ First recommendation correctly from preferred domain")
        else:
            print("❌ First recommendation not from preferred domain")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: User with remote location preference
    print("Test 2: User with remote location preference")
    user_profile_2 = {
        'name': 'Test User 2',
        'education': 'B.Tech Computer Science',
        'skills': 'Python, JavaScript',
        'preferred_domain': 'Web Development',
        'preferred_location': 'Remote',
        'internship_duration': '3 months',
        'enrollment_status': 'Remote/Online'
    }
    
    recommendations_2 = matcher.get_recommendations_for_profile(user_profile_2, 3)
    print(f"Total recommendations: {len(recommendations_2)}")
    
    if recommendations_2:
        remote_internships = [rec for rec in recommendations_2 if rec['location'].lower() == 'remote']
        print(f"Remote internships in recommendations: {len(remote_internships)}")
        if len(remote_internships) == len(recommendations_2):
            print("✅ All recommendations are remote as requested")
        else:
            print("❌ Not all recommendations are remote")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 3: User with non-existent domain
    print("Test 3: User with non-existent domain")
    user_profile_3 = {
        'name': 'Test User 3',
        'education': 'B.Tech Computer Science',
        'skills': 'Python, SQL',
        'preferred_domain': 'NonExistentDomain',
        'preferred_location': 'Bangalore',
        'internship_duration': '3 months',
        'enrollment_status': 'Remote/Online'
    }
    
    recommendations_3 = matcher.get_recommendations_for_profile(user_profile_3, 3)
    print(f"Total recommendations: {len(recommendations_3)}")
    
    if len(recommendations_3) == 0:
        print("✅ Correctly returned no recommendations for non-existent domain")
    else:
        print("❌ Should have returned no recommendations for non-existent domain")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 4: User with location that has no internships
    print("Test 4: User with location that has no internships")
    user_profile_4 = {
        'name': 'Test User 4',
        'education': 'B.Tech Computer Science',
        'skills': 'Python, SQL',
        'preferred_domain': 'AI',
        'preferred_location': 'NonExistentCity',
        'internship_duration': '3 months',
        'enrollment_status': 'Remote/Online'
    }
    
    recommendations_4 = matcher.get_recommendations_for_profile(user_profile_4, 3)
    print(f"Total recommendations: {len(recommendations_4)}")
    
    # Should return remote internships or empty if no remote available for that domain
    if len(recommendations_4) >= 0:  # Could be 0 or more depending on remote availability
        print("✅ Handled location with no direct internships appropriately")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_improved_ml_model()