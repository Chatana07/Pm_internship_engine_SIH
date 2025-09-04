import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_models'))

from ml_internship_matcher import MLInternshipMatcher

def test_ml_model():
    """Test the ML model with various scenarios to verify skills-based matching and domain priority."""
    
    # Initialize the matcher
    matcher = MLInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Train the model
    print("Training ML model...")
    matcher.train_model()
    print("Model trained successfully!")
    
    print("\n=== Testing Domain Priority and Skills-Based Matching ===\n")
    
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
            
        print("\nAll recommendations:")
        for i, rec in enumerate(recommendations_1, 1):
            print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']}) - Score: {rec['similarity_score']:.3f}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: User with Web Development domain preference but AI skills
    print("Test 2: User with Web Development domain preference but AI skills")
    user_profile_2 = {
        'name': 'Test User 2',
        'education': 'B.Tech Computer Science',
        'skills': 'Python, SQL, Machine Learning',  # AI skills
        'preferred_domain': 'Web Development',      # But preferring Web Development
        'preferred_location': 'Bangalore',
        'internship_duration': '3 months',
        'enrollment_status': 'Remote/Online'
    }
    
    recommendations_2 = matcher.get_recommendations_for_profile(user_profile_2, 3)
    print(f"Total recommendations: {len(recommendations_2)}")
    
    if recommendations_2:
        first_rec = recommendations_2[0]
        print(f"First recommendation domain: {first_rec['domain']}")
        print(f"First recommendation should be Web Development: {first_rec['domain'] == 'Web Development'}")
        
        # Check that the first recommendation is from the preferred domain
        if first_rec['domain'] == 'Web Development':
            print("✅ First recommendation correctly from preferred domain (Web Development)")
        else:
            print(f"❌ First recommendation not from preferred domain. Got: {first_rec['domain']}")
            
        print("\nAll recommendations:")
        domains = []
        for i, rec in enumerate(recommendations_2, 1):
            print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']}) - Score: {rec['similarity_score']:.3f}")
            domains.append(rec['domain'])
            
        # Check if subsequent recommendations include other domains based on skills
        if len(domains) > 1:
            other_domains = domains[1:]
            print(f"Subsequent recommendation domains: {other_domains}")
            if any(domain == 'AI' for domain in other_domains):
                print("✅ Subsequent recommendations include AI domain based on skills")
            else:
                print("ℹ️  Subsequent recommendations may not include AI domain")
    
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
        for i, rec in enumerate(recommendations_3, 1):
            print(f"{i}. {rec['company']} - {rec['role']} ({rec['domain']})")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_ml_model()