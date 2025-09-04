"""
Demo Interactive ML-based Internship Matcher
Shows how the system works with sample inputs
"""

import pandas as pd
from ml_internship_matcher import MLInternshipMatcher

def demo_user_input():
    """Demo user details for testing."""
    print("="*60)
    print("DEMO INTERACTIVE INTERNSHIP MATCHING SYSTEM")
    print("="*60)
    
    print("\nDemo user profile:")
    print("- Education: B.Tech Computer Science")
    print("- Skills: Python, Machine Learning, SQL")
    print("- Preferred Domain: AI")
    print("- Preferred Location: Bangalore")
    print("- Internship Duration: 6 months")
    print("- Enrollment Status: Full-time")
    
    # Create a user profile dictionary
    user_profile = {
        'UserID': 9999,  # Temporary ID for new user
        'Education': 'B.Tech Computer Science',
        'Skills': 'Python, Machine Learning, SQL',
        'PreferredDomain': 'AI',
        'PreferredLocation': 'Bangalore',
        'InternshipDuration': '6 months',
        'EnrollmentStatus': 'Full-time'
    }
    
    return user_profile

def create_temp_user_dataset(user_profile):
    """Create a temporary user dataset with the new user."""
    # Load existing user dataset
    existing_users = pd.read_csv('dataset/user_profile_dataset_100.csv')
    
    # Create a new row for the user
    new_user_df = pd.DataFrame([user_profile])
    
    # Combine with existing users
    combined_users = pd.concat([existing_users, new_user_df], ignore_index=True)
    
    # Save to temporary file
    temp_file = 'dataset/temp_user_dataset.csv'
    combined_users.to_csv(temp_file, index=False)
    
    return temp_file

def display_recommendations(recommendations, user_profile):
    """Display the recommendations in a formatted way."""
    print("\n" + "="*80)
    print("PERSONALIZED INTERNSHIP RECOMMENDATIONS")
    print("="*80)
    print(f"Education: {user_profile['Education']}")
    print(f"Skills: {user_profile['Skills']}")
    print(f"Preferred Domain: {user_profile['PreferredDomain']}")
    print(f"Preferred Location: {user_profile['PreferredLocation']}")
    print(f"Duration: {user_profile['InternshipDuration']}")
    print(f"Enrollment Status: {user_profile['EnrollmentStatus']}")
    print("="*80)
    
    if not recommendations:
        print("No matching internships found based on your criteria.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n🏆 RECOMMENDATION #{i}")
        print(f"Company: {rec['company']}")
        print(f"Role: {rec['role']}")
        print(f"Domain: {rec['domain']}")
        print(f"Location: {rec['location']}")
        print(f"Type: {rec['type']}")
        print(f"Duration: {rec['duration']}")
        print(f"Stipend: {rec['stipend']}")
        print(f"Similarity Score: {rec['similarity_score']:.2f}")
        print(f"💡 Why this internship: {rec['reason']}")
        print("-" * 60)

def main():
    """Main function to run the demo interactive matcher."""
    try:
        # Get demo user input
        user_profile = demo_user_input()
        
        # Create temporary user dataset
        temp_user_file = create_temp_user_dataset(user_profile)
        
        # Initialize ML matcher with temporary dataset
        matcher = MLInternshipMatcher(
            user_dataset_path=temp_user_file,
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        # Load the pre-trained model
        try:
            matcher.load_model('internship_matcher_model.joblib')
            print("\n✅ Pre-trained model loaded successfully")
        except Exception as e:
            print(f"\n⚠️ Failed to load pre-trained model: {e}")
            print("Training new model...")
            matcher.train_model()
            matcher.save_model('internship_matcher_model.joblib')
            print("✅ New model trained and saved")
        
        # Get recommendations
        print("\n🔍 Finding the best internships for you...")
        recommendations = matcher.get_recommendations(user_profile['UserID'])
        
        # Display recommendations
        display_recommendations(recommendations, user_profile)
        
        # Clean up temporary file
        try:
            import os
            os.remove(temp_user_file)
        except:
            pass
        
        print(f"\n{'='*80}")
        print("Thank you for using the Internship Matching System!")
        print("="*80)
        
        # Show how it works with different inputs
        print("\n" + "="*80)
        print("EXAMPLE WITH DIFFERENT USER PROFILE")
        print("="*80)
        
        # Different user profile
        user_profile2 = {
            'UserID': 9998,
            'Education': 'M.Tech AI',
            'Skills': 'SQL, Tableau, R, Business Strategy',
            'PreferredDomain': 'Business Analyst',
            'PreferredLocation': 'Delhi',
            'InternshipDuration': '12 months',
            'EnrollmentStatus': 'Remote/Online'
        }
        
        temp_user_file2 = create_temp_user_dataset(user_profile2)
        
        matcher2 = MLInternshipMatcher(
            user_dataset_path=temp_user_file2,
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        try:
            matcher2.load_model('internship_matcher_model.joblib')
        except:
            matcher2.train_model()
        
        recommendations2 = matcher2.get_recommendations(user_profile2['UserID'])
        display_recommendations(recommendations2, user_profile2)
        
        # Clean up
        try:
            os.remove(temp_user_file2)
        except:
            pass
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()