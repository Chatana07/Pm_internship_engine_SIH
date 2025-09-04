"""
Interactive ML-based Internship Matcher
Asks user for their details and suggests relevant internships
"""

import pandas as pd
import json
from ml_internship_matcher import MLInternshipMatcher

def get_user_input():
    """Get user details from input prompts."""
    print("="*60)
    print("INTERACTIVE INTERNSHIP MATCHING SYSTEM")
    print("="*60)
    
    print("\nPlease provide your details to get personalized internship recommendations:")
    
    # Get user input
    education = input("Education (e.g., B.Tech Computer Science, M.Tech AI): ").strip()
    skills = input("Skills (comma-separated, e.g., Python, SQL, Machine Learning): ").strip()
    preferred_domain = input("Preferred Domain (e.g., AI, Web Development, Data Science): ").strip()
    preferred_location = input("Preferred Location (e.g., Delhi, Remote, Bangalore): ").strip()
    internship_duration = input("Internship Duration (e.g., 3 months, 6 months, 12 months): ").strip()
    enrollment_status = input("Enrollment Status (e.g., Full-time, Part-time, Remote/Online): ").strip()
    
    # Create a user profile dictionary
    user_profile = {
        'UserID': 9999,  # Temporary ID for new user
        'Education': education,
        'Skills': skills,
        'PreferredDomain': preferred_domain,
        'PreferredLocation': preferred_location,
        'InternshipDuration': internship_duration,
        'EnrollmentStatus': enrollment_status
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
    """Main function to run the interactive matcher."""
    try:
        # Get user input
        user_profile = get_user_input()
        
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
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or contact support.")

if __name__ == "__main__":
    main()