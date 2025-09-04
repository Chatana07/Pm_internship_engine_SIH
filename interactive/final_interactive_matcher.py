"""
Final Interactive Internship Matcher
Asks user for input and provides internship recommendations based on JSON data
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_internship_data():
    """Load internship data from JSON file."""
    with open('dataset/internship_dataset_50.json', 'r') as f:
        internships = json.load(f)
    return internships

def get_user_input():
    """Get user details from interactive input."""
    print("="*60)
    print("INTERACTIVE INTERNSHIP MATCHING SYSTEM")
    print("="*60)
    print("Please provide your details to get personalized recommendations:\n")
    
    # Get user input
    education = input("Education (e.g., B.Tech Computer Science, M.Tech AI): ").strip()
    skills = input("Skills (comma-separated, e.g., Python, SQL, Machine Learning): ").strip()
    preferred_domain = input("Preferred Domain (e.g., AI, Web Development, Data Science): ").strip()
    preferred_location = input("Preferred Location (e.g., Delhi, Remote, Bangalore): ").strip()
    internship_duration = input("Internship Duration (e.g., 3 months, 6 months, 12 months): ").strip()
    enrollment_status = input("Enrollment Status (e.g., Full-time, Part-time, Remote/Online): ").strip()
    
    # Create user profile
    user_profile = {
        'education': education,
        'skills': skills,
        'preferred_domain': preferred_domain,
        'preferred_location': preferred_location,
        'internship_duration': internship_duration,
        'enrollment_status': enrollment_status
    }
    
    return user_profile

def create_internship_features(internships):
    """Create feature strings for each internship."""
    features = []
    for internship in internships:
        feature_str = f"{internship['Domain']} {internship['Role']} {internship['Company']} {internship['Location']} {internship['Type']} {internship['Duration']}"
        features.append(feature_str)
    return features

def create_user_feature(user_profile):
    """Create feature string for user profile."""
    feature_str = f"{user_profile['preferred_domain']} {user_profile['skills']} {user_profile['education']} {user_profile['preferred_location']} {user_profile['internship_duration']} {user_profile['enrollment_status']}"
    return feature_str

def find_matches(user_profile, internships, top_k=3):
    """Find top K matching internships for user."""
    # Create features
    internship_features = create_internship_features(internships)
    user_feature = create_user_feature(user_profile)
    
    # Combine all features for vectorization
    all_features = internship_features + [user_feature]
    
    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(all_features)
    
    # Calculate similarity between user and all internships
    user_vector = feature_vectors[-1]  # Last one is user
    internship_vectors = feature_vectors[:-1]  # All except last one
    
    similarities = cosine_similarity(user_vector, internship_vectors).flatten()
    
    # Get top K matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Prepare recommendations
    recommendations = []
    for idx in top_indices:
        internship = internships[idx]
        recommendation = {
            'internship': internship,
            'similarity_score': similarities[idx]
        }
        recommendations.append(recommendation)
    
    return recommendations

def display_recommendations(user_profile, recommendations):
    """Display recommendations in a formatted way."""
    print("\n" + "="*80)
    print("PERSONALIZED INTERNSHIP RECOMMENDATIONS")
    print("="*80)
    print(f"Education: {user_profile['education']}")
    print(f"Skills: {user_profile['skills']}")
    print(f"Preferred Domain: {user_profile['preferred_domain']}")
    print(f"Preferred Location: {user_profile['preferred_location']}")
    print(f"Duration: {user_profile['internship_duration']}")
    print(f"Enrollment Status: {user_profile['enrollment_status']}")
    print("="*80)
    
    if not recommendations:
        print("No matching internships found based on your criteria.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        internship = rec['internship']
        print(f"\n🏆 RECOMMENDATION #{i}")
        print(f"Company: {internship['Company']}")
        print(f"Role: {internship['Role']}")
        print(f"Domain: {internship['Domain']}")
        print(f"Location: {internship['Location']}")
        print(f"Type: {internship['Type']}")
        print(f"Duration: {internship['Duration']}")
        print(f"Stipend: {internship['Stipend']}")
        print(f"Similarity Score: {rec['similarity_score']:.2f}")
        print("-" * 60)

def main():
    """Main function."""
    try:
        print("Loading internship data from JSON...")
        internships = load_internship_data()
        print(f"Loaded {len(internships)} internships")
        
        # Get user input
        user_profile = get_user_input()
        
        print("\n🔍 Analyzing your profile and finding matching internships...")
        recommendations = find_matches(user_profile, internships)
        
        # Display recommendations
        display_recommendations(user_profile, recommendations)
        
        print(f"\n{'='*80}")
        print("Thank you for using the Interactive Internship Matcher!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again.")

if __name__ == "__main__":
    main()