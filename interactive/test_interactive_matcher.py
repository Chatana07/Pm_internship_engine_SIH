"""
Test Interactive Internship Matcher
Demonstrates functionality with predefined user profiles
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_internship_data():
    """Load internship data from JSON file."""
    with open('dataset/internship_dataset_50.json', 'r') as f:
        internships = json.load(f)
    return internships

def get_sample_user_profiles():
    """Get sample user profiles for testing."""
    user_profiles = [
        {
            'name': 'Computer Science Student',
            'education': 'B.Tech Computer Science',
            'skills': 'Python, Machine Learning, SQL',
            'preferred_domain': 'AI',
            'preferred_location': 'Bangalore',
            'internship_duration': '6 months',
            'enrollment_status': 'Full-time'
        },
        {
            'name': 'Business Analytics Student',
            'education': 'MBA Business Analytics',
            'skills': 'SQL, Tableau, R, Business Strategy',
            'preferred_domain': 'Business Analyst',
            'preferred_location': 'Delhi',
            'internship_duration': '12 months',
            'enrollment_status': 'Remote/Online'
        },
        {
            'name': 'Design Student',
            'education': 'B.Des Graphics',
            'skills': 'Adobe Creative Suite, UI/UX, Illustrator',
            'preferred_domain': 'UI/UX Designer',
            'preferred_location': 'Remote',
            'internship_duration': '3 months',
            'enrollment_status': 'Part-time'
        }
    ]
    return user_profiles

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
    print(f"PERSONALIZED INTERNSHIP RECOMMENDATIONS FOR {user_profile['name'].upper()}")
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
    print("Loading internship data from JSON...")
    internships = load_internship_data()
    print(f"Loaded {len(internships)} internships")
    
    # Get sample user profiles
    user_profiles = get_sample_user_profiles()
    
    # Process each user profile
    for user_profile in user_profiles:
        print(f"\n🔍 Analyzing profile for {user_profile['name']}...")
        recommendations = find_matches(user_profile, internships)
        display_recommendations(user_profile, recommendations)
    
    print(f"\n{'='*80}")
    print("DEMO COMPLETE - Thank you for using the Interactive Internship Matcher!")
    print("="*80)

if __name__ == "__main__":
    main()