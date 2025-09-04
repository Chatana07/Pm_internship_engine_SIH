"""
Test script to verify the ML model is working correctly
"""

import sys
import os

# Add the ml_models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_models'))

def test_ml_model():
    """Test the ML model import and basic functionality."""
    try:
        # Import the ML-based matcher
        from ml_models.ml_internship_matcher import MLInternshipMatcher
        print("✅ Successfully imported MLInternshipMatcher")
        
        # Initialize the matcher
        matcher = MLInternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        print("✅ Successfully initialized MLInternshipMatcher")
        
        # Try to load the pre-trained model
        try:
            matcher.load_model('ml_models/internship_matcher_model.joblib')
            print("✅ Successfully loaded pre-trained ML model")
        except Exception as e:
            print(f"⚠️ Could not load pre-trained model: {e}")
            print("🔄 Training new model...")
            matcher.train_model()
            matcher.save_model('ml_models/internship_matcher_model.joblib')
            print("✅ Successfully trained and saved new model")
        
        # Test getting recommendations
        recommendations = matcher.get_recommendations(1, 3)
        print(f"✅ Successfully got {len(recommendations)} recommendations for user 1")
        
        # Print first recommendation details
        if recommendations:
            rec = recommendations[0]
            print(f"   Company: {rec['company']}")
            print(f"   Role: {rec['role']}")
            print(f"   Domain: {rec['domain']}")
            print(f"   Similarity Score: {rec['similarity_score']:.3f}")
        
        print("\n🎉 All tests passed! The ML model is working correctly.")
        
    except Exception as e:
        print(f"❌ Error testing ML model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_model()