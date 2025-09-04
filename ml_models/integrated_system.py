"""
Integrated internship matching system that combines rule-based and ML-based approaches
"""

import pandas as pd
from internship_matcher import InternshipMatcher
from ml_internship_matcher import MLInternshipMatcher

class IntegratedInternshipMatcher:
    """Integrated system that uses both rule-based and ML-based approaches."""
    
    def __init__(self, user_dataset_path: str, internship_dataset_path: str, model_path: str = None):
        # Initialize both systems
        self.rule_based_matcher = InternshipMatcher(user_dataset_path, internship_dataset_path)
        self.ml_matcher = MLInternshipMatcher(user_dataset_path, internship_dataset_path)
        
        # Load or train ML model
        if model_path:
            try:
                self.ml_matcher.load_model(model_path)
                print("ML model loaded successfully")
            except Exception as e:
                print(f"Failed to load ML model: {e}. Training new model...")
                self._train_ml_model()
        else:
            self._train_ml_model()
    
    def _train_ml_model(self):
        """Train the ML model."""
        print("Training ML model...")
        self.ml_matcher.train_model()
        self.ml_matcher.save_model('internship_matcher_model.joblib')
        print("ML model trained and saved")
    
    def get_recommendations(self, user_id: int, top_k: int = 3, approach: str = 'hybrid') -> list:
        """
        Get internship recommendations using specified approach.
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations to return
            approach: 'rule-based', 'ml-based', or 'hybrid'
        """
        if approach == 'rule-based':
            return self.rule_based_matcher.get_top_recommendations(user_id, top_k)
        elif approach == 'ml-based':
            return self.ml_matcher.get_recommendations(user_id, top_k)
        elif approach == 'hybrid':
            # Try rule-based first, fall back to ML if no results
            try:
                rule_based_results = self.rule_based_matcher.get_top_recommendations(user_id, top_k)
                if rule_based_results:
                    return rule_based_results
                else:
                    return self.ml_matcher.get_recommendations(user_id, top_k)
            except Exception as e:
                print(f"Rule-based approach failed: {e}. Using ML-based approach.")
                return self.ml_matcher.get_recommendations(user_id, top_k)
        else:
            raise ValueError("Approach must be 'rule-based', 'ml-based', or 'hybrid'")
    
    def print_recommendations(self, user_id: int, approach: str = 'hybrid'):
        """Print formatted recommendations for a user."""
        # Get user info from rule-based matcher (same data)
        user_info = self.rule_based_matcher.get_user_info(user_id)
        if not user_info:
            print(f"User {user_id} not found!")
            return
        
        print(f"\n{'='*80}")
        print(f"INTERNSHIP RECOMMENDATIONS FOR USER {user_id} (Approach: {approach.upper()})")
        print(f"{'='*80}")
        print(f"Education: {user_info['education']}")
        print(f"Skills: {user_info['skills']}")
        print(f"Preferred Domain: {user_info['preferred_domain']}")
        print(f"Preferred Location: {user_info['preferred_location']}")
        print(f"Duration: {user_info['internship_duration']}")
        print(f"Enrollment Status: {user_info['enrollment_status']}")
        print(f"{'='*80}")
        
        recommendations = self.get_recommendations(user_id, approach=approach)
        
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
            
            # Print reason based on approach
            if approach == 'rule-based' or (approach == 'hybrid' and 'reason' in rec and 'Similarity score' not in rec['reason']):
                print(f"💡 Why this internship: {rec['reason']}")
            else:
                print(f"💡 Why this internship: {rec.get('reason', 'Recommended based on profile similarity')}")
                if 'similarity_score' in rec:
                    print(f"📈 Similarity Score: {rec['similarity_score']:.2f}")
            
            print("-" * 60)

def main():
    """Demo function to test the integrated system."""
    # Initialize integrated matcher
    integrated_matcher = IntegratedInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv',
        model_path='internship_matcher_model.joblib'
    )
    
    # Test with a few users using different approaches
    test_users = [1, 15, 25]
    approaches = ['rule-based', 'ml-based', 'hybrid']
    
    for user_id in test_users:
        for approach in approaches:
            try:
                integrated_matcher.print_recommendations(user_id, approach)
                print("\n" + "="*80 + "\n")
            except Exception as e:
                print(f"Error processing user {user_id} with {approach} approach: {e}")

if __name__ == "__main__":
    main()