"""
Simple comparison script for rule-based vs ML-based internship matching
"""

import time
from internship_matcher import InternshipMatcher
from ml_internship_matcher import MLInternshipMatcher

def test_rule_based():
    """Test the rule-based system."""
    print("Testing Rule-Based System...")
    start_time = time.time()
    
    matcher = InternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Test with a few users
    test_users = [1, 5, 10, 15, 20, 25, 30]
    results = {}
    
    for user_id in test_users:
        try:
            recommendations = matcher.get_top_recommendations(user_id)
            results[user_id] = len(recommendations)
            print(f"User {user_id}: {len(recommendations)} recommendations")
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            results[user_id] = 0
    
    end_time = time.time()
    print(f"Rule-Based System completed in {end_time - start_time:.2f} seconds")
    
    return results

def test_ml_based():
    """Test the ML-based system."""
    print("\nTesting ML-Based System...")
    start_time = time.time()
    
    matcher = MLInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Train model
    matcher.train_model()
    
    # Test with same users
    test_users = [1, 5, 10, 15, 20, 25, 30]
    results = {}
    
    for user_id in test_users:
        try:
            recommendations = matcher.get_recommendations(user_id)
            results[user_id] = len(recommendations)
            print(f"User {user_id}: {len(recommendations)} recommendations")
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            results[user_id] = 0
    
    end_time = time.time()
    print(f"ML-Based System completed in {end_time - start_time:.2f} seconds")
    
    return results

def main():
    """Run simple comparison."""
    print("🚀 SIMPLE INTERNSHIP MATCHING SYSTEM COMPARISON")
    print("="*60)
    
    # Test both systems
    rule_based_results = test_rule_based()
    ml_based_results = test_ml_based()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"{'User ID':<10} {'Rule-Based':<15} {'ML-Based':<15} {'Difference':<15}")
    print("-"*60)
    
    total_rule = 0
    total_ml = 0
    
    for user_id in [1, 5, 10, 15, 20, 25, 30]:
        rule_count = rule_based_results.get(user_id, 0)
        ml_count = ml_based_results.get(user_id, 0)
        diff = ml_count - rule_count
        
        total_rule += rule_count
        total_ml += ml_count
        
        print(f"{user_id:<10} {rule_count:<15} {ml_count:<15} {diff:+d}")
    
    print("-"*60)
    print(f"{'TOTAL':<10} {total_rule:<15} {total_ml:<15} {total_ml - total_rule:+d}")
    
    avg_rule = total_rule / 7
    avg_ml = total_ml / 7
    
    print(f"{'AVERAGE':<10} {avg_rule:<15.2f} {avg_ml:<15.2f} {avg_ml - avg_rule:+.2f}")

if __name__ == "__main__":
    main()