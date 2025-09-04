"""
Evaluation script to compare rule-based and ML-based internship matching approaches
"""

import pandas as pd
import json
from internship_matcher import InternshipMatcher
from ml_internship_matcher import MLInternshipMatcher

def evaluate_rule_based_system():
    """Evaluate the rule-based system."""
    print("Evaluating Rule-Based System...")
    
    matcher = InternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Test all users
    total_users = 100
    users_with_recommendations = 0
    total_recommendations = 0
    
    for user_id in range(1, total_users + 1):
        try:
            recommendations = matcher.get_top_recommendations(user_id)
            total_recommendations += len(recommendations)
            if recommendations:
                users_with_recommendations += 1
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
    
    accuracy = users_with_recommendations / total_users
    avg_recommendations = total_recommendations / total_users
    
    print(f"Rule-Based System Results:")
    print(f"  Users with recommendations: {users_with_recommendations}/{total_users} ({accuracy:.2%})")
    print(f"  Average recommendations per user: {avg_recommendations:.2f}")
    
    return {
        'accuracy': accuracy,
        'users_with_recommendations': users_with_recommendations,
        'total_recommendations': total_recommendations,
        'avg_recommendations': avg_recommendations
    }

def evaluate_ml_based_system():
    """Evaluate the ML-based system."""
    print("\nEvaluating ML-Based System...")
    
    matcher = MLInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Train model
    matcher.train_model()
    
    # Test all users
    total_users = 100
    users_with_recommendations = 0
    total_recommendations = 0
    
    for user_id in range(1, total_users + 1):
        try:
            recommendations = matcher.get_recommendations(user_id)
            total_recommendations += len(recommendations)
            if recommendations:
                users_with_recommendations += 1
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
    
    accuracy = users_with_recommendations / total_users
    avg_recommendations = total_recommendations / total_users
    
    print(f"ML-Based System Results:")
    print(f"  Users with recommendations: {users_with_recommendations}/{total_users} ({accuracy:.2%})")
    print(f"  Average recommendations per user: {avg_recommendations:.2f}")
    
    return {
        'accuracy': accuracy,
        'users_with_recommendations': users_with_recommendations,
        'total_recommendations': total_recommendations,
        'avg_recommendations': avg_recommendations
    }

def generate_comparison_report(rule_based_results, ml_based_results):
    """Generate a comparison report."""
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    
    print(f"{'Metric':<30} {'Rule-Based':<15} {'ML-Based':<15} {'Improvement':<15}")
    print("-"*60)
    
    # Accuracy comparison
    accuracy_improvement = ml_based_results['accuracy'] - rule_based_results['accuracy']
    print(f"{'Accuracy':<30} {rule_based_results['accuracy']:<15.2%} {ml_based_results['accuracy']:<15.2%} {accuracy_improvement:+.2%}")
    
    # Users with recommendations
    users_diff = ml_based_results['users_with_recommendations'] - rule_based_results['users_with_recommendations']
    print(f"{'Users with Recommendations':<30} {rule_based_results['users_with_recommendations']:<15} {ml_based_results['users_with_recommendations']:<15} {users_diff:+d}")
    
    # Average recommendations
    avg_diff = ml_based_results['avg_recommendations'] - rule_based_results['avg_recommendations']
    print(f"{'Avg Recommendations/User':<30} {rule_based_results['avg_recommendations']:<15.2f} {ml_based_results['avg_recommendations']:<15.2f} {avg_diff:+.2f}")
    
    print("-"*60)
    
    # Save results to file
    report = {
        'rule_based': rule_based_results,
        'ml_based': ml_based_results,
        'comparison': {
            'accuracy_improvement': accuracy_improvement,
            'users_improvement': users_diff,
            'avg_recommendations_improvement': avg_diff
        }
    }
    
    with open('model_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nReport saved to 'model_comparison_report.json'")
    
    return report

def main():
    """Run evaluation of both systems."""
    print("🚀 INTERNSHIP MATCHING SYSTEM EVALUATION")
    print("="*60)
    
    # Evaluate both systems
    rule_based_results = evaluate_rule_based_system()
    ml_based_results = evaluate_ml_based_system()
    
    # Generate comparison report
    report = generate_comparison_report(rule_based_results, ml_based_results)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()