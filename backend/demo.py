"""
Demo script for the Internship Matching System
Demonstrates the ML-based matching of students with internships
"""

from internship_matcher import InternshipMatcher
import json


def interactive_demo():
    """Interactive demo allowing users to test the matching system."""
    print("🎯 INTERNSHIP MATCHING SYSTEM DEMO")
    print("="*50)
    
    # Initialize matcher
    try:
        matcher = InternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    while True:
        print("\n🔍 CHOOSE AN OPTION:")
        print("1. Get recommendations for a specific user ID")
        print("2. Test with sample users")
        print("3. Show system statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            try:
                user_id = int(input("Enter User ID (1-100): "))
                if 1 <= user_id <= 100:
                    matcher.print_recommendations(user_id)
                else:
                    print("❌ Please enter a valid User ID between 1 and 100")
            except ValueError:
                print("❌ Please enter a valid number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '2':
            print("\n🧪 TESTING WITH SAMPLE USERS...")
            sample_users = [1, 15, 25, 35, 45]
            for user_id in sample_users:
                try:
                    matcher.print_recommendations(user_id)
                    input("\nPress Enter to continue to next user...")
                except Exception as e:
                    print(f"❌ Error processing user {user_id}: {e}")
        
        elif choice == '3':
            show_system_stats(matcher)
        
        elif choice == '4':
            print("👋 Thank you for using the Internship Matching System!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")


def show_system_stats(matcher):
    """Show system statistics and insights."""
    print("\n📊 SYSTEM STATISTICS")
    print("="*50)
    print(f"Total Users: {len(matcher.users)}")
    print(f"Total Internships: {len(matcher.internships)}")
    
    # Domain distribution
    user_domains = {}
    for user in matcher.users:
        domain = user.preferred_domain
        user_domains[domain] = user_domains.get(domain, 0) + 1
    
    internship_domains = {}
    for internship in matcher.internships:
        domain = internship.domain
        internship_domains[domain] = internship_domains.get(domain, 0) + 1
    
    print("\n🎯 DOMAIN DISTRIBUTION:")
    print("Users by preferred domain:")
    for domain, count in sorted(user_domains.items()):
        print(f"  {domain}: {count} users")
    
    print("\nInternships by domain:")
    for domain, count in sorted(internship_domains.items()):
        print(f"  {domain}: {count} internships")
    
    # Enrollment status distribution
    enrollment_stats = {}
    for user in matcher.users:
        status = user.enrollment_status
        enrollment_stats[status] = enrollment_stats.get(status, 0) + 1
    
    print(f"\n👥 ENROLLMENT STATUS DISTRIBUTION:")
    for status, count in sorted(enrollment_stats.items()):
        print(f"  {status}: {count} users")
    
    # Location distribution
    user_locations = {}
    for user in matcher.users:
        location = user.preferred_location
        user_locations[location] = user_locations.get(location, 0) + 1
    
    internship_locations = {}
    for internship in matcher.internships:
        location = internship.location
        internship_locations[location] = internship_locations.get(location, 0) + 1
    
    print(f"\n📍 LOCATION DISTRIBUTION:")
    print("Users by preferred location:")
    for location, count in sorted(user_locations.items()):
        print(f"  {location}: {count} users")
    
    print("\nInternships by location:")
    for location, count in sorted(internship_locations.items()):
        print(f"  {location}: {count} internships")


def test_matching_algorithm():
    """Test the core matching algorithm with detailed output."""
    print("\n🔬 ALGORITHM TESTING")
    print("="*50)
    
    matcher = InternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Test specific scenarios
    test_cases = [
        {"user_id": 1, "description": "M.Tech AI student seeking Business Analyst role"},
        {"user_id": 3, "description": "Part-time Data Science student"},
        {"user_id": 15, "description": "Full-time AI student from Kolkata"},
        {"user_id": 25, "description": "Remote Web Development student"},
        {"user_id": 40, "description": "Full-time Finance student preferring Remote"}
    ]
    
    for test_case in test_cases:
        print(f"\n🎯 TEST CASE: {test_case['description']}")
        print("-" * 60)
        
        try:
            user_info = matcher.get_user_info(test_case['user_id'])
            if user_info:
                print(f"User Details: {user_info['preferred_domain']} | "
                      f"{user_info['preferred_location']} | "
                      f"{user_info['enrollment_status']} | "
                      f"{user_info['internship_duration']}")
                
                recommendations = matcher.get_top_recommendations(test_case['user_id'])
                
                if recommendations:
                    print(f"✅ Found {len(recommendations)} recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec['company']} - {rec['role']} "
                              f"({rec['type']}, {rec['stipend']})")
                else:
                    print("❌ No recommendations found")
            else:
                print(f"❌ User {test_case['user_id']} not found")
                
        except Exception as e:
            print(f"❌ Error in test case: {e}")
        
        print()


def export_sample_results():
    """Export sample matching results to JSON for analysis."""
    print("\n📁 EXPORTING SAMPLE RESULTS")
    print("="*50)
    
    matcher = InternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Generate results for first 10 users
    results = []
    for user_id in range(1, 11):
        try:
            user_info = matcher.get_user_info(user_id)
            recommendations = matcher.get_top_recommendations(user_id)
            
            result = {
                'user_id': user_id,
                'user_info': user_info,
                'recommendations': recommendations,
                'recommendation_count': len(recommendations)
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
    
    # Save to JSON file
    try:
        with open('sample_matching_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✅ Results exported to 'sample_matching_results.json'")
    except Exception as e:
        print(f"❌ Error exporting results: {e}")


if __name__ == "__main__":
    print("🚀 INTERNSHIP MATCHING SYSTEM")
    print("Intelligent ML-based matching of students with internships")
    print("="*60)
    
    while True:
        print("\n🎯 MAIN MENU:")
        print("1. Interactive Demo")
        print("2. Algorithm Testing")
        print("3. Export Sample Results")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            interactive_demo()
        elif choice == '2':
            test_matching_algorithm()
        elif choice == '3':
            export_sample_results()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")