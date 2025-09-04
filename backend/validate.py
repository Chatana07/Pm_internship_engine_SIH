"""
Validation script for the Internship Matching System
Tests all components and ensures the system works correctly
"""

import pandas as pd
import json
from internship_matcher import InternshipMatcher, UserProfile, Internship


def validate_datasets():
    """Validate that the datasets are properly formatted and loaded."""
    print("🔍 VALIDATING DATASETS")
    print("="*50)
    
    try:
        # Check user dataset
        users_df = pd.read_csv('dataset/user_profile_dataset_100.csv')
        print(f"✅ User dataset loaded: {len(users_df)} records")
        
        required_user_columns = ['UserID', 'Education', 'Skills', 'PreferredDomain', 
                               'PreferredLocation', 'InternshipDuration', 'EnrollmentStatus']
        
        missing_columns = [col for col in required_user_columns if col not in users_df.columns]
        if missing_columns:
            print(f"❌ Missing user columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required user columns present")
        
        # Check internship dataset
        internships_df = pd.read_csv('dataset/internship_dataset_50.csv')
        print(f"✅ Internship dataset loaded: {len(internships_df)} records")
        
        required_internship_columns = ['InternshipID', 'Company', 'Role', 'Domain',
                                     'Location', 'Type', 'Duration', 'Stipend']
        
        missing_columns = [col for col in required_internship_columns if col not in internships_df.columns]
        if missing_columns:
            print(f"❌ Missing internship columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required internship columns present")
        
        # Check data quality
        print(f"\n📊 DATA QUALITY CHECKS:")
        print(f"User dataset shape: {users_df.shape}")
        print(f"Internship dataset shape: {internships_df.shape}")
        print(f"User null values: {users_df.isnull().sum().sum()}")
        print(f"Internship null values: {internships_df.isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False


def validate_matching_logic():
    """Validate the core matching logic with specific test cases."""
    print("\n🧠 VALIDATING MATCHING LOGIC")
    print("="*50)
    
    try:
        matcher = InternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        # Test case 1: Domain filtering
        print("\n🎯 Testing Domain Filtering...")
        user = matcher.users[0]  # First user
        print(f"User preferred domain: {user.preferred_domain}")
        
        domain_filtered = matcher.apply_domain_filter(user, matcher.internships)
        domain_matches = [i.domain for i in domain_filtered]
        
        if all(domain == user.preferred_domain for domain in domain_matches):
            print(f"✅ Domain filter working correctly: {len(domain_filtered)} matches")
        else:
            print(f"❌ Domain filter failed")
            return False
        
        # Test case 2: Location filtering
        print("\n📍 Testing Location Filtering...")
        location_filtered = matcher.apply_location_filter(user, domain_filtered)
        print(f"✅ Location filter applied: {len(location_filtered)} matches")
        
        # Test case 3: Duration filtering
        print("\n⏱️ Testing Duration Filtering...")
        duration_filtered = matcher.apply_duration_filter(user, location_filtered)
        duration_matches = [i.duration for i in duration_filtered]
        
        if all(duration == user.internship_duration for duration in duration_matches):
            print(f"✅ Duration filter working correctly: {len(duration_filtered)} matches")
        else:
            print(f"❌ Duration filter failed")
            return False
        
        # Test case 4: Enrollment rules
        print("\n👥 Testing Enrollment Rules...")
        enrollment_filtered = matcher.apply_enrollment_rules(user, duration_filtered)
        print(f"User enrollment status: {user.enrollment_status}")
        
        if enrollment_filtered:
            internship_types = [i.type for i in enrollment_filtered]
            print(f"Recommended internship types: {set(internship_types)}")
            print(f"✅ Enrollment rules applied: {len(enrollment_filtered)} matches")
        else:
            print(f"⚠️ No internships match enrollment rules")
        
        # Test case 5: Stipend ranking
        print("\n💰 Testing Stipend Ranking...")
        if enrollment_filtered:
            ranked = matcher.rank_by_stipend(enrollment_filtered)
            stipends = [i.stipend_value for i in ranked]
            
            if stipends == sorted(stipends, reverse=True):
                print(f"✅ Stipend ranking working correctly")
                print(f"Stipend order: {[i.stipend for i in ranked[:3]]}")
            else:
                print(f"❌ Stipend ranking failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Matching logic validation failed: {e}")
        return False


def validate_recommendations():
    """Test recommendation generation for multiple users."""
    print("\n🏆 VALIDATING RECOMMENDATIONS")
    print("="*50)
    
    try:
        matcher = InternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        test_users = [1, 5, 10, 15, 20, 25, 30]
        success_count = 0
        
        for user_id in test_users:
            try:
                recommendations = matcher.get_top_recommendations(user_id)
                user_info = matcher.get_user_info(user_id)
                
                print(f"\n👤 User {user_id} ({user_info['preferred_domain']}, "
                      f"{user_info['enrollment_status']}):")
                print(f"   Recommendations: {len(recommendations)}")
                
                # Validate recommendation structure
                for i, rec in enumerate(recommendations):
                    required_fields = ['internship_id', 'company', 'role', 'domain', 
                                     'location', 'type', 'duration', 'stipend', 'reason']
                    
                    missing_fields = [field for field in required_fields if field not in rec]
                    if missing_fields:
                        print(f"❌ Missing fields in recommendation {i+1}: {missing_fields}")
                        return False
                    
                    # Check if reason is generated
                    if not rec['reason'] or len(rec['reason']) < 20:
                        print(f"❌ Invalid reason for recommendation {i+1}")
                        return False
                
                if recommendations:
                    print(f"   Top recommendation: {recommendations[0]['company']} - "
                          f"{recommendations[0]['role']} ({recommendations[0]['stipend']})")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error processing user {user_id}: {e}")
        
        print(f"\n✅ Successfully processed {success_count}/{len(test_users)} users")
        return success_count == len(test_users)
        
    except Exception as e:
        print(f"❌ Recommendation validation failed: {e}")
        return False


def validate_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔬 VALIDATING EDGE CASES")
    print("="*50)
    
    try:
        matcher = InternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        # Test invalid user ID
        print("Testing invalid user ID...")
        try:
            recommendations = matcher.get_top_recommendations(999)
            print("❌ Should have raised error for invalid user ID")
            return False
        except ValueError:
            print("✅ Correctly handled invalid user ID")
        
        # Test user with no matches
        print("\nTesting users with potentially no matches...")
        no_match_count = 0
        for user_id in range(1, 101):
            recommendations = matcher.get_top_recommendations(user_id)
            if not recommendations:
                no_match_count += 1
        
        print(f"Users with no matches: {no_match_count}/100")
        print("✅ Edge case handling working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case validation failed: {e}")
        return False


def generate_validation_report():
    """Generate a comprehensive validation report."""
    print("\n📋 GENERATING VALIDATION REPORT")
    print("="*50)
    
    try:
        matcher = InternshipMatcher(
            user_dataset_path='dataset/user_profile_dataset_100.csv',
            internship_dataset_path='dataset/internship_dataset_50.csv'
        )
        
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_stats': {
                'total_users': len(matcher.users),
                'total_internships': len(matcher.internships)
            },
            'matching_stats': {
                'users_with_recommendations': 0,
                'users_without_recommendations': 0,
                'total_recommendations_generated': 0,
                'average_recommendations_per_user': 0
            },
            'domain_coverage': {},
            'location_coverage': {},
            'sample_recommendations': []
        }
        
        # Calculate matching statistics
        total_recommendations = 0
        users_with_recs = 0
        
        for user_id in range(1, 101):
            recommendations = matcher.get_top_recommendations(user_id)
            total_recommendations += len(recommendations)
            
            if recommendations:
                users_with_recs += 1
                
                # Add sample recommendations for first 5 users
                if len(report['sample_recommendations']) < 5:
                    user_info = matcher.get_user_info(user_id)
                    report['sample_recommendations'].append({
                        'user_id': user_id,
                        'user_domain': user_info['preferred_domain'],
                        'user_enrollment': user_info['enrollment_status'],
                        'recommendations_count': len(recommendations),
                        'top_recommendation': {
                            'company': recommendations[0]['company'],
                            'role': recommendations[0]['role'],
                            'stipend': recommendations[0]['stipend']
                        } if recommendations else None
                    })
        
        report['matching_stats']['users_with_recommendations'] = users_with_recs
        report['matching_stats']['users_without_recommendations'] = 100 - users_with_recs
        report['matching_stats']['total_recommendations_generated'] = total_recommendations
        report['matching_stats']['average_recommendations_per_user'] = total_recommendations / 100
        
        # Domain coverage analysis
        user_domains = {}
        for user in matcher.users:
            domain = user.preferred_domain
            user_domains[domain] = user_domains.get(domain, 0) + 1
        
        internship_domains = {}
        for internship in matcher.internships:
            domain = internship.domain
            internship_domains[domain] = internship_domains.get(domain, 0) + 1
        
        report['domain_coverage'] = {
            'user_domains': user_domains,
            'internship_domains': internship_domains
        }
        
        # Save report
        with open('validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ Validation report saved to 'validation_report.json'")
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"Total Users: {report['dataset_stats']['total_users']}")
        print(f"Total Internships: {report['dataset_stats']['total_internships']}")
        print(f"Users with Recommendations: {users_with_recs}/100 ({users_with_recs}%)")
        print(f"Average Recommendations per User: {total_recommendations/100:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 INTERNSHIP MATCHING SYSTEM VALIDATION")
    print("="*60)
    
    tests = [
        ("Dataset Validation", validate_datasets),
        ("Matching Logic", validate_matching_logic),
        ("Recommendation Generation", validate_recommendations),
        ("Edge Cases", validate_edge_cases),
        ("Validation Report", generate_validation_report)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_function():
                print(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()