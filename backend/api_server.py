"""
Flask API Server for Internship Matching System
Provides REST endpoints for the ML-based internship matching service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from internship_matcher import InternshipMatcher
import json
import traceback
import sys
import os

# Add the ml_models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

# Import the ML-based matcher
try:
    from ml_internship_matcher import MLInternshipMatcher
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    ML_MODEL_AVAILABLE = False
    print(f"⚠️ ML model not available. Please check ml_models/ml_internship_matcher.py: {e}")

app = Flask(__name__)
# Enable CORS for all routes with specific configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the matchers globally
matcher = None
ml_matcher = None
ml_model_loaded = False


def initialize_matchers():
    """Initialize both rule-based and ML-based matchers with dataset paths."""
    global matcher, ml_matcher, ml_model_loaded
    try:
        # Get the root directory (parent of backend directory)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Construct correct paths to dataset files
        user_dataset_path = os.path.join(root_dir, 'dataset', 'user_profile_dataset_100.csv')
        internship_dataset_path = os.path.join(root_dir, 'dataset', 'internship_dataset_50.csv')
        
        # Initialize rule-based matcher
        matcher = InternshipMatcher(
            user_dataset_path=user_dataset_path,
            internship_dataset_path=internship_dataset_path
        )
        print("✅ Rule-based Internship Matcher initialized successfully")
        
        # Initialize ML-based matcher if available
        if ML_MODEL_AVAILABLE:
            try:
                ml_matcher = MLInternshipMatcher(
                    user_dataset_path=user_dataset_path,
                    internship_dataset_path=internship_dataset_path
                )
                # Try to load the pre-trained model
                try:
                    model_path = os.path.join(root_dir, 'ml_models', 'internship_matcher_model.joblib')
                    ml_matcher.load_model(model_path)
                    ml_model_loaded = True
                    print("✅ ML-based Internship Matcher initialized successfully")
                except Exception as e:
                    print(f"⚠️ Could not load pre-trained ML model: {e}")
                    print("🔄 Training new ML model...")
                    ml_matcher.train_model()
                    model_path = os.path.join(root_dir, 'ml_models', 'internship_matcher_model.joblib')
                    ml_matcher.save_model(model_path)
                    ml_model_loaded = True
                    print("✅ ML model trained and saved")
            except Exception as e:
                print(f"❌ Error initializing ML matcher: {e}")
                ml_model_loaded = False
        else:
            print("⚠️ ML model not available")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing matchers: {e}")
        return False


@app.route('/', methods=['GET'])
def home():
    """API documentation and status endpoint."""
    return jsonify({
        'service': 'Internship Matching System API',
        'version': '1.0.0',
        'status': 'active' if matcher else 'inactive',
        'ml_model_status': 'available' if ml_model_loaded else 'not available',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'GET /stats': 'System statistics',
            'POST /recommend': 'Get internship recommendations (rule-based)',
            'POST /ml_recommend': 'Get internship recommendations (ML-based)',
            'POST /ai_recommend': 'Get AI recommendations from frontend form',
            'GET /user/<user_id>': 'Get user information',
            'GET /users': 'List all users',
            'GET /internships': 'List all internships'
        },
        'description': 'ML-based system for matching students with internships based on preferences and enrollment rules'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if matcher:
        return jsonify({
            'status': 'healthy',
            'users_loaded': len(matcher.users),
            'internships_loaded': len(matcher.internships),
            'ml_model_loaded': ml_model_loaded
        })
    else:
        return jsonify({'status': 'unhealthy', 'error': 'Matcher not initialized'}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        # Domain distribution
        user_domains = {}
        for user in matcher.users:
            domain = user.preferred_domain
            user_domains[domain] = user_domains.get(domain, 0) + 1
        
        internship_domains = {}
        for internship in matcher.internships:
            domain = internship.domain
            internship_domains[domain] = internship_domains.get(domain, 0) + 1
        
        # Enrollment status distribution
        enrollment_stats = {}
        for user in matcher.users:
            status = user.enrollment_status
            enrollment_stats[status] = enrollment_stats.get(status, 0) + 1
        
        # Location distribution
        user_locations = {}
        for user in matcher.users:
            location = user.preferred_location
            user_locations[location] = user_locations.get(location, 0) + 1
        
        internship_locations = {}
        for internship in matcher.internships:
            location = internship.location
            internship_locations[location] = internship_locations.get(location, 0) + 1
        
        return jsonify({
            'total_users': len(matcher.users),
            'total_internships': len(matcher.internships),
            'user_domains': user_domains,
            'internship_domains': internship_domains,
            'enrollment_distribution': enrollment_stats,
            'user_locations': user_locations,
            'internship_locations': internship_locations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get internship recommendations for a user."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id is required in request body'}), 400
        
        user_id = data['user_id']
        top_k = data.get('top_k', 3)  # Default to top 3
        
        # Validate user_id
        if not isinstance(user_id, int) or user_id < 1 or user_id > 100:
            return jsonify({'error': 'user_id must be an integer between 1 and 100'}), 400
        
        # Get user info
        user_info = matcher.get_user_info(user_id)
        if not user_info:
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        # Get recommendations
        recommendations = matcher.get_top_recommendations(user_id, top_k)
        
        return jsonify({
            'user_id': user_id,
            'user_info': user_info,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'requested_count': top_k
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/ml_recommend', methods=['POST'])
def get_ml_recommendations():
    """Get ML-based internship recommendations for a user."""
    if not ml_model_loaded or not ml_matcher:
        return jsonify({'error': 'ML model not available or not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id is required in request body'}), 400
        
        user_id = data['user_id']
        top_k = data.get('top_k', 3)  # Default to top 3
        
        # Validate user_id
        if not isinstance(user_id, int) or user_id < 1 or user_id > 100:
            return jsonify({'error': 'user_id must be an integer between 1 and 100'}), 400
        
        # Get recommendations from ML model
        recommendations = ml_matcher.get_recommendations(user_id, top_k)
        
        # Get user info from rule-based matcher (same data)
        user_info = matcher.get_user_info(user_id) if matcher else {}
        
        return jsonify({
            'user_id': user_id,
            'user_info': user_info,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'requested_count': top_k,
            'model_type': 'ml-based'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/ai_recommend', methods=['POST'])
def get_ai_recommendations():
    """Get AI recommendations based on form data from frontend."""
    if not ml_model_loaded or not ml_matcher:
        return jsonify({'error': 'ML model not available or not initialized'}), 500
    
    try:
        # Get form data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided in request body'}), 400
        
        # Extract form fields and map to user profile
        # Note: This is a simplified mapping based on the form fields
        user_profile = {
            'name': data.get('name', ''),
            'citizenship': data.get('citizenship', 'Indian'),
            'age': data.get('age', 0),
            'education': data.get('eduMin', ''),
            'skills': data.get('skills', ''),
            'preferred_domain': data.get('domain', ''),
            'preferred_location': data.get('location', ''),
            'internship_duration': data.get('duration', '12 Months'),
            'enrollment_status': map_enrollment_status(data.get('edu', '')),
            'family_income': data.get('income', ''),
            'aadhaar_linked': data.get('aadhaarLink', 'no'),
            'govt_job_family': data.get('govtJob', 'no')
        }
        
        # Get recommendations directly from ML model without modifying dataset files
        recommendations = ml_matcher.get_recommendations_for_profile(user_profile, 3)
        
        # Handle case where no recommendations are found
        if not recommendations:
            return jsonify({
                'user_profile': user_profile,
                'recommendations': [],
                'total_recommendations': 0,
                'model_type': 'ml-based',
                'message': 'No internships found matching your criteria. Please try adjusting your preferences.'
            })
        
        return jsonify({
            'user_profile': user_profile,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'model_type': 'ml-based',
            'message': f'Found {len(recommendations)} internship{"s" if len(recommendations) != 1 else ""} matching your criteria.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def map_enrollment_status(form_value):
    """Map form enrollment status to system values."""
    mapping = {
        'Not in full-time': 'Remote/Online',
        'Enrolled full-time': 'Full-time',
        'Distance/Online OK': 'Remote/Online'
    }
    return mapping.get(form_value, 'Remote/Online')


def create_temp_user(user_profile):
    """Create a temporary user in the dataset."""
    try:
        import pandas as pd
        import os
        
        # Get the root directory (parent of backend directory)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load existing user dataset with correct path
        user_dataset_path = os.path.join(root_dir, 'dataset', 'user_profile_dataset_100.csv')
        user_df = pd.read_csv(user_dataset_path)
        
        # Get the next available user ID
        next_user_id = user_df['UserID'].max() + 1 if not user_df.empty else 1001
        
        # Create new user row
        new_user = {
            'UserID': next_user_id,
            'Education': user_profile['education'],
            'Skills': user_profile['skills'],
            'PreferredDomain': user_profile['preferred_domain'],
            'PreferredLocation': user_profile['preferred_location'],
            'InternshipDuration': user_profile['internship_duration'].replace(' Months', ' months'),
            'EnrollmentStatus': user_profile['enrollment_status']
        }
        
        # Add new user to dataframe
        new_user_df = pd.DataFrame([new_user])
        updated_df = pd.concat([user_df, new_user_df], ignore_index=True)
        
        # Save updated dataset with correct path
        temp_file = os.path.join(root_dir, 'dataset', 'temp_user_dataset.csv')
        updated_df.to_csv(temp_file, index=False)
        
        # Update ML matcher with new dataset
        ml_matcher.user_dataset_path = temp_file
        ml_matcher.load_datasets()
        
        return next_user_id
    except Exception as e:
        print(f"Error creating temporary user: {e}")
        traceback.print_exc()
        return None


def remove_temp_user(user_id):
    """Remove temporary user from dataset."""
    try:
        import pandas as pd
        import os
        
        # Get the root directory (parent of backend directory)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_file = os.path.join(root_dir, 'dataset', 'temp_user_dataset.csv')
        
        if os.path.exists(temp_file):
            # Load dataset
            user_df = pd.read_csv(temp_file)
            
            # Remove temporary user
            updated_df = user_df[user_df['UserID'] != user_id]
            
            # Save updated dataset
            updated_df.to_csv(temp_file, index=False)
            
            # Update ML matcher with new dataset
            ml_matcher.user_dataset_path = temp_file
            ml_matcher.load_datasets()
    except Exception as e:
        print(f"Error removing temporary user: {e}")
        traceback.print_exc()


@app.route('/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get information for a specific user."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        if user_id < 1 or user_id > 100:
            return jsonify({'error': 'user_id must be between 1 and 100'}), 400
        
        user_info = matcher.get_user_info(user_id)
        if not user_info:
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        return jsonify(user_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users', methods=['GET'])
def get_all_users():
    """Get list of all users with basic information."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        users_list = []
        for user in matcher.users:
            users_list.append({
                'user_id': user.user_id,
                'education': user.education,
                'preferred_domain': user.preferred_domain,
                'preferred_location': user.preferred_location,
                'enrollment_status': user.enrollment_status,
                'internship_duration': user.internship_duration
            })
        
        return jsonify({
            'total_users': len(users_list),
            'users': users_list
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/internships', methods=['GET'])
def get_all_internships():
    """Get list of all internships."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        internships_list = []
        for internship in matcher.internships:
            internships_list.append({
                'internship_id': internship.internship_id,
                'company': internship.company,
                'role': internship.role,
                'domain': internship.domain,
                'location': internship.location,
                'type': internship.type,
                'duration': internship.duration,
                'stipend': internship.stipend,
                'stipend_value': internship.stipend_value
            })
        
        return jsonify({
            'total_internships': len(internships_list),
            'internships': internships_list
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_recommend', methods=['POST'])
def batch_recommendations():
    """Get recommendations for multiple users at once."""
    if not matcher:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'user_ids' not in data:
            return jsonify({'error': 'user_ids array is required in request body'}), 400
        
        user_ids = data['user_ids']
        top_k = data.get('top_k', 3)
        
        if not isinstance(user_ids, list):
            return jsonify({'error': 'user_ids must be an array'}), 400
        
        results = []
        for user_id in user_ids:
            try:
                if not isinstance(user_id, int) or user_id < 1 or user_id > 100:
                    results.append({
                        'user_id': user_id,
                        'error': 'Invalid user_id'
                    })
                    continue
                
                user_info = matcher.get_user_info(user_id)
                if not user_info:
                    results.append({
                        'user_id': user_id,
                        'error': 'User not found'
                    })
                    continue
                
                recommendations = matcher.get_top_recommendations(user_id, top_k)
                results.append({
                    'user_id': user_id,
                    'user_info': user_info,
                    'recommendations': recommendations,
                    'total_recommendations': len(recommendations)
                })
                
            except Exception as e:
                results.append({
                    'user_id': user_id,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_results': results,
            'processed_count': len(results),
            'requested_count': len(user_ids)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Internship Matching API Server...")
    
    # Initialize the matchers
    if initialize_matchers():
        print("🌐 Server starting on http://localhost:5000")
        print("📖 API documentation available at http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize matchers. Server not started.")