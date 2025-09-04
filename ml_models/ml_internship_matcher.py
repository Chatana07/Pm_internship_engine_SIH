import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import re
import os
from typing import List, Dict, Tuple

class MLInternshipMatcher:
    """ML-based internship matching engine."""
    
    def __init__(self, user_dataset_path: str, internship_dataset_path: str):
        # Convert to absolute paths to avoid path issues
        self.user_dataset_path = os.path.abspath(user_dataset_path)
        self.internship_dataset_path = os.path.abspath(internship_dataset_path)
        self.users_df = None
        self.internships_df = None
        self.model = None
        self.vectorizers = {}
        self.load_datasets()
    
    def load_datasets(self):
        """Load both user and internship datasets from CSV files."""
        try:
            # Print debug information
            print(f"Loading user dataset from: {self.user_dataset_path}")
            print(f"Loading internship dataset from: {self.internship_dataset_path}")
            
            self.users_df = pd.read_csv(self.user_dataset_path)
            self.internships_df = pd.read_csv(self.internship_dataset_path)
            print(f"Loaded {len(self.users_df)} user profiles and {len(self.internships_df)} internships")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def _parse_stipend(self, stipend: str) -> float:
        """Parse stipend string to numerical value."""
        if pd.isna(stipend) or stipend.lower() == 'unpaid':
            return 0.0
        try:
            # Extract numbers from stipend string (e.g., "20000 INR" -> 20000)
            numbers = re.findall(r'\d+', str(stipend))
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 0.0
    
    def _preprocess_data(self):
        """Preprocess data for ML training."""
        # Add numerical stipend values
        self.internships_df['stipend_value'] = self.internships_df['Stipend'].apply(self._parse_stipend)
        
        # Create a combined dataset for training
        # For this example, we'll create a synthetic training dataset
        # In a real scenario, you would have historical data of user-internship interactions
        
        # Create features for users
        user_features = self.users_df.copy()
        
        # Create features for internships
        internship_features = self.internships_df.copy()
        
        return user_features, internship_features
    
    def _create_features(self, user_features, internship_features):
        """Create feature vectors for users and internships."""
        # For simplicity, we'll use a content-based filtering approach
        # In a real implementation, you would use more sophisticated feature engineering
        
        # Combine text features for vectorization
        user_texts = (
            user_features['Education'].fillna('') + ' ' +
            user_features['Skills'].fillna('') + ' ' +
            user_features['PreferredDomain'].fillna('') + ' ' +
            user_features['PreferredLocation'].fillna('')
        )
        
        internship_texts = (
            internship_features['Company'].fillna('') + ' ' +
            internship_features['Role'].fillna('') + ' ' +
            internship_features['Domain'].fillna('') + ' ' +
            internship_features['Location'].fillna('')
        )
        
        # Vectorize text features
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Fit on combined texts to ensure same feature space
        all_texts = pd.concat([user_texts, internship_texts])
        tfidf.fit(all_texts)
        
        # Transform user and internship texts
        user_vectors = tfidf.transform(user_texts)
        internship_vectors = tfidf.transform(internship_texts)
        
        self.vectorizers['tfidf'] = tfidf
        
        return user_vectors, internship_vectors
    
    def train_model(self):
        """Train the ML model for internship recommendations."""
        print("Preprocessing data...")
        user_features, internship_features = self._preprocess_data()
        
        print("Creating features...")
        user_vectors, internship_vectors = self._create_features(user_features, internship_features)
        
        # For this example, we'll create a simple similarity-based model
        # In a real implementation, you would have historical interaction data
        
        # Create a synthetic training dataset
        # We'll generate training examples based on the rule-based system logic
        training_data = []
        
        for user_idx, user_row in user_features.iterrows():
            user_id = user_row['UserID']
            
            # Find matching internships based on domain
            matching_internships = internship_features[
                internship_features['Domain'] == user_row['PreferredDomain']
            ]
            
            # For each matching internship, create a training example
            for _, internship_row in matching_internships.iterrows():
                # Calculate a synthetic score based on various factors
                score = self._calculate_synthetic_score(user_row, internship_row)
                training_data.append({
                    'user_id': user_id,
                    'internship_id': internship_row['InternshipID'],
                    'score': score
                })
        
        if not training_data:
            print("No training data generated. Using similarity-based approach.")
            self.model = {
                'user_vectors': user_vectors,
                'internship_vectors': internship_vectors,
                'user_features': user_features,
                'internship_features': internship_features
            }
            return
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        print(f"Generated {len(training_df)} training examples")
        
        # For this example, we'll use the similarity-based approach
        # A more sophisticated implementation would use the training data
        self.model = {
            'user_vectors': user_vectors,
            'internship_vectors': internship_vectors,
            'user_features': user_features,
            'internship_features': internship_features
        }
        
        print("Model training completed.")
    
    def _calculate_synthetic_score(self, user_row, internship_row):
        """Calculate a synthetic score for training data."""
        score = 0
        
        # Domain match (highest weight)
        if user_row['PreferredDomain'] == internship_row['Domain']:
            score += 10
        
        # Location match
        if user_row['PreferredLocation'] == internship_row['Location']:
            score += 5
        elif internship_row['Location'].lower() == 'remote':
            score += 3
        
        # Duration match
        if user_row['InternshipDuration'] == internship_row['Duration']:
            score += 5
        
        # Enrollment rules
        user_status = user_row['EnrollmentStatus'].lower()
        internship_type = internship_row['Type'].lower()
        
        if user_status == 'full-time' and internship_type == 'part-time':
            score += 3
        elif user_status == 'part-time' and internship_type == 'full-time':
            score += 3
        elif user_status == 'remote/online' and internship_type == 'full-time':
            score += 3
        
        # Stipend factor
        stipend_value = self._parse_stipend(internship_row['Stipend'])
        score += min(stipend_value / 1000, 10)  # Normalize stipend
        
        return score
    
    def get_recommendations(self, user_id: int, top_k: int = 3) -> List[Dict]:
        """Get ML-based recommendations for a user."""
        if not self.model:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        print(f"Looking for user with ID {user_id}")
        print(f"Available user IDs: {list(self.model['user_features']['UserID'])}")
        
        # Find user
        user_row = self.model['user_features'][self.model['user_features']['UserID'] == user_id]
        
        if user_row.empty:
            raise ValueError(f"User with ID {user_id} not found")
        
        user_idx = user_row.index[0]
        user_vector = self.model['user_vectors'][user_idx]
        
        # Calculate similarities with all internships
        similarities = cosine_similarity(user_vector, self.model['internship_vectors']).flatten()
        
        # Get top K internships
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            internship = self.model['internship_features'].iloc[idx]
            similarity_score = similarities[idx]
            
            recommendation = {
                'internship_id': int(internship['InternshipID']),
                'company': internship['Company'],
                'role': internship['Role'],
                'domain': internship['Domain'],
                'location': internship['Location'],
                'type': internship['Type'],
                'duration': internship['Duration'],
                'stipend': internship['Stipend'],
                'similarity_score': float(similarity_score),
                'reason': self._generate_reason(user_row.iloc[0], internship, similarity_score)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_recommendations_for_profile(self, user_profile: dict, top_k: int = 3) -> List[Dict]:
        """Get ML-based recommendations for a user profile without modifying datasets."""
        if not self.model:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get preferred domain and location from user profile
        preferred_domain = user_profile.get('preferred_domain', '')
        preferred_location = user_profile.get('preferred_location', '')
        
        # Create a temporary user row based on the profile
        temp_user_data = {
            'UserID': 99999,  # Temporary ID
            'Education': user_profile.get('education', ''),
            'Skills': user_profile.get('skills', ''),
            'PreferredDomain': preferred_domain,
            'PreferredLocation': preferred_location,
            'InternshipDuration': user_profile.get('internship_duration', '').replace(' Months', ' months'),
            'EnrollmentStatus': user_profile.get('enrollment_status', 'Remote/Online')
        }
        
        # First, get domain-matched internships (at least 1)
        domain_matched_internships = []
        if preferred_domain:
            domain_filtered = self.model['internship_features'][
                self.model['internship_features']['Domain'] == preferred_domain
            ]
            domain_matched_internships = domain_filtered.copy()
        
        # Check if we have domain-matched internships
        if preferred_domain and len(domain_matched_internships) == 0:
            # No internships in preferred domain, return empty recommendations
            return []
        
        # Create a temporary user row based on the profile
        temp_user_data = {
            'UserID': 99999,  # Temporary ID
            'Education': user_profile.get('education', ''),
            'Skills': user_profile.get('skills', ''),
            'PreferredDomain': preferred_domain,
            'PreferredLocation': preferred_location,
            'InternshipDuration': user_profile.get('internship_duration', '').replace(' Months', ' months'),
            'EnrollmentStatus': user_profile.get('enrollment_status', 'Remote/Online')
        }
        
        # Create a DataFrame with the temporary user
        temp_user_df = pd.DataFrame([temp_user_data])
        
        # Combine with existing user features for vectorization
        combined_users_df = pd.concat([self.model['user_features'], temp_user_df], ignore_index=True)
        
        # Re-vectorize user texts with the temporary user
        user_texts = (
            combined_users_df['Education'].fillna('') + ' ' +
            combined_users_df['Skills'].fillna('') + ' ' +
            combined_users_df['PreferredDomain'].fillna('') + ' ' +
            combined_users_df['PreferredLocation'].fillna('')
        )
        
        # Transform user texts using the existing vectorizer
        user_vectors = self.vectorizers['tfidf'].transform(user_texts)
        
        # Get the vector for our temporary user (last row)
        temp_user_vector = user_vectors[-1]
        
        # Get all internships for skills-based matching
        all_internships = self.model['internship_features'].copy()
        
        # Filter by location if specified and not remote
        if preferred_location and preferred_location.lower() != 'remote':
            # Check if there are internships in the preferred location
            location_filtered = all_internships[
                (all_internships['Location'] == preferred_location) |
                (all_internships['Location'].str.lower() == 'remote')
            ]
            
            # If no internships in preferred location or remote, return empty
            if len(location_filtered) == 0:
                return []
            
            all_internships = location_filtered
        
        # Re-vectorize internships for skills-based matching
        internship_texts = (
            all_internships['Company'].fillna('') + ' ' +
            all_internships['Role'].fillna('') + ' ' +
            all_internships['Domain'].fillna('') + ' ' +
            all_internships['Location'].fillna('')
        )
        filtered_internship_vectors = self.vectorizers['tfidf'].transform(internship_texts)
        
        # Calculate similarities with all internships
        similarities = cosine_similarity(temp_user_vector, filtered_internship_vectors).flatten()
        
        # Create a dataframe with similarities for sorting
        similarity_df = all_internships.copy()
        similarity_df['similarity_score'] = similarities
        
        # Sort by similarity score (descending)
        similarity_df = similarity_df.sort_values('similarity_score', ascending=False)
        
        # Get recommendations
        recommendations = []
        
        # First recommendation should be from preferred domain if specified
        if preferred_domain and len(domain_matched_internships) > 0:
            # Filter domain-matched internships by location
            if preferred_location and preferred_location.lower() != 'remote':
                domain_location_filtered = domain_matched_internships[
                    (domain_matched_internships['Location'] == preferred_location) |
                    (domain_matched_internships['Location'].str.lower() == 'remote')
                ]
                
                # If no internships in preferred location or remote, check if there are any domain matches
                if len(domain_location_filtered) > 0:
                    domain_matched_internships = domain_location_filtered
                elif len(domain_matched_internships) == 0:
                    # No internships available in preferred location or remote
                    return []
            elif preferred_location and preferred_location.lower() == 'remote':
                # If user prefers remote, filter for remote internships
                domain_location_filtered = domain_matched_internships[
                    domain_matched_internships['Location'].str.lower() == 'remote'
                ]
                if len(domain_location_filtered) > 0:
                    domain_matched_internships = domain_location_filtered
            
            # Re-vectorize domain-matched internships
            if len(domain_matched_internships) > 0:
                domain_texts = (
                    domain_matched_internships['Company'].fillna('') + ' ' +
                    domain_matched_internships['Role'].fillna('') + ' ' +
                    domain_matched_internships['Domain'].fillna('') + ' ' +
                    domain_matched_internships['Location'].fillna('')
                )
                domain_vectors = self.vectorizers['tfidf'].transform(domain_texts)
                
                # Calculate similarities with domain-matched internships
                domain_similarities = cosine_similarity(temp_user_vector, domain_vectors).flatten()
                
                # Add similarity scores to domain-matched internships
                domain_df = domain_matched_internships.copy()
                domain_df['similarity_score'] = domain_similarities
                
                # Sort by similarity score
                domain_df = domain_df.sort_values('similarity_score', ascending=False)
                
                # Add the best domain-matched internship as the first recommendation
                if len(domain_df) > 0:
                    best_domain_match = domain_df.iloc[0]
                    recommendation = {
                        'internship_id': int(best_domain_match['InternshipID']),
                        'company': best_domain_match['Company'],
                        'role': best_domain_match['Role'],
                        'domain': best_domain_match['Domain'],
                        'location': best_domain_match['Location'],
                        'type': best_domain_match['Type'],
                        'duration': best_domain_match['Duration'],
                        'stipend': best_domain_match['Stipend'],
                        'similarity_score': float(best_domain_match['similarity_score']),
                        'reason': self._generate_reason(temp_user_data, best_domain_match, best_domain_match['similarity_score'])
                    }
                    recommendations.append(recommendation)
        
        # Add skills-based recommendations for remaining slots
        remaining_slots = top_k - len(recommendations)
        if remaining_slots > 0:
            # Remove already recommended internships from similarity_df
            recommended_ids = [rec['internship_id'] for rec in recommendations]
            if recommended_ids:
                similarity_df = similarity_df[~similarity_df['InternshipID'].isin(recommended_ids)]
            
            # Get top remaining recommendations based on skills matching
            top_remaining = similarity_df.head(remaining_slots)
            
            for _, internship_row in top_remaining.iterrows():
                recommendation = {
                    'internship_id': int(internship_row['InternshipID']),
                    'company': internship_row['Company'],
                    'role': internship_row['Role'],
                    'domain': internship_row['Domain'],
                    'location': internship_row['Location'],
                    'type': internship_row['Type'],
                    'duration': internship_row['Duration'],
                    'stipend': internship_row['Stipend'],
                    'similarity_score': float(internship_row['similarity_score']),
                    'reason': self._generate_reason(temp_user_data, internship_row, internship_row['similarity_score'])
                }
                recommendations.append(recommendation)
        
        # If we still don't have enough recommendations and have domain preferences
        if len(recommendations) < top_k and preferred_domain and len(domain_matched_internships) > 0:
            # Fill remaining slots with other domain-matched internships
            remaining_slots = top_k - len(recommendations)
            recommended_ids = [rec['internship_id'] for rec in recommendations]
            
            # Get domain-matched internships not yet recommended
            available_domain_internships = domain_matched_internships[
                ~domain_matched_internships['InternshipID'].isin(recommended_ids)
            ]
            
            # Sort by InternshipID or other criteria to get consistent results
            available_domain_internships = available_domain_internships.sort_values('InternshipID')
            
            # Add remaining domain-matched internships
            for _, internship_row in available_domain_internships.head(remaining_slots).iterrows():
                # Re-calculate similarity for this internship
                internship_text = (
                    str(internship_row['Company']).fillna('') + ' ' +
                    str(internship_row['Role']).fillna('') + ' ' +
                    str(internship_row['Domain']).fillna('') + ' ' +
                    str(internship_row['Location']).fillna('')
                )
                internship_vector = self.vectorizers['tfidf'].transform([internship_text])
                similarity = cosine_similarity(temp_user_vector, internship_vector)[0][0]
                
                recommendation = {
                    'internship_id': int(internship_row['InternshipID']),
                    'company': internship_row['Company'],
                    'role': internship_row['Role'],
                    'domain': internship_row['Domain'],
                    'location': internship_row['Location'],
                    'type': internship_row['Type'],
                    'duration': internship_row['Duration'],
                    'stipend': internship_row['Stipend'],
                    'similarity_score': float(similarity),
                    'reason': self._generate_reason(temp_user_data, internship_row, similarity)
                }
                recommendations.append(recommendation)
                
                if len(recommendations) >= top_k:
                    break
        
        # Limit to top_k recommendations
        recommendations = recommendations[:top_k]
        
        return recommendations
    
    def _generate_reason(self, user_row, internship_row, similarity_score):
        """Generate explanation for recommendation."""
        reasons = []
        
        # Domain match
        if user_row['PreferredDomain'] == internship_row['Domain']:
            reasons.append(f"matches your preferred domain in {user_row['PreferredDomain']}")
        else:
            reasons.append(f"matches your skills and preferences")
        
        # Location match
        if user_row['PreferredLocation'] == internship_row['Location']:
            reasons.append(f"is available in your chosen location ({user_row['PreferredLocation']})")
        elif internship_row['Location'].lower() == 'remote':
            reasons.append("offers remote work flexibility")
        
        # Duration match
        if user_row['InternshipDuration'] == internship_row['Duration']:
            reasons.append(f"matches your preferred duration of {user_row['InternshipDuration']}")
        
        # Enrollment rule explanation
        user_status = user_row['EnrollmentStatus'].lower()
        internship_type = internship_row['Type'].lower()
        
        if user_status == 'full-time' and internship_type == 'part-time':
            reasons.append(f"offers a {internship_type} role since you are currently {user_row['EnrollmentStatus']}")
        elif user_status == 'part-time' and internship_type == 'full-time':
            reasons.append(f"offers a {internship_type} role since you are currently {user_row['EnrollmentStatus']}")
        elif user_status == 'remote/online' and internship_type == 'full-time':
            reasons.append(f"offers a {internship_type} role which is suitable for your {user_row['EnrollmentStatus']} status")
        
        # Stipend information
        stipend_value = self._parse_stipend(internship_row['Stipend'])
        if stipend_value > 0:
            reasons.append(f"offers competitive compensation of {internship_row['Stipend']}")
        
        if not reasons:
            reasons.append("has content similar to your profile")
        
        return f"This internship {', '.join(reasons)}. (Similarity score: {similarity_score:.2f})"
    
    def save_model(self, filepath: str):
        """Save the trained model using joblib."""
        if not self.model:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'vectorizers': self.vectorizers
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model using joblib."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizers = model_data['vectorizers']
        print(f"Model loaded from {filepath}")

def main():
    """Demo function to test the ML matching system."""
    # Initialize matcher
    matcher = MLInternshipMatcher(
        user_dataset_path='dataset/user_profile_dataset_100.csv',
        internship_dataset_path='dataset/internship_dataset_50.csv'
    )
    
    # Train model
    matcher.train_model()
    
    # Save model
    matcher.save_model('internship_matcher_model.joblib')
    
    # Test with a few users
    test_users = [1, 5, 10, 15, 20]
    
    for user_id in test_users:
        try:
            print(f"\n{'='*80}")
            print(f"ML-BASED INTERNSHIP RECOMMENDATIONS FOR USER {user_id}")
            print(f"{'='*80}")
            
            recommendations = matcher.get_recommendations(user_id)
            
            if not recommendations:
                print("No matching internships found based on your criteria.")
                continue
            
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
                
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

if __name__ == "__main__":
    main()