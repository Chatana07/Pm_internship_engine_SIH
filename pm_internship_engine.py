# PM Internship Recommendation Engine
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_preprocessing import DataPreprocessor

# Initialize
preprocessor = DataPreprocessor()

# Process your internship CSV
df = p
import pickle
import json
from datetime import datetime
import re
from typing import List, Dict, Tuple

class PMInternshipRecommendationEngine:
    def __init__(self):
        self.internships_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.sector_keywords = {
            'Technology': ['software', 'IT', 'computer', 'programming', 'coding', 'web', 'app', 'digital', 'tech'],
            'Finance': ['finance', 'banking', 'accounting', 'investment', 'economics', 'money', 'budget'],
            'Healthcare': ['health', 'medical', 'hospital', 'nursing', 'pharmacy', 'medicine', 'care'],
            'Education': ['education', 'teaching', 'school', 'learning', 'training', 'academic'],
            'Manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'mechanical', 'engineering'],
            'Agriculture': ['agriculture', 'farming', 'rural', 'crop', 'livestock', 'food'],
            'Marketing': ['marketing', 'sales', 'advertising', 'promotion', 'brand', 'customer'],
            'Government': ['government', 'public', 'administration', 'policy', 'bureaucracy', 'civil']
        }
    
    def load_internship_data(self, csv_path: str):
        """Load internship data from CSV file"""
        try:
            self.internships_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.internships_df)} internships from {csv_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_internships(self):
        """Preprocess internship data for ML model"""
        if self.internships_df is None:
            raise ValueError("No internship data loaded")
        
        # Create combined text features for content-based filtering
        text_columns = ['title', 'description', 'skills_required', 'sector', 'location']
        self.internships_df['combined_text'] = ''
        
        for col in text_columns:
            if col in self.internships_df.columns:
                self.internships_df['combined_text'] += ' ' + self.internships_df[col].fillna('').astype(str)
        
        # Clean text
        self.internships_df['combined_text'] = self.internships_df['combined_text'].str.lower()
        self.internships_df['combined_text'] = self.internships_df['combined_text'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.internships_df['combined_text'])
        
        # Encode categorical variables
        categorical_columns = ['sector', 'company_type', 'location']
        for col in categorical_columns:
            if col in self.internships_df.columns:
                le = LabelEncoder()
                self.internships_df[f'{col}_encoded'] = le.fit_transform(self.internships_df[col].fillna('unknown'))
                self.label_encoders[col] = le
        
        print("Preprocessing completed successfully")
    
    def check_eligibility(self, candidate_profile: Dict) -> Tuple[bool, List[str]]:
        """Check if candidate meets PM Internship Scheme eligibility criteria"""
        issues = []
        
        # Age check (21-24 years)
        age = candidate_profile.get('age', 0)
        if age < 21 or age > 24:
            issues.append("Age must be between 21-24 years")
        
        # Education check
        education = candidate_profile.get('education', '').lower()
        excluded_degrees = ['mba', 'phd', 'ca', 'cma', 'cs', 'mbbs', 'bds', 'master']
        if any(degree in education for degree in excluded_degrees):
            issues.append("Advanced degree holders are not eligible")
        
        # Family income check
        family_income = candidate_profile.get('family_income', 0)
        if family_income > 800000:  # 8 lakh
            issues.append("Family income exceeds ₹8 lakh limit")
        
        # Employment status
        if candidate_profile.get('is_employed', False):
            issues.append("Full-time employed candidates are not eligible")
        
        # Government employee family check
        if candidate_profile.get('govt_employee_family', False):
            issues.append("Family member is a government employee")
        
        return len(issues) == 0, issues
    
    def calculate_candidate_vector(self, candidate_profile: Dict) -> np.ndarray:
        """Convert candidate profile to feature vector for similarity calculation"""
        # Create candidate text from profile
        candidate_text = f"{candidate_profile.get('interests', '')} {candidate_profile.get('skills', '')} {candidate_profile.get('preferred_sector', '')} {candidate_profile.get('education', '')}"
        candidate_text = candidate_text.lower()
        
        # Transform using existing TF-IDF vectorizer
        candidate_tfidf = self.tfidf_vectorizer.transform([candidate_text])
        
        return candidate_tfidf
    
    def calculate_location_score(self, candidate_location: str, internship_location: str) -> float:
        """Calculate location preference score"""
        if not candidate_location or not internship_location:
            return 0.5
        
        candidate_location = candidate_location.lower()
        internship_location = internship_location.lower()
        
        # Exact match
        if candidate_location == internship_location:
            return 1.0
        
        # State match
        if candidate_location in internship_location or internship_location in candidate_location:
            return 0.8
        
        # Remote work bonus
        if 'remote' in internship_location or 'work from home' in internship_location:
            return 0.9
        
        return 0.3
    
    def get_recommendations(self, candidate_profile: Dict, top_k: int = 5) -> List[Dict]:
        """Get top-k internship recommendations for a candidate"""
        
        # Check eligibility first
        is_eligible, issues = self.check_eligibility(candidate_profile)
        if not is_eligible:
            return {
                'eligible': False,
                'issues': issues,
                'recommendations': []
            }
        
        # Calculate content similarity
        candidate_vector = self.calculate_candidate_vector(candidate_profile)
        content_similarities = cosine_similarity(candidate_vector, self.tfidf_matrix).flatten()
        
        # Calculate additional scores
        location_scores = []
        sector_match_scores = []
        
        candidate_location = candidate_profile.get('preferred_location', '')
        candidate_sectors = candidate_profile.get('preferred_sector', '').lower().split(',')
        
        for idx, row in self.internships_df.iterrows():
            # Location score
            loc_score = self.calculate_location_score(candidate_location, row.get('location', ''))
            location_scores.append(loc_score)
            
            # Sector match score
            sector_score = 0
            internship_sector = str(row.get('sector', '')).lower()
            for sector in candidate_sectors:
                if sector.strip() in internship_sector:
                    sector_score = 1.0
                    break
            sector_match_scores.append(sector_score)
        
        # Combine scores (weighted)
        content_weight = 0.4
        location_weight = 0.3
        sector_weight = 0.3
        
        final_scores = (
            content_weight * content_similarities +
            location_weight * np.array(location_scores) +
            sector_weight * np.array(sector_match_scores)
        )
        
        # Get top recommendations
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            internship = self.internships_df.iloc[idx]
            
            recommendation = {
                'internship_id': internship.get('id', idx),
                'title': internship.get('title', 'N/A'),
                'company': internship.get('company', 'N/A'),
                'sector': internship.get('sector', 'N/A'),
                'location': internship.get('location', 'N/A'),
                'description': internship.get('description', 'N/A')[:200] + '...',
                'skills_required': internship.get('skills_required', 'N/A'),
                'duration': internship.get('duration', '12 months'),
                'stipend': '₹5,000/month',
                'match_score': round(final_scores[idx] * 100, 1),
                'why_recommended': self.generate_recommendation_reason(candidate_profile, internship, final_scores[idx])
            }
            recommendations.append(recommendation)
        
        return {
            'eligible': True,
            'issues': [],
            'recommendations': recommendations
        }
    
    def generate_recommendation_reason(self, candidate_profile: Dict, internship: pd.Series, score: float) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = []
        
        # Sector match
        candidate_sectors = candidate_profile.get('preferred_sector', '').lower()
        if candidate_sectors and candidate_sectors in str(internship.get('sector', '')).lower():
            reasons.append("matches your preferred sector")
        
        # Location match
        candidate_location = candidate_profile.get('preferred_location', '').lower()
        internship_location = str(internship.get('location', '')).lower()
        if candidate_location and candidate_location in internship_location:
            reasons.append("located in your preferred area")
        
        # Skills match
        candidate_skills = candidate_profile.get('skills', '').lower()
        required_skills = str(internship.get('skills_required', '')).lower()
        skill_overlap = set(candidate_skills.split()) & set(required_skills.split())
        if skill_overlap:
            reasons.append("matches your skills")
        
        if not reasons:
            if score > 0.7:
                reasons.append("highly relevant to your profile")
            elif score > 0.5:
                reasons.append("good fit for your background")
            else:
                reasons.append("potential learning opportunity")
        
        return "This internship " + " and ".join(reasons[:2])
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoders': self.label_encoders,
            'sector_keywords': self.sector_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoders = model_data['label_encoders']
        self.sector_keywords = model_data['sector_keywords']
        
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the recommendation engine
    engine = PMInternshipRecommendationEngine()
    
    # Example internship data structure (replace with your actual CSV)
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'title': [
            'Software Development Intern',
            'Digital Marketing Intern', 
            'Financial Analyst Intern',
            'Healthcare Assistant Intern',
            'Agricultural Research Intern'
        ],
        'company': [
            'TechCorp India',
            'Marketing Solutions Ltd',
            'Finance Pro Services',
            'HealthCare Plus',
            'AgriTech Innovations'
        ],
        'sector': ['Technology', 'Marketing', 'Finance', 'Healthcare', 'Agriculture'],
        'location': ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Pune'],
        'description': [
            'Develop web applications using Python and JavaScript frameworks',
            'Create digital marketing campaigns and analyze social media metrics',
            'Assist in financial analysis and report preparation',
            'Support healthcare operations and patient care activities',
            'Research sustainable farming techniques and crop optimization'
        ],
        'skills_required': [
            'Python, JavaScript, HTML, CSS',
            'Social Media, Analytics, Content Creation',
            'Excel, Financial Modeling, Analysis',
            'Healthcare Knowledge, Communication',
            'Research, Data Analysis, Agriculture'
        ],
        'duration': ['12 months'] * 5
    }
    
    # Create sample DataFrame
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_internships.csv', index=False)
    
    # Load and process data
    engine.load_internship_data('sample_internships.csv')
    engine.preprocess_internships()
    
    # Example candidate profile
    candidate = {
        'age': 22,
        'education': 'B.Tech Computer Science',
        'skills': 'Python, web development, programming',
        'interests': 'software development, coding, technology',
        'preferred_sector': 'Technology, IT',
        'preferred_location': 'Bangalore',
        'family_income': 500000,
        'is_employed': False,
        'govt_employee_family': False
    }
    
    # Get recommendations
    recommendations = engine.get_recommendations(candidate, top_k=3)
    
    print("\n=== PM INTERNSHIP RECOMMENDATIONS ===")
    print(f"Eligibility Status: {'✓ Eligible' if recommendations['eligible'] else '✗ Not Eligible'}")
    
    if not recommendations['eligible']:
        print("Issues:")
        for issue in recommendations['issues']:
            print(f"  - {issue}")
    else:
        print(f"\nTop {len(recommendations['recommendations'])} Recommendations:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Company: {rec['company']}")
            print(f"   Sector: {rec['sector']}")
            print(f"   Location: {rec['location']}")
            print(f"   Match Score: {rec['match_score']}%")
            print(f"   Why: {rec['why_recommended']}")
    
    # Save the model
    engine.save_model('pm_internship_model.pkl')