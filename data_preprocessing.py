# data_preprocessing.py - Data preprocessing utilities for PM Internship Recommendation System
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from typing import List, Dict, Union

# Configure logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Utility class for preprocessing internship and candidate data
    for the PM Internship Recommendation System
    """
    
    def __init__(self):
        """Initialize preprocessor with mappings and configurations"""
        
        # State name mappings for location standardization
        self.state_mappings = {
            'andhra pradesh': 'ap', 
            'arunachal pradesh': 'ar', 
            'assam': 'as',
            'bihar': 'br', 
            'chhattisgarh': 'ct', 
            'goa': 'ga', 
            'gujarat': 'gj',
            'haryana': 'hr', 
            'himachal pradesh': 'hp', 
            'jharkhand': 'jh',
            'karnataka': 'ka', 
            'kerala': 'kl', 
            'madhya pradesh': 'mp',
            'maharashtra': 'mh', 
            'manipur': 'mn', 
            'meghalaya': 'ml',
            'mizoram': 'mz', 
            'nagaland': 'nl', 
            'odisha': 'or', 
            'punjab': 'pb',
            'rajasthan': 'rj', 
            'sikkim': 'sk', 
            'tamil nadu': 'tn',
            'telangana': 'tg', 
            'tripura': 'tr', 
            'uttar pradesh': 'up',
            'uttarakhand': 'uk', 
            'west bengal': 'wb',
            'delhi': 'delhi',
            'jammu and kashmir': 'jk',
            'ladakh': 'ladakh',
            'chandigarh': 'chandigarh',
            'puducherry': 'puducherry'
        }
        
        # Major cities mapping
        self.major_cities = {
            'mumbai': 'mumbai',
            'delhi': 'delhi', 
            'new delhi': 'delhi',
            'bangalore': 'bangalore',
            'bengaluru': 'bangalore',
            'hyderabad': 'hyderabad',
            'chennai': 'chennai',
            'kolkata': 'kolkata',
            'pune': 'pune',
            'ahmedabad': 'ahmedabad',
            'jaipur': 'jaipur',
            'surat': 'surat',
            'lucknow': 'lucknow',
            'kanpur': 'kanpur',
            'nagpur': 'nagpur',
            'indore': 'indore',
            'thane': 'thane',
            'bhopal': 'bhopal',
            'visakhapatnam': 'visakhapatnam',
            'pimpri chinchwad': 'pune',
            'patna': 'patna',
            'vadodara': 'vadodara',
            'ghaziabad': 'ghaziabad',
            'ludhiana': 'ludhiana',
            'agra': 'agra',
            'nashik': 'nashik',
            'faridabad': 'faridabad',
            'meerut': 'meerut',
            'rajkot': 'rajkot',
            'kalyan dombivali': 'mumbai',
            'vasai virar': 'mumbai',
            'varanasi': 'varanasi',
            'srinagar': 'srinagar',
            'dhanbad': 'dhanbad',
            'jodhpur': 'jodhpur',
            'amritsar': 'amritsar',
            'raipur': 'raipur',
            'allahabad': 'allahabad',
            'coimbatore': 'coimbatore',
            'jabalpur': 'jabalpur',
            'gwalior': 'gwalior',
            'vijayawada': 'vijayawada',
            'madurai': 'madurai',
            'guwahati': 'guwahati',
            'chandigarh': 'chandigarh',
            'hubli dharwad': 'hubli',
            'mysore': 'mysore',
            'tiruchirappalli': 'tiruchirappalli'
        }
        
        # Remote work keywords
        self.remote_keywords = [
            'remote', 'work from home', 'wfh', 'online', 'virtual',
            'anywhere', 'location independent', 'telecommute'
        ]
        
        # Skill standardization mappings
        self.skill_mappings = {
            # Programming Languages
            'python': 'python',
            'java': 'java',
            'javascript': 'javascript',
            'js': 'javascript',
            'c++': 'cpp',
            'cpp': 'cpp',
            'c#': 'csharp',
            'csharp': 'csharp',
            'php': 'php',
            'ruby': 'ruby',
            'swift': 'swift',
            'kotlin': 'kotlin',
            'go': 'go',
            'rust': 'rust',
            'scala': 'scala',
            'perl': 'perl',
            'r': 'r',
            'matlab': 'matlab',
            
            # Web Technologies
            'html': 'html',
            'html5': 'html',
            'css': 'css',
            'css3': 'css',
            'react': 'react',
            'reactjs': 'react',
            'angular': 'angular',
            'angularjs': 'angular',
            'vue': 'vue',
            'vuejs': 'vue',
            'nodejs': 'nodejs',
            'node.js': 'nodejs',
            'express': 'express',
            'django': 'django',
            'flask': 'flask',
            'spring': 'spring',
            'bootstrap': 'bootstrap',
            'jquery': 'jquery',
            
            # Databases
            'sql': 'sql',
            'mysql': 'mysql',
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'mongodb': 'mongodb',
            'mongo': 'mongodb',
            'sqlite': 'sqlite',
            'oracle': 'oracle',
            'redis': 'redis',
            'cassandra': 'cassandra',
            
            # Data Science & ML
            'machine learning': 'machine_learning',
            'ml': 'machine_learning',
            'artificial intelligence': 'ai',
            'ai': 'ai',
            'data science': 'data_science',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'sklearn': 'sklearn',
            'tensorflow': 'tensorflow',
            'pytorch': 'pytorch',
            'keras': 'keras',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'tableau': 'tableau',
            'power bi': 'powerbi',
            'powerbi': 'powerbi',
            
            # Office & Business Tools
            'excel': 'excel',
            'microsoft excel': 'excel',
            'powerpoint': 'powerpoint',
            'word': 'word',
            'microsoft word': 'word',
            'google sheets': 'sheets',
            'google docs': 'docs',
            'outlook': 'outlook',
            
            # Soft Skills
            'communication': 'communication',
            'teamwork': 'teamwork',
            'leadership': 'leadership',
            'problem solving': 'problem_solving',
            'critical thinking': 'critical_thinking',
            'time management': 'time_management',
            'project management': 'project_management',
            'customer service': 'customer_service',
            'sales': 'sales',
            'marketing': 'marketing',
            'social media': 'social_media',
            'content writing': 'content_writing',
            'copywriting': 'copywriting',
            'seo': 'seo',
            'digital marketing': 'digital_marketing',
            'email marketing': 'email_marketing',
            'social media marketing': 'social_media_marketing',
            'content creation': 'content_creation',
            'graphic design': 'graphic_design',
            'ui/ux': 'ui_ux',
            'ui design': 'ui_design',
            'ux design': 'ux_design',
            
            # Domain Specific
            'accounting': 'accounting',
            'finance': 'finance',
            'financial analysis': 'financial_analysis',
            'financial modeling': 'financial_modeling',
            'investment': 'investment',
            'banking': 'banking',
            'healthcare': 'healthcare',
            'nursing': 'nursing',
            'pharmacy': 'pharmacy',
            'medical': 'medical',
            'agriculture': 'agriculture',
            'farming': 'farming',
            'research': 'research',
            'data analysis': 'data_analysis',
            'statistics': 'statistics',
            'mathematics': 'mathematics',
            'teaching': 'teaching',
            'education': 'education',
            'training': 'training',
            'hr': 'hr',
            'human resources': 'hr',
            'recruitment': 'recruitment',
            'operations': 'operations',
            'supply chain': 'supply_chain',
            'logistics': 'logistics',
            'manufacturing': 'manufacturing',
            'quality control': 'quality_control',
            'quality assurance': 'quality_assurance',
            'testing': 'testing',
            'debugging': 'debugging'
        }
    
    def clean_text(self, text: Union[str, None]) -> str:
        """
        Clean and normalize text data
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove special characters but keep spaces, alphanumeric, and some punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def standardize_location(self, location: Union[str, None]) -> str:
        """
        Standardize location names for consistent matching
        
        Args:
            location: Raw location string
            
        Returns:
            Standardized location string
        """
        if pd.isna(location) or location is None:
            return ''
        
        # Clean the location text
        location = self.clean_text(location)
        
        # Check for remote work keywords first
        for keyword in self.remote_keywords:
            if keyword in location:
                return 'remote'
        
        # Check for major cities
        for city, standard in self.major_cities.items():
            if city in location:
                return standard
        
        # Check for states
        for state, code in self.state_mappings.items():
            if state in location:
                return state
        
        # If no match found, return the cleaned location
        return location
    
    def extract_skills(self, skills_text: Union[str, None]) -> List[str]:
        """
        Extract and standardize skills from text
        
        Args:
            skills_text: Raw skills text (comma-separated or free text)
            
        Returns:
            List of standardized skill names
        """
        if pd.isna(skills_text) or skills_text is None:
            return []
        
        # Clean the text
        skills_text = self.clean_text(skills_text)
        
        # Split by common delimiters
        skills_list = re.split(r'[,;\|\n\t]+', skills_text)
        
        # Clean individual skills
        cleaned_skills = []
        for skill in skills_list:
            skill = skill.strip()
            if skill:  # Skip empty skills
                # Check if skill matches any in our mappings
                if skill in self.skill_mappings:
                    standardized_skill = self.skill_mappings[skill]
                    if standardized_skill not in cleaned_skills:
                        cleaned_skills.append(standardized_skill)
                else:
                    # Keep original skill if not in mappings
                    if skill not in cleaned_skills:
                        cleaned_skills.append(skill)
        
        return cleaned_skills
    
    def categorize_education(self, education: Union[str, None]) -> str:
        """
        Categorize education level into standard categories
        
        Args:
            education: Raw education text
            
        Returns:
            Standardized education category
        """
        if pd.isna(education) or education is None:
            return 'other'
        
        education = self.clean_text(education)
        
        # Education level patterns
        patterns = {
            'class_10': [
                '10th', 'tenth', 'ssc', 'class 10', 'class x',
                'secondary', 'high school', 'matriculation'
            ],
            'class_12': [
                '12th', 'twelfth', 'hsc', 'class 12', 'class xii',
                'intermediate', 'higher secondary', 'pre university'
            ],
            'diploma': [
                'diploma', 'polytechnic', 'poly', 'dip',
                'advanced diploma', 'post graduate diploma'
            ],
            'iti': [
                'iti', 'industrial training', 'trade certificate'
            ],
            'bachelors': [
                'ba', 'bsc', 'b.sc', 'bcom', 'b.com', 'bba', 'b.ba',
                'bca', 'b.ca', 'bachelor', 'graduation', 'graduate',
                'b.a', 'b.sc', 'b.com'
            ],
            'engineering': [
                'btech', 'b.tech', 'be', 'b.e', 'engineering',
                'bachelor of technology', 'bachelor of engineering'
            ],
            'masters': [
                'mba', 'mca', 'm.ca', 'msc', 'm.sc', 'ma', 'm.a',
                'mcom', 'm.com', 'master', 'post graduate', 'pg',
                'm.tech', 'mtech', 'me', 'm.e'
            ],
            'professional': [
                'ca', 'chartered accountant', 'cma', 'cs', 'company secretary',
                'mbbs', 'bds', 'md', 'ms', 'phd', 'doctorate'
            ]
        }
        
        # Check each category
        for category, keywords in patterns.items():
            if any(keyword in education for keyword in keywords):
                return category
        
        return 'other'
    
    def extract_duration_months(self, duration_text: Union[str, None]) -> int:
        """
        Extract duration in months from text
        
        Args:
            duration_text: Raw duration text
            
        Returns:
            Duration in months (default: 12)
        """
        if pd.isna(duration_text) or duration_text is None:
            return 12  # Default PM Internship duration
        
        duration_text = self.clean_text(duration_text)
        
        # Look for patterns like "12 months", "1 year", "6 month"
        if 'month' in duration_text:
            months = re.findall(r'(\d+)', duration_text)
            if months:
                return min(int(months[0]), 24)  # Cap at 24 months
        elif 'year' in duration_text:
            years = re.findall(r'(\d+)', duration_text)
            if years:
                return min(int(years[0]) * 12, 24)  # Cap at 24 months
        elif 'week' in duration_text:
            weeks = re.findall(r'(\d+)', duration_text)
            if weeks:
                return max(int(int(weeks[0]) / 4), 1)  # Convert weeks to months, min 1 month
        
        return 12  # Default
    
    def validate_internship_data(self, df: pd.DataFrame) -> bool:
        """
        Validate internship dataset for required columns and data quality
        
        Args:
            df: Internship DataFrame
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Required columns for the recommendation system
        required_columns = [
            'title', 'company', 'sector', 'location', 'description'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty values in critical columns
        for col in required_columns:
            empty_count = df[col].isna().sum()
            empty_percentage = (empty_count / len(df)) * 100
            
            if empty_percentage > 50:  # More than 50% empty
                logger.warning(f"Column '{col}' has {empty_percentage:.1f}% empty values")
            elif empty_count > 0:
                logger.info(f"Column '{col}' has {empty_count} empty values ({empty_percentage:.1f}%)")
        
        # Check data types and ranges
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Dataset validation passed: {len(df)} internships found")
        return True
    
    def preprocess_internships_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire internships dataset
        
        Args:
            df: Raw internship DataFrame
            
        Returns:
            Preprocessed DataFrame with additional columns
        """
        logger.info(f"Starting preprocessing of {len(df)} internships")
        
        # Validate data first
        self.validate_internship_data(df)
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text columns
        text_columns = ['title', 'description', 'skills_required', 'company']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[f'{col}_clean'] = processed_df[col].apply(self.clean_text)
        
        # Standardize locations
        if 'location' in processed_df.columns:
            processed_df['location_clean'] = processed_df['location'].apply(self.standardize_location)
        
        # Extract and standardize skills
        if 'skills_required' in processed_df.columns:
            processed_df['skills_list'] = processed_df['skills_required'].apply(self.extract_skills)
            processed_df['skills_count'] = processed_df['skills_list'].apply(len)
        
        # Add duration in months
        if 'duration' in processed_df.columns:
            processed_df['duration_months'] = processed_df['duration'].apply(self.extract_duration_months)
        else:
            processed_df['duration_months'] = 12  # Default PM Internship duration
        
        # Create combined text for ML processing
        text_cols_for_ml = []
        for col in ['title_clean', 'description_clean', 'sector', 'location_clean']:
            if col in processed_df.columns:
                text_cols_for_ml.append(col)
        
        if 'skills_list' in processed_df.columns:
            processed_df['skills_text'] = processed_df['skills_list'].apply(lambda x: ' '.join(x) if x else '')
            text_cols_for_ml.append('skills_text')
        
        # Combine all text for ML
        processed_df['combined_text'] = ''
        for col in text_cols_for_ml:
            processed_df['combined_text'] += ' ' + processed_df[col].fillna('').astype(str)
        
        # Clean the combined text
        processed_df['combined_text'] = processed_df['combined_text'].apply(self.clean_text)
        
        # Add metadata
        processed_df['processed_date'] = datetime.now().isoformat()
        processed_df['is_remote'] = processed_df['location_clean'] == 'remote'
        
        # Add sector standardization
        if 'sector' in processed_df.columns:
            processed_df['sector_clean'] = processed_df['sector'].apply(self.clean_text)
        
        logger.info(f"Preprocessing completed successfully. Added {len(processed_df.columns) - len(df.columns)} new columns")
        
        return processed_df
    
    def preprocess_candidate_profile(self, candidate_data: Dict) -> Dict:
        """
        Preprocess candidate profile data
        
        Args:
            candidate_data: Raw candidate profile dictionary
            
        Returns:
            Preprocessed candidate profile
        """
        processed_profile = candidate_data.copy()
        
        # Clean text fields
        text_fields = ['skills', 'interests', 'education']
        for field in text_fields:
            if field in processed_profile:
                processed_profile[f'{field}_clean'] = self.clean_text(processed_profile[field])
        
        # Standardize location preference
        if 'preferred_location' in processed_profile:
            processed_profile['preferred_location_clean'] = self.standardize_location(
                processed_profile['preferred_location']
            )
        
        # Extract skills list
        if 'skills' in processed_profile:
            processed_profile['skills_list'] = self.extract_skills(processed_profile['skills'])
        
        # Categorize education
        if 'education' in processed_profile:
            processed_profile['education_category'] = self.categorize_education(processed_profile['education'])
        
        # Create combined text for matching
        text_parts = []
        for field in ['skills_clean', 'interests_clean', 'preferred_sector', 'education_clean']:
            if field in processed_profile and processed_profile[field]:
                text_parts.append(str(processed_profile[field]))
        
        processed_profile['combined_text'] = ' '.join(text_parts)
        
        return processed_profile
    
    def get_preprocessing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """
        Get statistics about the preprocessing results
        
        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'original_rows': len(original_df),
            'processed_rows': len(processed_df),
            'original_columns': len(original_df.columns),
            'processed_columns': len(processed_df.columns),
            'new_columns': len(processed_df.columns) - len(original_df.columns),
            'processing_date': datetime.now().isoformat()
        }
        
        # Location statistics
        if 'location_clean' in processed_df.columns:
            stats['unique_locations'] = processed_df['location_clean'].nunique()
            stats['remote_internships'] = (processed_df['location_clean'] == 'remote').sum()
        
        # Skills statistics
        if 'skills_list' in processed_df.columns:
            all_skills = []
            for skills in processed_df['skills_list'].dropna():
                all_skills.extend(skills)
            stats['total_unique_skills'] = len(set(all_skills))
            stats['avg_skills_per_internship'] = len(all_skills) / len(processed_df) if len(processed_df) > 0 else 0
        
        # Sector statistics
        if 'sector' in processed_df.columns:
            stats['unique_sectors'] = processed_df['sector'].nunique()
            stats['sector_distribution'] = processed_df['sector'].value_counts().to_dict()
        
        return stats


# Utility functions for standalone use
def preprocess_csv_file(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Preprocess a CSV file of internships
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional)
        
    Returns:
        Processed DataFrame
    """
    # Load data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} records from {input_csv}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_internships_dataset(df)
    
    # Save if output path provided
    if output_csv:
        processed_df.to_csv(output_csv, index=False)
        logger.info(f"Saved processed data to {output_csv}")
    
    # Print stats
    stats = preprocessor.get_preprocessing_stats(df, processed_df)
    logger.info(f"Preprocessing completed: {stats}")
    
    return processed_df


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'title': [
            'Software Developer - Full Stack',
            'Marketing Intern (Digital)',
            'Data Analyst - Finance Sector!!!',
            'Healthcare Assistant @ Hospital'
        ],
        'company': [
            'TechCorp India Pvt. Ltd.',
            'Marketing Pro Solutions',
            'FinanceFirst Ltd',
            'City Hospital & Research Center'
        ],
        'sector': [
            'Technology',
            'Marketing',
            'Finance', 
            'Healthcare'
        ],
        'location': [
            'Bangalore, Karnataka',
            'Mumbai, Maharashtra',
            'New Delhi, NCR',
            'Chennai, Tamil Nadu'
        ],
        'description': [
            'Develop web applications using React and Node.js. Work with agile team.',
            'Create digital marketing campaigns for various clients and platforms.',
            'Analyze financial data and prepare reports for management team.',
            'Assist medical staff and help with patient care coordination.'
        ],
        'skills_required': [
            'React, Node.js, JavaScript, HTML, CSS, Git',
            'Social Media Marketing, Content Creation, Analytics, SEO',
            'Excel, SQL, Financial Modeling, Data Analysis',
            'Healthcare, Communication, Computer Skills, Empathy'
        ],
        'duration': [
            '12 months',
            '1 year', 
            '10 months',
            '12 months'
        ]
    })
    
    print("Original Data:")
    print(sample_data.head())
    print("\n" + "="*50 + "\n")
    
    # Preprocess the data
    processed_data = preprocessor.preprocess_internships_dataset(sample_data)
    
    print("Processed Data (selected columns):")
    columns_to_show = [
        'title', 'title_clean', 'location', 'location_clean', 
        'skills_required', 'skills_list', 'duration_months'
    ]
    print(processed_data[columns_to_show].head())
    
    print("\n" + "="*50 + "\n")
    
    # Get preprocessing statistics
    stats = preprocessor.get_preprocessing_stats(sample_data, processed_data)
    print("Preprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Test candidate preprocessing
    sample_candidate = {
        'age': 22,
        'education': 'B.Tech Computer Science and Engineering',
        'skills': 'Python, JavaScript, Web Development, Machine Learning',
        'interests': 'Software Development, AI, Data Science',
        'preferred_sector': 'Technology',
        'preferred_location': 'Bengaluru, Karnataka',
        'family_income': 500000,
        'is_employed': False,
        'govt_employee_family': False
    }
    
    print("Original Candidate Profile:")
    for key, value in sample_candidate.items():
        print(f"  {key}: {value}")
    
    processed_candidate = preprocessor.preprocess_candidate_profile(sample_candidate)
    
    print("\nProcessed Candidate Profile:")
    for key, value in processed_candidate.items():
        if key not in sample_candidate:  # Show only new fields
            print(f"  {key}: {value}")
    
    print("\nPreprocessing completed successfully!")