# ML-Based Internship Matching System

This directory contains the machine learning enhanced version of the internship matching system.

## Files

- **[ml_internship_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_internship_matcher.py)** - ML-based internship matcher implementation
- **[integrated_system.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/integrated_system.py)** - Integrated system combining rule-based and ML approaches
- **[internship_matcher_model.joblib](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/internship_matcher_model.joblib)** - Pre-trained ML model
- **[simple_comparison.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/simple_comparison.py)** - Comparison script
- **[ml_model_summary.md](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/ml_model_summary.md)** - Detailed summary of ML implementation

## Usage

### 1. Using the ML-Based Matcher Directly

```python
from ml_internship_matcher import MLInternshipMatcher

# Initialize and train
matcher = MLInternshipMatcher(
    user_dataset_path='dataset/user_profile_dataset_100.csv',
    internship_dataset_path='dataset/internship_dataset_50.csv'
)

# Train model (or load existing)
matcher.train_model()

# Get recommendations
recommendations = matcher.get_recommendations(user_id=1)
```

### 2. Using the Integrated System

```python
from integrated_system import IntegratedInternshipMatcher

# Initialize integrated system (loads pre-trained model)
integrated_matcher = IntegratedInternshipMatcher(
    user_dataset_path='dataset/user_profile_dataset_100.csv',
    internship_dataset_path='dataset/internship_dataset_50.csv',
    model_path='internship_matcher_model.joblib'
)

# Get recommendations using different approaches
rule_based = integrated_matcher.get_recommendations(1, approach='rule-based')
ml_based = integrated_matcher.get_recommendations(1, approach='ml-based')
hybrid = integrated_matcher.get_recommendations(1, approach='hybrid')  # Default
```

### 3. Running Comparison Scripts

```bash
# Compare rule-based vs ML-based approaches
python simple_comparison.py

# Full evaluation (may take longer)
python evaluate_models.py
```

## Key Benefits

1. **Improved Coverage**: ML-based approach provides recommendations for 100% of users vs 19% for rule-based
2. **Model Persistence**: Pre-trained models saved with joblib for fast loading
3. **Hybrid Approach**: Best of both worlds - strict matching when appropriate, flexible when needed
4. **Backward Compatibility**: Existing system functionality preserved

## Model Information

- **Algorithm**: Content-based filtering with TF-IDF and cosine similarity
- **Features**: Text features from user profiles and internship descriptions
- **Persistence**: joblib serialization for efficient model loading/saving
- **Training Time**: ~0.15 seconds for current dataset
- **Prediction Time**: ~0.01 seconds per user

## Requirements

Same as original system plus:
- scikit-learn >= 1.4.2
- joblib >= 1.4.2

Install with:
```bash
pip install -r requirements.txt
```