# Project Organization

This document describes the organized structure of the Internship Matching System project.

## Directory Structure

```
ml for pm internship/
├── dataset/
│   ├── internship_dataset_50.csv
│   ├── internship_dataset_50.json
│   └── user_profile_dataset_100.csv
├── docs/
│   ├── INTERACTIVE_MATCHER_README.md
│   ├── ML_README.md
│   ├── ml_model_summary.md
│   ├── model_comparison_report.json
│   └── validation_report.json
├── interactive/
│   ├── demo_interactive_matcher.py
│   ├── final_interactive_matcher.py
│   ├── interactive_ml_matcher.py
│   ├── json_based_matcher.py
│   └── test_interactive_matcher.py
├── ml_models/
│   ├── evaluate_models.py
│   ├── integrated_system.py
│   ├── internship_matcher_model.joblib
│   ├── ml_internship_matcher.py
│   └── simple_comparison.py
├── api_server.py
├── demo.py
├── internship_matcher.py
├── README.md
├── requirements.txt
├── SYSTEM_SUMMARY.md
└── validate.py
```

## Directory Descriptions

### dataset/
Contains all data files used by the system:
- **internship_dataset_50.csv**: Original internship data in CSV format
- **internship_dataset_50.json**: Internship data converted to JSON format for easier processing
- **user_profile_dataset_100.csv**: User profile data in CSV format

### docs/
Contains documentation and reports:
- **INTERACTIVE_MATCHER_README.md**: Documentation for interactive matcher programs
- **ML_README.md**: Documentation for ML-based matching system
- **ml_model_summary.md**: Detailed summary of ML model implementation
- **model_comparison_report.json**: Comparison between rule-based and ML-based approaches
- **validation_report.json**: System validation report

### interactive/
Contains interactive programs that ask for user input:
- **demo_interactive_matcher.py**: Demo version with sample user profiles
- **final_interactive_matcher.py**: Interactive program that asks for user input
- **interactive_ml_matcher.py**: Interactive ML matcher
- **json_based_matcher.py**: JSON-based matching implementation
- **test_interactive_matcher.py**: Test version with predefined user profiles

### ml_models/
Contains ML implementation files and trained models:
- **evaluate_models.py**: Comparison script between rule-based and ML-based approaches
- **integrated_system.py**: Integrated system combining both approaches
- **internship_matcher_model.joblib**: Pre-trained ML model saved with joblib
- **ml_internship_matcher.py**: ML-based internship matcher implementation
- **simple_comparison.py**: Simplified comparison script

### Root Directory Files
- **api_server.py**: Flask-based REST API server
- **demo.py**: Interactive demo for testing and demonstration
- **internship_matcher.py**: Original rule-based internship matcher
- **README.md**: Main project documentation
- **requirements.txt**: Python dependency list
- **SYSTEM_SUMMARY.md**: System architecture and functionality overview
- **validate.py**: Validation and testing framework

## Usage

### Rule-Based System
```bash
python demo.py              # Run interactive demo
python api_server.py        # Start API server
python validate.py          # Run validation tests
```

### ML-Based System
```bash
# Run ML-based interactive matcher
python interactive/final_interactive_matcher.py

# Run comparison between approaches
python ml_models/simple_comparison.py

# Evaluate models
python ml_models/evaluate_models.py
```

## Key Improvements

1. **Organized Structure**: Files are now grouped by functionality
2. **ML Integration**: Enhanced system with machine learning capabilities
3. **Interactive Features**: Programs that ask for user input and provide personalized recommendations
4. **Model Persistence**: Joblib pipeline for saving and loading trained models
5. **Comprehensive Documentation**: Detailed documentation for all components