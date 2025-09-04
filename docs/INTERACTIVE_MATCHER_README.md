# Interactive Internship Matcher

This directory contains interactive Python programs that ask for user details and suggest relevant internships based on those inputs.

## Files

- **[final_interactive_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/final_interactive_matcher.py)** - Interactive program that asks for user input
- **[test_interactive_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/test_interactive_matcher.py)** - Demo version with sample user profiles
- **[json_based_matcher.py](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/json_based_matcher.py)** - JSON-based matching implementation
- **[dataset/internship_dataset_50.json](file:///c%3A/Users/MADHURIMA/Desktop/ml%20for%20pm%20internship/dataset/internship_dataset_50.json)** - Internship data in JSON format

## How It Works

1. **Data Conversion**: The CSV internship dataset is converted to JSON format for easier processing
2. **User Input**: The program asks for your education, skills, preferred domain, location, duration, and enrollment status
3. **Feature Extraction**: Both your profile and internship data are converted to feature vectors using TF-IDF
4. **Matching**: Cosine similarity is used to find the top 3 matching internships
5. **Recommendations**: Results are displayed with similarity scores and details

## Usage

### Interactive Version
```bash
python final_interactive_matcher.py
```
This will prompt you to enter your details and then provide personalized internship recommendations.

### Demo Version
```bash
python test_interactive_matcher.py
```
This runs with predefined user profiles to show how the system works.

### JSON-Based Version
```bash
python json_based_matcher.py
```
This demonstrates the JSON-based approach with a sample user profile.

## Example User Input

When running the interactive version, you might enter:
- Education: `B.Tech Computer Science`
- Skills: `Python, Machine Learning, SQL`
- Preferred Domain: `AI`
- Preferred Location: `Bangalore`
- Internship Duration: `6 months`
- Enrollment Status: `Full-time`

## Sample Output

```
PERSONALIZED INTERNSHIP RECOMMENDATIONS
================================================================================
Education: B.Tech Computer Science
Skills: Python, Machine Learning, SQL
Preferred Domain: AI
Preferred Location: Bangalore
Duration: 6 months
Enrollment Status: Full-time
================================================================================

🏆 RECOMMENDATION #1
Company: PhonePe
Role: Computer Vision Intern
Domain: AI
Location: Remote
Type: Part-time
Duration: 3 months
Stipend: 5000 INR
Similarity Score: 0.26
------------------------------------------------------------

🏆 RECOMMENDATION #2
Company: Flipkart
Role: Computer Vision Intern
Domain: AI
Location: Mumbai
Type: Full-time
Duration: 6 months
Stipend: Unpaid
Similarity Score: 0.25
------------------------------------------------------------

🏆 RECOMMENDATION #3
Company: Swiggy
Role: AI Research Intern
Domain: AI
Location: Bangalore
Type: Part-time
Duration: 12 months
Stipend: 15000 INR
Similarity Score: 0.23
------------------------------------------------------------
```

## Technical Details

- **Data Format**: JSON (converted from CSV)
- **Matching Algorithm**: TF-IDF vectorization + Cosine similarity
- **Libraries Used**: scikit-learn, pandas, json
- **Processing Time**: < 1 second

## Requirements

- Python 3.6+
- scikit-learn
- pandas

Install requirements with:
```bash
pip install -r requirements.txt
```