import urllib.request
import urllib.parse
import json

# This test shows what the API could look like if it accepted a top_k parameter
# This is just for demonstration - the current API doesn't support this

print("=== API MODIFICATION DEMONSTRATION ===")
print("The current API endpoint is hardcoded to return 3 recommendations.")
print("To support requesting a specific number of recommendations, we would need to:")
print("1. Modify the API endpoint to accept a 'top_k' parameter")
print("2. Pass this parameter to the ML model's get_recommendations_for_profile method")
print("")
print("Example of what the modified API call might look like:")
print("""
@app.route('/ai_recommend', methods=['POST'])
def get_ai_recommendations():
    
    # Get top_k from request, default to 3
    top_k = data.get('top_k', 3)
    
    # Get recommendations with specified top_k
    recommendations = ml_matcher.get_recommendations_for_profile(user_profile, top_k)
    
    # ... rest of existing code ...
""")

print("")
print("With this modification, we could test single result behavior by requesting top_k=1")