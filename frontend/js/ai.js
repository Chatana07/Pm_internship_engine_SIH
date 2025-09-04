// Get reference to enrollment message element
const eduErrMessage = document.getElementById("eduErrMessage");

// Handle enrollment status changes
document.querySelectorAll('input[name="edu"]').forEach(radio => {
  radio.addEventListener("change", function() {
    // Process enrollment eligibility immediately when option is selected
    processEnrollmentEligibility();
    
    // Add visual feedback for selection
    this.closest('.radio').classList.add('selected');
    document.querySelectorAll('input[name="edu"]').forEach(otherRadio => {
      if (otherRadio !== this) {
        otherRadio.closest('.radio').classList.remove('selected');
      }
    });
  });
});

// Process enrollment eligibility
function processEnrollmentEligibility() {
  const selected = document.querySelector('input[name="edu"]:checked');
  
  if (!selected) return; // No option selected yet
  
  if (selected.value === "Enrolled full-time") {
    // Show rejection message
    eduErrMessage.innerHTML = "<i class='fas fa-times-circle'></i> Sorry, you are not eligible for this internship as you are enrolled in full-time study/job.";
    eduErrMessage.classList.remove("success");
    eduErrMessage.classList.add("error");
    eduErrMessage.style.display = "block";

    // Disable further actions if not eligible
    const submitBtn = document.querySelector(".submit-btn");
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.style.backgroundColor = "#9ca3af";
      submitBtn.style.cursor = "not-allowed";
      
      // Add visual indication
      submitBtn.innerHTML = "<i class='fas fa-ban'></i> Not Eligible";
    }
    
    // Highlight the section to draw attention
    document.querySelector('.edu').classList.add('error-section');
  } else {
    // If not "Enrolled full-time" → Eligible
    eduErrMessage.innerHTML = "<i class='fas fa-check-circle'></i> You are eligible to apply for this internship.";
    eduErrMessage.classList.remove("error");
    eduErrMessage.classList.add("success");
    eduErrMessage.style.display = "block";
    
    // Re-enable submit button if it was disabled and govt job is not "yes"
    const submitBtn = document.querySelector(".submit-btn");
    const govtJobSelected = document.querySelector('input[name="govtJob"]:checked');
    
    if (submitBtn && (!govtJobSelected || govtJobSelected.value !== "yes")) {
      submitBtn.disabled = false;
      submitBtn.style.backgroundColor = "";
      submitBtn.style.cursor = "";
      
      // Restore original text
      submitBtn.innerHTML = "<i class='fas fa-robot'></i> Get AI Recommendations";
    }
    
    // Remove error highlighting
    document.querySelector('.edu').classList.remove('error-section');
  }
}



// Get references to DOM elements
const govtJobForm = document.getElementById("govtJobForm");
const govtJobMessage = document.getElementById("govtJobMessage");
const govtDetailsBox = document.getElementById("govtDetailsBox"); // This might be null if commented out in HTML
const aadhaarMessage = document.getElementById("aadhaarMessage");

// Add visual feedback for form interactions
document.querySelectorAll('input, select').forEach(element => {
  // Add focus effect
  element.addEventListener('focus', function() {
    this.closest('.field')?.classList.add('focused');
  });
  
  // Remove focus effect
  element.addEventListener('blur', function() {
    this.closest('.field')?.classList.remove('focused');
  });
});

// Show/hide government details box based on selection
document.querySelectorAll('input[name="govtJob"]').forEach(radio => {
  radio.addEventListener("change", function() {
    // Only manipulate govtDetailsBox if it exists in the DOM
    if (govtDetailsBox) {
      if (this.value === "yes") {
        govtDetailsBox.style.display = "block";
      } else {
        govtDetailsBox.style.display = "none";
      }
    }
    
    // Process eligibility immediately when option is selected
    processGovtJobEligibility();
    
    // Add visual feedback for selection
    this.closest('.radio').classList.add('selected');
    document.querySelectorAll('input[name="govtJob"]').forEach(otherRadio => {
      if (otherRadio !== this) {
        otherRadio.closest('.radio').classList.remove('selected');
      }
    });
  });
});

// Process government job eligibility
function processGovtJobEligibility() {
  const selected = document.querySelector('input[name="govtJob"]:checked');
  
  if (!selected) return; // No option selected yet
  
  if (selected.value === "yes") {
    // Show rejection message
    govtJobMessage.innerHTML = "<i class='fas fa-times-circle'></i> Sorry, you are not eligible for this internship as you or your family members or your spouse has a government job.";
    govtJobMessage.classList.remove("success");
    govtJobMessage.classList.add("error");
    govtJobMessage.style.display = "block";

    // Disable further actions if not eligible
    const submitBtn = document.querySelector(".submit-btn");
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.style.backgroundColor = "#9ca3af";
      submitBtn.style.cursor = "not-allowed";
      
      // Add visual indication
      submitBtn.innerHTML = "<i class='fas fa-ban'></i> Not Eligible";
    }
    
    // Highlight the section to draw attention
    document.querySelector('.govt-job-section').classList.add('error-section');
  } else {
    // If "No" → Eligible
    govtJobMessage.innerHTML = "<i class='fas fa-check-circle'></i> You are eligible to apply for this internship.";
    govtJobMessage.classList.remove("error");
    govtJobMessage.classList.add("success");
    govtJobMessage.style.display = "block";
    
    // Re-enable submit button if it was disabled
    const submitBtn = document.querySelector(".submit-btn");
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.style.backgroundColor = "";
      submitBtn.style.cursor = "";
      
      // Restore original text
      submitBtn.innerHTML = "<i class='fas fa-robot'></i> Get AI Recommendations";
    }
    
    // Remove error highlighting
    document.querySelector('.govt-job-section').classList.remove('error-section');
  }
}

// Handle Aadhaar linking section
document.querySelectorAll('input[name="aadhaarLink"]').forEach(radio => {
  radio.addEventListener("change", function() {
    if (this.value === "yes") {
      // Show success message for Aadhaar linking
      aadhaarMessage.innerHTML = "<i class='fas fa-check-circle'></i> Great! Your bank account is linked with Aadhaar.";
      aadhaarMessage.classList.remove("error", "warning");
      aadhaarMessage.classList.add("success");
      aadhaarMessage.style.display = "block";
    } else {
      // Show warning message for Aadhaar linking
      aadhaarMessage.innerHTML = "<i class='fas fa-exclamation-triangle'></i> Please link your bank account with Aadhaar before the internship starts for stipend disbursement.";
      aadhaarMessage.classList.remove("success", "error");
      aadhaarMessage.classList.add("warning");
      aadhaarMessage.style.display = "block";
    }
    
    // Add visual feedback for selection
    this.closest('.radio').classList.add('selected');
    document.querySelectorAll('input[name="aadhaarLink"]').forEach(otherRadio => {
      if (otherRadio !== this) {
        otherRadio.closest('.radio').classList.remove('selected');
      }
    });
  });
});

// Handle main form submission
document.querySelector("form").addEventListener("submit", function(e) {
  e.preventDefault();
  
  // Check if user is eligible (not in govt job)
  const govtJobSelected = document.querySelector('input[name="govtJob"]:checked');
  if (!govtJobSelected || (govtJobSelected.value === "yes")) {
    alert("Please complete the Government Job Status section. You must not be in a government job to be eligible.");
    return;
  }
  
  // Check if user is eligible (not enrolled full-time)
  const eduSelected = document.querySelector('input[name="edu"]:checked');
  if (!eduSelected || (eduSelected.value === "Enrolled full-time")) {
    alert("Please complete the Enrollment Status section. You must not be enrolled in full-time study/job to be eligible.");
    return;
  }
  
  // Check if all required fields are filled
  const requiredFields = document.querySelectorAll("[required]");
  let allFilled = true;
  
  requiredFields.forEach(field => {
    if (!field.value && !field.checked) {
      allFilled = false;
    }
  });
  
  if (!allFilled) {
    alert("Please fill in all required fields.");
    return;
  }
  
  // Collect form data
  const formData = {
    name: document.getElementById("name").value,
    citizenship: document.getElementById("citizenship").value,
    age: document.getElementById("age").value,
    eduMin: document.getElementById("eduMin").value,
    skills: document.getElementById("skills").value,
    domain: document.getElementById("domain").value,
    location: document.getElementById("location").value,
    duration: document.getElementById("duration").value,
    edu: document.querySelector('input[name="edu"]:checked').value,
    income: document.getElementById("income").value,
    aadhaarLink: document.querySelector('input[name="aadhaarLink"]:checked').value,
    govtJob: document.querySelector('input[name="govtJob"]:checked').value
  };
  
  // Send data to backend
  getAIRecommendations(formData);
});

// Function to send data to backend and get AI recommendations
async function getAIRecommendations(formData) {
  try {
    // Show loading state
    const submitBtn = document.querySelector(".submit-btn");
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = "<i class='fas fa-spinner fa-spin'></i> Processing...";
    submitBtn.disabled = true;
    
    // Send request to backend
    const response = await fetch('http://localhost:5000/ai_recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Restore button state
    submitBtn.innerHTML = originalText;
    submitBtn.disabled = false;
    
    // Display results
    displayRecommendations(result);
    
  } catch (error) {
    console.error('Error getting AI recommendations:', error);
    alert("Error getting AI recommendations. Please try again.");
    
    // Restore button state
    const submitBtn = document.querySelector(".submit-btn");
    submitBtn.innerHTML = "<i class='fas fa-robot'></i> Get AI Recommendations";
    submitBtn.disabled = false;
  }
}

// Function to display recommendations
function displayRecommendations(result) {
  // Check if there are any recommendations
  if (result.total_recommendations === 0) {
    // Show message when no recommendations are found
    const noResultsHTML = `
      <div class="recommendations-modal">
        <div class="modal-content">
          <div class="modal-header">
            <h2><i class="fas fa-robot"></i> AI Recommendations</h2>
            <button class="close-btn">&times;</button>
          </div>
          <div class="modal-body">
            <div class="no-results-message">
              <h3><i class="fas fa-exclamation-circle"></i> No Internships Found</h3>
              <p>${result.message || 'No internships found matching your criteria. Please try adjusting your preferences.'}</p>
              <div class="user-profile-summary">
                <h4>Your Profile Summary</h4>
                <p><strong>Name:</strong> ${result.user_profile.name}</p>
                <p><strong>Education:</strong> ${result.user_profile.education}</p>
                <p><strong>Skills:</strong> ${result.user_profile.skills}</p>
                <p><strong>Preferred Domain:</strong> ${result.user_profile.preferred_domain}</p>
                <p><strong>Preferred Location:</strong> ${result.user_profile.preferred_location}</p>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button class="btn primary close-modal-btn">Close</button>
          </div>
        </div>
      </div>
    `;
    
    // Add modal to page
    document.body.insertAdjacentHTML('beforeend', noResultsHTML);
  } else {
    // Create a modal to display results
    const recommendationsHTML = `
      <div class="recommendations-modal">
        <div class="modal-content">
          <div class="modal-header">
            <h2><i class="fas fa-robot"></i> AI Recommendations</h2>
            <button class="close-btn">&times;</button>
          </div>
          <div class="modal-body">
            <div class="user-profile-summary">
              <h3>Your Profile Summary</h3>
              <p><strong>Name:</strong> ${result.user_profile.name}</p>
              <p><strong>Education:</strong> ${result.user_profile.education}</p>
              <p><strong>Skills:</strong> ${result.user_profile.skills}</p>
              <p><strong>Preferred Domain:</strong> ${result.user_profile.preferred_domain}</p>
              <p><strong>Preferred Location:</strong> ${result.user_profile.preferred_location}</p>
            </div>
            
            <div class="recommendations-list">
              <h3>${result.message || `Top ${result.total_recommendations} Internship Recommendation${result.total_recommendations !== 1 ? 's' : ''}`}</h3>
              ${result.recommendations.map((rec, index) => `
                <div class="recommendation-card">
                  <div class="card-header">
                    <h4>${index + 1}. ${rec.company} - ${rec.role}</h4>
                    <span class="similarity-score">Score: ${(rec.similarity_score * 100).toFixed(1)}%</span>
                  </div>
                  <div class="card-body">
                    <p><strong>Domain:</strong> ${rec.domain}</p>
                    <p><strong>Location:</strong> ${rec.location}</p>
                    <p><strong>Type:</strong> ${rec.type}</p>
                    <p><strong>Duration:</strong> ${rec.duration}</p>
                    <p><strong>Stipend:</strong> ${rec.stipend}</p>
                    <p><strong>Why Recommended:</strong> ${rec.reason}</p>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
          <div class="modal-footer">
            <button class="btn primary close-modal-btn">Close</button>
          </div>
        </div>
      </div>
    `;
    
    // Add modal to page
    document.body.insertAdjacentHTML('beforeend', recommendationsHTML);
  }
  
  // Add event listeners for closing modal
  document.querySelector('.close-btn').addEventListener('click', closeModal);
  document.querySelector('.close-modal-btn').addEventListener('click', closeModal);
  
  // Close modal when clicking outside
  document.querySelector('.recommendations-modal').addEventListener('click', function(e) {
    if (e.target === this) {
      closeModal();
    }
  });
  
  // Show modal
  document.querySelector('.recommendations-modal').style.display = 'block';
}

// Function to close modal
function closeModal() {
  const modal = document.querySelector('.recommendations-modal');
  if (modal) {
    modal.remove();
  }
}

// Add CSS for recommendations modal
document.head.insertAdjacentHTML('beforeend', `
  <style>
    .focused {
      background-color: rgba(25, 118, 210, 0.05);
      padding: 5px;
      border-radius: 5px;
      transition: all 0.3s ease;
    }
    
    .radio.selected {
      background-color: #e3f2fd;
      border-left: 3px solid #1976d2;
      font-weight: 500;
    }
    
    .error-section {
      border: 1px solid #ffcdd2 !important;
      background-color: #ffebee !important;
    }
    
    /* Smooth transitions */
    .section, .field, .radio, .btn, .message {
      transition: all 0.3s ease;
    }
    
    /* Hover effects for better desktop experience */
    .section:hover {
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Recommendations Modal */
    .recommendations-modal {
      display: none;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.7); /* Darker background for better contrast */
      overflow: auto;
    }
    
    .modal-content {
      background-color: #fff;
      margin: 2% auto;
      padding: 0;
      border-radius: 8px;
      width: 90%;
      max-width: 800px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
      max-height: 90vh;
      overflow-y: auto;
    }
    
    .modal-header {
      padding: 20px;
      background-color: #1976d2;
      color: white;
      border-top-left-radius: 8px;
      border-top-right-radius: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .modal-header h2 {
      margin: 0;
      font-size: 1.5rem;
    }
    
    .close-btn {
      background: none;
      border: none;
      color: white;
      font-size: 2rem;
      cursor: pointer;
      padding: 0;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .modal-body {
      padding: 20px;
      background-color: #ffffff; /* Ensure white background */
      color: #333333; /* Dark text for better readability */
    }
    
    .user-profile-summary {
      background-color: #f5f5f5;
      padding: 15px;
      border-radius: 5px;
      margin-bottom: 20px;
      color: #333333; /* Dark text */
    }
    
    .user-profile-summary h3, .user-profile-summary h4 {
      margin-top: 0;
      color: #1976d2;
    }
    
    .recommendations-list h3 {
      color: #1976d2;
      border-bottom: 2px solid #1976d2;
      padding-bottom: 5px;
    }
    
    .recommendation-card {
      border: 1px solid #ddd;
      border-radius: 5px;
      margin-bottom: 15px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      background-color: #ffffff; /* White background */
      color: #333333; /* Dark text */
    }
    
    .card-header {
      background-color: #e3f2fd;
      padding: 15px;
      border-top-left-radius: 5px;
      border-top-right-radius: 5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .card-header h4 {
      margin: 0;
      color: #1976d2;
    }
    
    .similarity-score {
      background-color: #1976d2;
      color: white;
      padding: 5px 10px;
      border-radius: 15px;
      font-size: 0.9rem;
    }
    
    .card-body {
      padding: 15px;
      background-color: #ffffff; /* White background */
      color: #333333; /* Dark text */
    }
    
    .card-body p {
      margin: 5px 0;
      color: #333333; /* Ensure dark text */
    }
    
    .modal-footer {
      padding: 20px;
      text-align: right;
      border-top: 1px solid #eee;
      background-color: #ffffff; /* White background */
    }
    
    .close-modal-btn {
      background-color: #1976d2;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 1rem;
    }
    
    .close-modal-btn:hover {
      background-color: #1565c0;
    }
    
    /* Ensure all text in modal is visible */
    .recommendations-modal * {
      color: #333333;
    }
    
    .recommendations-modal h1, 
    .recommendations-modal h2, 
    .recommendations-modal h3, 
    .recommendations-modal h4 {
      color: #1976d2;
    }
    
    /* No results message */
    .no-results-message {
      text-align: center;
      padding: 20px;
    }
    
    .no-results-message h3 {
      color: #1976d2;
      margin-bottom: 15px;
    }
    
    .no-results-message p {
      font-size: 1.1rem;
      margin-bottom: 20px;
    }
    
    .no-results-message .user-profile-summary {
      text-align: left;
      margin-top: 20px;
    }
  </style>
`);

// Add smooth scrolling to sections
document.querySelectorAll('.subhead').forEach(subhead => {
  subhead.addEventListener('click', function() {
    const section = this.closest('.section');
    section.scrollIntoView({ behavior: 'smooth' });
  });
});