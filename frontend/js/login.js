const registerBtn = document.getElementById("registerBtn");
const loginBtn = document.getElementById("loginBtn");
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modalTitle");
const authForm = document.getElementById("authForm");
const profilePic = document.getElementById("profilePic");
const closeModalBtn = document.getElementById("closeModal");

let currentMode = "register";

// Open modal
registerBtn.addEventListener("click", () => {
  currentMode = "register";
  modalTitle.textContent = "Register";
  modal.style.display = "flex";
});

loginBtn.addEventListener("click", () => {
  currentMode = "login";
  modalTitle.textContent = "Login";
  modal.style.display = "flex";
});

// Close modal
closeModalBtn.addEventListener("click", () => {
  modal.style.display = "none";
});

// Handle form submit
authForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const picInput = document.getElementById("pic");

  if (picInput.files.length > 0) {
    const file = picInput.files[0];
    const reader = new FileReader();

    reader.onload = function (event) {
      profilePic.src = event.target.result; // show uploaded image
      profilePic.title = `${name} (${email})`;
      profilePic.style.display = "inline-block";
    };

    reader.readAsDataURL(file);
  }

  // Hide only buttons, NOT the logo
  registerBtn.style.display = "none";
  loginBtn.style.display = "none";

  // Close modal
  modal.style.display = "none";
});

// Close modal when clicking outside
window.addEventListener("click", (e) => {
  if (e.target === modal) {
    modal.style.display = "none";
  }
});


const govtJobForm = document.getElementById("govtJobForm");
const govtDetailsBox = document.getElementById("govtDetailsBox");
const govtJobMessage = document.getElementById("govtJobMessage");

// Show/hide details box based on selection
document.querySelectorAll('input[name="govtJob"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (input.value === "yes") {
      govtDetailsBox.style.display = "block";
    } else {
      govtDetailsBox.style.display = "none";
    }
  });
});

