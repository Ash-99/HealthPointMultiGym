// IMPORTANT: Replace this with the URL of your DEPLOYED backend API
// For local testing, it would be 'http://127.0.0.1:5000/api/ask'
const API_URL = 'http://127.0.0.1:5000/api/ask';
// Add this to your script.js
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
}
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const answerDiv = document.getElementById('answer');
    const question = questionInput.value;

    if (!question) {
        alert('Please enter a question!');
        return;
    }

    // Show loading state
    answerDiv.classList.add('loading');
    answerDiv.textContent = 'Thinking...';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        answerDiv.textContent = data.answer;

    } catch (error) {
        console.error('Error:', error);
        answerDiv.textContent = 'Sorry, an error occurred. Please try again.';
    } finally {
        answerDiv.classList.remove('loading');
    }
}