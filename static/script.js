document.getElementById('safetyForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent the form from submitting in the traditional way

    const age = document.getElementById('age').value;
    const chemicals = document.getElementById('chemicals').value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ age: age, chemicals: chemicals }),
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
});
