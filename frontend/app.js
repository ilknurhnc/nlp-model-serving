const button = document.getElementById("analyzeButton");

button.addEventListener("click", async () => {

    const input = document.getElementById("messageInput").value;

    const resultBox = document.getElementById("resultBox");

    try {

        const response = await fetch("http://localhost:8001/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: input
            })
        });

        const data = await response.json();

        console.log(data);

        resultBox.innerHTML = `
            <p><strong>Label:</strong> ${data.predicted_label}</p>
            <p><strong>Confidence:</strong> ${data.confidence}</p>
        `;

    } catch (error) {

        console.error(error);

        resultBox.innerHTML = `
            <p>Error connecting to API.</p>
        `;
    }
});