const API_BASE_URL = 'https://weatherwise-backend-ok73.onrender.com';
const EXPECTED_FIELDS = [
    "precipitation", 
    "temp_max", 
    "temp_min",
    "wind",
    "lag_wind_1",
    "lag_precipitation_1",
    "lag_temp_max_1",
    "lag_temp_min_1"
];

// Storage keys
const SINGLE_PREDICTION_KEY = 'singlePredictionData';
const BULK_PREDICTION_KEY = 'bulkPredictionData';

// Initialize form fields dynamically
function initializeFormFields() {
    const formGrid = document.getElementById("formGridContainer");
    formGrid.innerHTML = '';
    
    EXPECTED_FIELDS.forEach(field => {
        const div = document.createElement("div");
        div.className = "form-item";
        
        const label = document.createElement("label");
        label.htmlFor = field;
        label.textContent = `${field.replace(/_/g, " ")}:`;
        
        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.0001";
        input.id = field;
        input.name = field;
        input.required = true;
        
        div.appendChild(label);
        div.appendChild(input);
        formGrid.appendChild(div);
    });
}

// Spinner visibility control
function toggleSpinner(spinnerId, show) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
}

// Session storage handlers
function saveSinglePredictionToSession(prediction, probability) {
    sessionStorage.setItem(SINGLE_PREDICTION_KEY, JSON.stringify({
        prediction,
        probability,
        timestamp: Date.now()
    }));
}

function saveBulkPredictionToSession(results) {
    sessionStorage.setItem(BULK_PREDICTION_KEY, JSON.stringify({
        results,
        timestamp: Date.now()
    }));
}

function loadSinglePredictionFromSession() {
    const data = sessionStorage.getItem(SINGLE_PREDICTION_KEY);
    return data ? JSON.parse(data) : null;
}

function loadBulkPredictionFromSession() {
    const data = sessionStorage.getItem(BULK_PREDICTION_KEY);
    return data ? JSON.parse(data) : null;
}

// Display single prediction results
function showPredictionResult(prediction, probability) {
    const resultContainer = document.getElementById("singlePredictionResult");
    const confidenceBar = document.getElementById("confidenceBar");
    const weatherPercentage = document.getElementById("weatherPercentage");
    const weatherExplanation = document.getElementById("weatherExplanation");
    
    const isRain = prediction === 1;
    const confidencePercent = Math.round(probability * 100);
    
    // Update styling
    resultContainer.className = `result-container ${isRain ? 'weather-rain' : 'weather-sun'}`;
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceBar.style.backgroundColor = isRain ? '#3498db' : '#f1c40f';
    
    // Update content
    weatherPercentage.textContent = isRain 
        ? `RAIN LIKELY (${confidencePercent}% chance)`
        : `SUNNY (${confidencePercent}% chance)`;
    
    weatherExplanation.textContent = isRain
        ? "High probability of precipitation"
        : "Clear skies expected";
    
    resultContainer.style.display = 'block';
}

// Display bulk prediction results
function showBulkPredictionResults(results) {
    const resultContainer = document.getElementById('bulkPredictionResult');
    resultContainer.innerHTML = '';
    resultContainer.style.display = 'none';

    if (!results || !Array.isArray(results)) {
        resultContainer.innerHTML = `
            <div class="weather-rain" style="padding: 15px;">
                <strong>Error:</strong> Invalid results format
            </div>
        `;
        resultContainer.style.display = 'block';
        return;
    }

    const rainCount = results.filter(p => p.prediction === 1).length;
    const total = results.length;
    const rainPercent = total > 0 ? Math.round((rainCount / total) * 100) : 0;

    // Build summary
    const summaryHTML = `
        <div class="summary-card">
            <strong>Batch Summary:</strong> 
            Processed ${total} records with
            <span class="weather-badge ${rainCount ? 'badge-rain' : 'badge-sun'}">
                ${rainCount} rainy days (${rainPercent}%)
            </span>
        </div>
    `;

    // Build results table
    let tableHTML = `
        <table class="result-table">
            <thead>
                <tr>
                    <th>Record #</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((pred, index) => {
        const isRain = pred.prediction === 1;
        const confidence = pred.probability 
            ? `${Math.round(pred.probability * 100)}%`
            : 'N/A';

        tableHTML += `
            <tr>
                <td>${index + 1}</td>
                <td>
                    <span class="weather-badge ${isRain ? 'badge-rain' : 'badge-sun'}">
                        ${isRain ? 'RAIN' : 'SUN'}
                    </span>
                </td>
                <td>${confidence}</td>
            </tr>
        `;
    });

    tableHTML += `</tbody></table>`;
    
    resultContainer.innerHTML = summaryHTML + tableHTML;
    resultContainer.style.display = 'block';
}

// Handle bulk prediction form submission
async function handleBulkPrediction(e) {
    e.preventDefault();
    const fileInput = document.getElementById("dataset");
    
    if (!fileInput.files.length) {
        alert("Please select a file first");
        return;
    }

    toggleSpinner('bulkPredictionSpinner', true);
    
    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch(`${API_BASE_URL}/predict-bulk/`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const result = await response.json();
        
        if (!result.results || !Array.isArray(result.results)) {
            throw new Error("Invalid server response format");
        }

        saveBulkPredictionToSession(result.results);
        showBulkPredictionResults(result.results);

    } catch (error) {
        console.error("Bulk prediction error:", error);
        document.getElementById('bulkPredictionResult').innerHTML = `
            <div class="weather-rain" style="padding: 15px;">
                <strong>Error:</strong> ${error.message || "Prediction failed"}
            </div>
        `;
        document.getElementById('bulkPredictionResult').style.display = 'block';
    } finally {
        toggleSpinner('bulkPredictionSpinner', false);
    }
}

// Handle single prediction form submission
async function handleSinglePrediction(e) {
    e.preventDefault();
    toggleSpinner('singlePredictionSpinner', true);

    const formData = {};
    EXPECTED_FIELDS.forEach(field => {
        formData[field] = parseFloat(document.getElementById(field).value);
    });

    try {
        const response = await fetch(`${API_BASE_URL}/predict-single/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const result = await response.json();
        saveSinglePredictionToSession(result.prediction, result.probability);
        showPredictionResult(result.prediction, result.probability);

    } catch (error) {
        console.error("Prediction error:", error);
        alert(`Prediction failed: ${error.message}`);
    } finally {
        toggleSpinner('singlePredictionSpinner', false);
    }
}

// Load previous results from session storage
function loadPreviousResults() {
    const singlePrediction = loadSinglePredictionFromSession();
    if (singlePrediction) {
        showPredictionResult(singlePrediction.prediction, singlePrediction.probability);
    }

    const bulkPrediction = loadBulkPredictionFromSession();
    if (bulkPrediction) {
        showBulkPredictionResults(bulkPrediction.results);
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeFormFields();
    document.getElementById("singlePredictionForm")
        .addEventListener("submit", handleSinglePrediction);
    document.getElementById("bulkPredictionForm")
        .addEventListener("submit", handleBulkPrediction);
    loadPreviousResults();
});