const API_BASE_URL = 'https://weatherwise-backend-ok73.onrender.com'; // Update with your backend URL
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

// DOM Elements
const singleTab = document.getElementById("singleTab");
const bulkTab = document.getElementById("bulkTab");
const singlePredictionSection = document.getElementById("singlePredictionSection");
const bulkPredictionSection = document.getElementById("bulkPredictionSection");
const fileInput = document.getElementById("dataset");
const fileNameDisplay = document.getElementById("fileNameDisplay");

// Tab switching
singleTab.addEventListener("click", () => {
    singleTab.classList.add("active");
    bulkTab.classList.remove("active");
    singlePredictionSection.classList.add("active");
    bulkPredictionSection.classList.remove("active");
});

bulkTab.addEventListener("click", () => {
    bulkTab.classList.add("active");
    singleTab.classList.remove("active");
    bulkPredictionSection.classList.add("active");
    singlePredictionSection.classList.remove("active");
});

// File input display
fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        fileNameDisplay.textContent = "No file selected";
    }
});

// Drag and drop functionality
const fileUploadBox = document.querySelector(".file-upload-box");
fileUploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileUploadBox.classList.add("dragover");
});

fileUploadBox.addEventListener("dragleave", () => {
    fileUploadBox.classList.remove("dragover");
});

fileUploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    fileUploadBox.classList.remove("dragover");
    
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileNameDisplay.textContent = fileInput.files[0].name;
    }
});

// Initialize form fields
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
        input.placeholder = `Enter ${field.replace(/_/g, " ")}`;
        
        div.appendChild(label);
        div.appendChild(input);
        formGrid.appendChild(div);
    });
}

// Spinner control
function toggleSpinner(spinnerId, show) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = show ? 'inline-block' : 'none';
    }
}

// Session storage for predictions
const SINGLE_PREDICTION_KEY = 'singlePredictionData';
const BULK_PREDICTION_KEY = 'bulkPredictionData';

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
    const weatherIcon = document.getElementById("weatherIcon");
    
    const isRain = prediction === 1;
    const confidencePercent = Math.round(probability * 100);
    
    // Update styling and content
    resultContainer.className = `result-container ${isRain ? 'weather-rain' : 'weather-sun'}`;
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceBar.style.backgroundColor = isRain ? '#3498db' : '#f1c40f';
    
    weatherPercentage.textContent = isRain 
        ? `RAIN LIKELY (${confidencePercent}% confidence)`
        : `SUNNY (${confidencePercent}% confidence)`;
    
    weatherExplanation.textContent = isRain
        ? "High probability of precipitation expected"
        : "Clear skies and sunny weather expected";
    
    weatherIcon.innerHTML = isRain 
        ? '<i class="fas fa-cloud-rain"></i>'
        : '<i class="fas fa-sun"></i>';
    
    resultContainer.style.display = 'block';
}

// Display bulk prediction results
function showBulkPredictionResults(results) {
    const resultContainer = document.getElementById('bulkPredictionResult');
    const resultsContent = document.getElementById('bulkResultsContent');
    resultsContent.innerHTML = '';
    resultContainer.style.display = 'none';

    if (!results || !Array.isArray(results)) {
        resultsContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> Invalid results format received from server
            </div>
        `;
        resultContainer.style.display = 'block';
        return;
    }

    const rainCount = results.filter(p => p.prediction === 1).length;
    const total = results.length;
    const rainPercent = total > 0 ? Math.round((rainCount / total) * 100) : 0;

    // Build summary card
    const summaryHTML = `
        <div class="summary-card ${rainCount > 0 ? 'has-rain' : ''}">
            <div class="summary-header">
                <h4><i class="fas fa-chart-pie"></i> Batch Summary</h4>
            </div>
            <div class="summary-content">
                <div class="summary-metric">
                    <span class="metric-value">${total}</span>
                    <span class="metric-label">Total Records</span>
                </div>
                <div class="summary-metric">
                    <span class="metric-value ${rainCount > 0 ? 'rain-value' : ''}">${rainCount}</span>
                    <span class="metric-label">Rainy Days</span>
                </div>
                <div class="summary-metric">
                    <span class="metric-value">${rainPercent}%</span>
                    <span class="metric-label">Rain Probability</span>
                </div>
            </div>
        </div>
    `;

    // Build results table
    let tableHTML = `
        <div class="results-table-container">
            <div class="table-header">
                <h4><i class="fas fa-table"></i> Detailed Results</h4>
                <button id="exportResults" class="export-button">
                    <i class="fas fa-download"></i> Export Results
                </button>
            </div>
            <div class="table-scroll">
                <table class="result-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            <th>Details</th>
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
            <tr class="${isRain ? 'rain-row' : 'sun-row'}">
                <td>${index + 1}</td>
                <td>
                    <span class="weather-badge ${isRain ? 'badge-rain' : 'badge-sun'}">
                        <i class="fas ${isRain ? 'fa-cloud-rain' : 'fa-sun'}"></i>
                        ${isRain ? 'RAIN' : 'SUN'}
                    </span>
                </td>
                <td>${confidence}</td>
                <td>
                    <button class="details-button" data-index="${index}">
                        <i class="fas fa-info-circle"></i> Details
                    </button>
                </td>
            </tr>
        `;
    });

    tableHTML += `</tbody></table></div></div>`;
    
    resultsContent.innerHTML = summaryHTML + tableHTML;
    resultContainer.style.display = 'block';

    // Add event listeners to detail buttons
    document.querySelectorAll('.details-button').forEach(button => {
        button.addEventListener('click', (e) => {
            const index = e.target.getAttribute('data-index') || 
                          e.target.parentElement.getAttribute('data-index');
            showPredictionDetails(results[index], parseInt(index) + 1);
        });
    });

    // Add export functionality
    document.getElementById('exportResults')?.addEventListener('click', () => {
        exportResultsToCSV(results);
    });
}

// Show detailed prediction view
function showPredictionDetails(prediction, recordNumber) {
    const modal = document.createElement('div');
    modal.className = 'prediction-modal';
    
    const isRain = prediction.prediction === 1;
    const confidencePercent = prediction.probability ? Math.round(prediction.probability * 100) : 0;
    
    modal.innerHTML = `
        <div class="modal-content ${isRain ? 'weather-rain' : 'weather-sun'}">
            <span class="close-modal">&times;</span>
            <h3><i class="fas fa-info-circle"></i> Prediction Details</h3>
            <p class="record-number">Record #${recordNumber}</p>
            
            <div class="confidence-meter-container">
                <div class="confidence-labels">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                </div>
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: ${confidencePercent}%; 
                        background-color: ${isRain ? '#3498db' : '#f1c40f'}"></div>
                </div>
            </div>
            
            <div class="weather-percentage">
                ${isRain ? 'RAIN LIKELY' : 'SUNNY'} (${confidencePercent}% confidence)
            </div>
            
            <div class="weather-icon-large">
                <i class="fas ${isRain ? 'fa-cloud-rain' : 'fa-sun'}"></i>
            </div>
            
            <div class="weather-explanation">
                ${isRain ? 
                    'High probability of precipitation expected' : 
                    'Clear skies and sunny weather expected'}
            </div>
            
            <div class="raw-data">
                <h4><i class="fas fa-database"></i> Raw Prediction Data</h4>
                <pre>${JSON.stringify(prediction, null, 2)}</pre>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal when clicking X or outside
    modal.querySelector('.close-modal').addEventListener('click', () => {
        modal.remove();
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Export results to CSV
function exportResultsToCSV(results) {
    if (!results || !Array.isArray(results)) return;
    
    const headers = ['Record', 'Prediction', 'Confidence'];
    const csvRows = [];
    
    // Add header row
    csvRows.push(headers.join(','));
    
    // Add data rows
    results.forEach((result, index) => {
        const prediction = result.prediction === 1 ? 'Rain' : 'Sun';
        const confidence = result.probability ? 
            `${Math.round(result.probability * 100)}%` : 'N/A';
        
        csvRows.push([
            index + 1,
            prediction,
            confidence
        ].join(','));
    });
    
    // Create CSV content
    const csvContent = csvRows.join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `weather_predictions_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Handle bulk prediction form submission
async function handleBulkPrediction(e) {
    e.preventDefault();
    
    if (!fileInput.files.length) {
        showToast('Please select a file first', 'error');
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
            let errorDetail = 'Prediction failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorDetail;
            } catch (e) {
                console.error('Error parsing error response:', e);
            }
            throw new Error(errorDetail);
        }

        const result = await response.json();
        
        if (!result.results || !Array.isArray(result.results)) {
            throw new Error("Invalid server response format");
        }

        saveBulkPredictionToSession(result.results);
        showBulkPredictionResults(result.results);
        showToast('Bulk prediction completed successfully!', 'success');

    } catch (error) {
        console.error("Bulk prediction error:", error);
        showToast(`Prediction failed: ${error.message}`, 'error');
    } finally {
        toggleSpinner('bulkPredictionSpinner', false);
    }
}

// Handle single prediction form submission
async function handleSinglePrediction(e) {
    e.preventDefault();
    toggleSpinner('singlePredictionSpinner', true);

    const formData = {};
    let isValid = true;
    
    EXPECTED_FIELDS.forEach(field => {
        const value = document.getElementById(field).value;
        if (value === '' || isNaN(value)) {
            isValid = false;
            document.getElementById(field).classList.add('error');
        } else {
            formData[field] = parseFloat(value);
            document.getElementById(field).classList.remove('error');
        }
    });

    if (!isValid) {
        showToast('Please fill in all fields with valid numbers', 'error');
        toggleSpinner('singlePredictionSpinner', false);
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/predict-single/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            let errorDetail = 'Prediction failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorDetail;
            } catch (e) {
                console.error('Error parsing error response:', e);
            }
            throw new Error(errorDetail);
        }

        const result = await response.json();
        saveSinglePredictionToSession(result.prediction, result.probability);
        showPredictionResult(result.prediction, result.probability);
        showToast('Prediction completed successfully!', 'success');

    } catch (error) {
        console.error("Prediction error:", error);
        showToast(`Prediction failed: ${error.message}`, 'error');
    } finally {
        toggleSpinner('singlePredictionSpinner', false);
    }
}

// Show toast notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 
                          type === 'success' ? 'fa-check-circle' : 
                          'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 5000);
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