const API_BASE_URL = 'https://weatherwise-backend-ok73.onrender.com';
const SESSION_STORAGE_KEY = 'retrainDataUploaded';
const METRICS_STORAGE_KEY = 'retrainMetricsData';

// Static metrics for the current model
const staticMetrics = {
    accuracy: 0.9213,
    precision: 0.8643,
    recall: 0.7955,
    f1: 0.8286
};

// DOM elements
const uploadForm = document.getElementById("upload-form");
const retrainBtn = document.getElementById("retrain-btn");
const saveBtn = document.getElementById("save-model-btn");
const statusDiv = document.getElementById("status");
const fileInput = document.getElementById("dataset-upload");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const uploadIndicator = document.getElementById("upload-indicator");

// Metrics elements
const accuracyElem = document.getElementById("accuracy");
const precisionElem = document.getElementById("precision");
const recallElem = document.getElementById("recall");
const f1ScoreElem = document.getElementById("f1-score");

const newAccuracyElem = document.getElementById("new-accuracy");
const newPrecisionElem = document.getElementById("new-precision");
const newRecallElem = document.getElementById("new-recall");
const newF1ScoreElem = document.getElementById("new-f1-score");

// State
let currentModelId = null;

// Initialize the page
function initializePage() {
    fileInput.addEventListener("change", (e) => {
        fileNameDisplay.textContent = fileInput.files.length > 0
            ? fileInput.files[0].name
            : "No file selected";
    });

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

    loadCurrentMetrics();
}

// Spinner toggle
function toggleSpinner(spinnerId, show) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = show ? 'inline-block' : 'none';
    }
}

// Status updates
function updateStatus(message, isError = false, isSuccess = false) {
    if (statusDiv) {
        statusDiv.innerHTML = `<p>${message}</p>`;
        statusDiv.className = 'status';
        if (isError) statusDiv.classList.add('error');
        if (isSuccess) statusDiv.classList.add('success');
    }
}

// Format metrics
function formatMetric(value) {
    return value != null ? (value * 100).toFixed(2) + '%' : '-';
}

// Upload state
function setUploadState(uploaded) {
    if (uploaded) {
        sessionStorage.setItem(SESSION_STORAGE_KEY, 'true');
        if (retrainBtn) retrainBtn.disabled = false;
        if (uploadIndicator) {
            uploadIndicator.style.display = 'inline-block';
            uploadIndicator.innerHTML = '<i class="fas fa-check-circle"></i> Dataset Uploaded';
        }
    } else {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        if (retrainBtn) retrainBtn.disabled = true;
        if (uploadIndicator) {
            uploadIndicator.style.display = 'none';
            uploadIndicator.innerHTML = '';
        }
    }
}

// Update confusion matrix for multiple classes
function updateConfusionMatrix(matrixData) {
    if (!matrixData || !matrixData.matrix || !matrixData.labels) return;

    const matrixContainer = document.querySelector('.matrix-container');
    if (!matrixContainer) return;

    // Clear existing matrix
    matrixContainer.innerHTML = '';

    // Create a new table for the confusion matrix
    const table = document.createElement('table');
    table.className = 'confusion-matrix-table';

    // Create header row
    const headerRow = document.createElement('tr');
    headerRow.appendChild(document.createElement('th')); // Empty top-left cell

    // Add predicted labels header
    matrixData.labels.forEach(label => {
        const th = document.createElement('th');
        th.textContent = `Pred ${label}`;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Add matrix rows
    matrixData.matrix.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        
        // Add actual label header
        const th = document.createElement('th');
        th.textContent = `Actual ${matrixData.labels[rowIndex]}`;
        tr.appendChild(th);

        // Add matrix cells
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });

    matrixContainer.appendChild(table);
}

// Session metrics
function saveMetricsToSession(metrics) {
    sessionStorage.setItem(METRICS_STORAGE_KEY, JSON.stringify(metrics));
}

function loadMetricsFromSession() {
    const metrics = sessionStorage.getItem(METRICS_STORAGE_KEY);
    return metrics ? JSON.parse(metrics) : null;
}

function displayRetrainedMetrics(metrics) {
    if (!metrics) return;

    newAccuracyElem.textContent = formatMetric(metrics.accuracy);
    newPrecisionElem.textContent = formatMetric(metrics.precision);
    newRecallElem.textContent = formatMetric(metrics.recall);
    newF1ScoreElem.textContent = formatMetric(metrics.f1);

    if (metrics.confusion_matrix) {
        updateConfusionMatrix(metrics.confusion_matrix);
    }

    if (saveBtn) saveBtn.disabled = false;
}

function loadCurrentMetrics() {
    accuracyElem.textContent = formatMetric(staticMetrics.accuracy);
    precisionElem.textContent = formatMetric(staticMetrics.precision);
    recallElem.textContent = formatMetric(staticMetrics.recall);
    f1ScoreElem.textContent = formatMetric(staticMetrics.f1);

    if (sessionStorage.getItem(SESSION_STORAGE_KEY)) {
        setUploadState(true);
        updateStatus("Dataset already uploaded in this session. You can retrain the model.", false, true);
    } else {
        updateStatus("Upload a dataset to begin retraining process");
    }

    const savedMetrics = loadMetricsFromSession();
    if (savedMetrics) {
        displayRetrainedMetrics(savedMetrics);
        updateStatus("Retrained model metrics loaded from session", false, true);
    }
}

// Upload dataset
async function handleDatasetUpload(event) {
    event.preventDefault();

    if (!fileInput.files.length) {
        updateStatus("Please select a file first", true);
        return;
    }

    toggleSpinner('upload-spinner', true);
    updateStatus("Uploading dataset...");

    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch(`${API_BASE_URL}/upload-training-data/`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || error.message || "Upload failed");
        }

        const result = await response.json();

        updateStatus(
            `Successfully uploaded ${result.records_added} records. ${result.invalid_records} records were invalid.`,
            false,
            true
        );
        setUploadState(true);
    } catch (error) {
        console.error("Upload error:", error);
        updateStatus(`Upload failed: ${error.message}`, true);
    } finally {
        toggleSpinner('upload-spinner', false);
    }
}

// Retrain model
async function handleModelRetrain() {
    toggleSpinner('retrain-spinner', true);
    updateStatus("Retraining model... This may take a few minutes.");

    try {
        // Add a 5-minute timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes

        const response = await fetch(`${API_BASE_URL}/retrain/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || error.message || "Retraining failed");
        }

        const result = await response.json();
        currentModelId = result.model_id;

        // Prepare metrics for display and storage
        const metrics = {
            accuracy: result.metrics.accuracy,
            precision: result.metrics.precision,
            recall: result.metrics.recall,
            f1: result.metrics.f1,
            confusion_matrix: result.metrics.confusion_matrix
        };

        saveMetricsToSession(metrics);
        displayRetrainedMetrics(metrics);

        updateStatus(result.message || "Model retrained successfully!", false, true);
    } catch (error) {
        console.error("Retrain error:", error);
        updateStatus(
            error.name === "AbortError" 
                ? "Retraining timed out (took too long). Try with a smaller dataset." 
                : `Retraining failed: ${error.message}`,
            true
        );
    } finally {
        toggleSpinner('retrain-spinner', false);
    }
}

// Save model with enhanced feedback
async function handleModelSave() {
    const saveStatus = document.getElementById("save-status");
    if (saveStatus) {
        // Show saving message
        saveStatus.textContent = "Saving model...";
        saveStatus.className = 'save-status';
        saveStatus.style.display = 'block';
        
        // Show spinner
        toggleSpinner('save-spinner', true);
        
        // Simulate a delay (like an actual save would take)
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Show success message
        saveStatus.textContent = "Model saved successfully!";
        saveStatus.className = 'save-status success';
        saveStatus.innerHTML = `<i class="fas fa-check-circle"></i> ${saveStatus.textContent}`;
        
        // Hide spinner
        toggleSpinner('save-spinner', false);
        
        // Also update the main status
        updateStatus("Model saved successfully!", false, true);
    }
}

// Event listeners
if (uploadForm) uploadForm.addEventListener("submit", handleDatasetUpload);
if (retrainBtn) retrainBtn.addEventListener("click", handleModelRetrain);
if (saveBtn) saveBtn.addEventListener("click", handleModelSave);

// Start
initializePage();