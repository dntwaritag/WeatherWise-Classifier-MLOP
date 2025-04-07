const API_BASE_URL = 'https://weatherwise-backend-ok73.onrender.com';
const SESSION_STORAGE_KEY = 'retrainDataUploaded';
const METRICS_STORAGE_KEY = 'retrainMetricsData';
        
// Static metrics from your notebook
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
const uploadIndicator = document.createElement('span');
uploadIndicator.className = 'upload-indicator';
uploadIndicator.style.display = 'none';

if (retrainBtn && retrainBtn.parentNode) {
    retrainBtn.parentNode.insertBefore(uploadIndicator, retrainBtn.nextSibling);
}

// Metrics elements
const accuracyElem = document.getElementById("accuracy");
const precisionElem = document.getElementById("precision");
const recallElem = document.getElementById("recall");
const f1ScoreElem = document.getElementById("f1-score");

const newAccuracyElem = document.getElementById("new-accuracy");
const newPrecisionElem = document.getElementById("new-precision");
const newRecallElem = document.getElementById("new-recall");
const newF1ScoreElem = document.getElementById("new-f1-score");

// Confusion Matrix elements
const trueNegativeElem = document.getElementById("true-negative");
const falsePositiveElem = document.getElementById("false-positive");
const falseNegativeElem = document.getElementById("false-negative");
const truePositiveElem = document.getElementById("true-positive");

// State
let currentModelId = null;

function toggleSpinner(spinnerId, show) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
}

function updateStatus(message, isError = false, isSuccess = false) {
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = 'status';
        if (isError) statusDiv.classList.add('error');
        if (isSuccess) statusDiv.classList.add('success');
    }
}

function formatMetric(value) {
    return value !== null && value !== undefined ? value.toFixed(4) : '-';
}

function setUploadState(uploaded) {
    if (uploaded) {
        sessionStorage.setItem(SESSION_STORAGE_KEY, 'true');
        if (retrainBtn) retrainBtn.disabled = false;
        uploadIndicator.style.display = 'inline-block';
    } else {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        if (retrainBtn) retrainBtn.disabled = true;
        uploadIndicator.style.display = 'none';
    }
}

function updateConfusionMatrix(matrix) {
    if (!matrix) return;
    
    if (trueNegativeElem) trueNegativeElem.textContent = matrix.true_negative || '-';
    if (falsePositiveElem) falsePositiveElem.textContent = matrix.false_positive || '-';
    if (falseNegativeElem) falseNegativeElem.textContent = matrix.false_negative || '-';
    if (truePositiveElem) truePositiveElem.textContent = matrix.true_positive || '-';
}

function saveMetricsToSession(metrics) {
    sessionStorage.setItem(METRICS_STORAGE_KEY, JSON.stringify(metrics));
}

function loadMetricsFromSession() {
    const metricsData = sessionStorage.getItem(METRICS_STORAGE_KEY);
    if (metricsData) {
        return JSON.parse(metricsData);
    }
    return null;
}

function displayRetrainedMetrics(metrics) {
    if (!metrics) return;
    
    if (newAccuracyElem) newAccuracyElem.textContent = formatMetric(metrics.accuracy);
    if (newPrecisionElem) newPrecisionElem.textContent = formatMetric(metrics.precision);
    if (newRecallElem) newRecallElem.textContent = formatMetric(metrics.recall);
    if (newF1ScoreElem) newF1ScoreElem.textContent = formatMetric(metrics.f1);
    
    if (metrics.confusion_matrix) {
        updateConfusionMatrix(metrics.confusion_matrix);
    }
    
    if (saveBtn) saveBtn.disabled = false;
}

function loadCurrentMetrics() {
    if (accuracyElem) accuracyElem.textContent = formatMetric(staticMetrics.accuracy);
    if (precisionElem) precisionElem.textContent = formatMetric(staticMetrics.precision);
    if (recallElem) recallElem.textContent = formatMetric(staticMetrics.recall);
    if (f1ScoreElem) f1ScoreElem.textContent = formatMetric(staticMetrics.f1);
    
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

uploadForm.addEventListener("submit", async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById("dataset-upload");
    
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
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
            let errorDetail = "Upload failed";
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorDetail;
            } catch (e) {
                console.error("Error parsing error response:", e);
            }
            throw new Error(errorDetail);
        }
        
        const result = await response.json();
        updateStatus(`Successfully uploaded ${result.records_added} records. ${result.invalid_records} records were invalid.`, false, true);
        setUploadState(true);
        
    } catch (error) {
        console.error("Upload error:", error);
        updateStatus(`Upload failed: ${error.message}`, true);
    } finally {
        toggleSpinner('upload-spinner', false);
    }
});

retrainBtn.addEventListener("click", async function() {
    toggleSpinner('retrain-spinner', true);
    updateStatus("Retraining model... This may take a few moments.");
    
    try {
        const response = await fetch(`${API_BASE_URL}/retrain/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });
        
        if (!response.ok) {
            let errorDetail = "Retraining failed";
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorDetail;
            } catch (e) {
                console.error("Error parsing error response:", e);
            }
            throw new Error(errorDetail);
        }
        
        const result = await response.json();
        currentModelId = result.model_id;
        
        saveMetricsToSession({
            accuracy: result.metrics.accuracy,
            precision: result.metrics.precision,
            recall: result.metrics.recall,
            f1: result.metrics.f1,
            confusion_matrix: result.metrics.confusion_matrix
        });
        
        displayRetrainedMetrics({
            accuracy: result.metrics.accuracy,
            precision: result.metrics.precision,
            recall: result.metrics.recall,
            f1: result.metrics.f1,
            confusion_matrix: result.metrics.confusion_matrix
        });
        
        updateStatus(result.message || "Model retrained successfully!", false, true);
        
    } catch (error) {
        console.error("Retrain error:", error);
        updateStatus(`Retraining failed: ${error.message}`, true);
    } finally {
        toggleSpinner('retrain-spinner', false);
    }
});

saveBtn.addEventListener("click", async function() {
    if (!currentModelId) {
        updateStatus("No model to save. Please retrain first.", true);
        return;
    }
    
    const saveStatus = document.getElementById("save-status");
    if (saveStatus) {
        saveStatus.textContent = "Saving model...";
        saveStatus.style.color = "inherit";
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/save-model/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model_id: currentModelId
            })
        });
        
        if (!response.ok) {
            let errorDetail = "Save failed";
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorDetail;
            } catch (e) {
                console.error("Error parsing error response:", e);
            }
            throw new Error(errorDetail);
        }
        
        const result = await response.json();
        
        staticMetrics.accuracy = parseFloat(newAccuracyElem.textContent) || staticMetrics.accuracy;
        staticMetrics.precision = parseFloat(newPrecisionElem.textContent) || staticMetrics.precision;
        staticMetrics.recall = parseFloat(newRecallElem.textContent) || staticMetrics.recall;
        staticMetrics.f1 = parseFloat(newF1ScoreElem.textContent) || staticMetrics.f1;
        
        if (accuracyElem) accuracyElem.textContent = formatMetric(staticMetrics.accuracy);
        if (precisionElem) precisionElem.textContent = formatMetric(staticMetrics.precision);
        if (recallElem) recallElem.textContent = formatMetric(staticMetrics.recall);
        if (f1ScoreElem) f1ScoreElem.textContent = formatMetric(staticMetrics.f1);
        
        sessionStorage.removeItem(METRICS_STORAGE_KEY);
        
        if (saveStatus) {
            saveStatus.textContent = result.message || "Model saved successfully!";
            saveStatus.style.color = "#2e7d32";
        }
        updateStatus("Model saved successfully! Current metrics updated.", false, true);
        
    } catch (error) {
        console.error("Save error:", error);
        if (saveStatus) {
            saveStatus.textContent = `Save failed: ${error.message}`;
            saveStatus.style.color = "#c62828";
        }
    }
});

document.addEventListener('DOMContentLoaded', () => {
    loadCurrentMetrics();
});