document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const modelCards = document.querySelectorAll('.model-card');
    const startTrainingBtn = document.getElementById('start-training');
    const cancelTrainingBtn = document.getElementById('cancel-training');
    const trainingProgress = document.getElementById('training-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const saveModelBtn = document.getElementById('save-model');
    const saveStatus = document.getElementById('save-status');
    const statusMessage = document.getElementById('status-message');
    const comparisonChartCtx = document.getElementById('comparison-chart');
    const API_BASE_URL = 'https://your-api-url.com/api';

    // State
    let selectedFile = null;
    let selectedModel = 'random_forest';
    let trainingInProgress = false;
    let trainingInterval = null;
    let currentMetrics = {
        accuracy: 0.923,
        precision: 0.891,
        recall: 0.856,
        f1: 0.873
    };
    let newMetrics = null;
    let comparisonChart = null;

    // Initialize current metrics display
    function initCurrentMetrics() {
        document.getElementById('current-accuracy').textContent = (currentMetrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('current-precision').textContent = (currentMetrics.precision * 100).toFixed(1) + '%';
        document.getElementById('current-recall').textContent = (currentMetrics.recall * 100).toFixed(1) + '%';
        document.getElementById('current-f1').textContent = (currentMetrics.f1 * 100).toFixed(1) + '%';
    }

    // Handle file selection
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            selectedFile = this.files[0];
            fileInfo.textContent = selectedFile.name;
            startTrainingBtn.disabled = false;
            updateStatus('File ready for training. Select model and click "Start Training"', 'info');
        } else {
            selectedFile = null;
            fileInfo.textContent = 'No file selected';
            startTrainingBtn.disabled = true;
        }
    });

    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = '#3498db';
        this.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
    });

    uploadArea.addEventListener('dragleave', function() {
        this.style.borderColor = '#95a5a6';
        this.style.backgroundColor = 'white';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = '#95a5a6';
        this.style.backgroundColor = 'white';
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            selectedFile = e.dataTransfer.files[0];
            fileInfo.textContent = selectedFile.name;
            startTrainingBtn.disabled = false;
            updateStatus('File ready for training. Select model and click "Start Training"', 'info');
        }
    });

    // Handle model selection
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            modelCards.forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedModel = this.dataset.model;
        });
    });

    // Update status message
    function updateStatus(message, type = 'info') {
        statusMessage.className = 'status-message';
        statusMessage.classList.add(`status-${type}`);
        statusMessage.innerHTML = `<p>${message}</p>`;
    }

    // Simulate training progress
    function simulateTraining() {
        let progress = 0;
        trainingInProgress = true;
        startTrainingBtn.disabled = true;
        cancelTrainingBtn.disabled = false;
        trainingProgress.style.display = 'block';
        
        trainingInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 100) progress = 100;
            
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Training: ${Math.round(progress)}%`;
            
            if (progress === 100) {
                completeTraining();
            }
        }, 800);
    }

    // Complete training
    function completeTraining() {
        clearInterval(trainingInterval);
        trainingInProgress = false;
        cancelTrainingBtn.disabled = true;
        
        // Generate random metrics (in a real app, this would come from the API)
        newMetrics = {
            accuracy: currentMetrics.accuracy + (Math.random() * 0.1 - 0.03),
            precision: currentMetrics.precision + (Math.random() * 0.1 - 0.03),
            recall: currentMetrics.recall + (Math.random() * 0.1 - 0.03),
            f1: currentMetrics.f1 + (Math.random() * 0.1 - 0.03)
        };
        
        // Ensure metrics are within bounds
        Object.keys(newMetrics).forEach(key => {
            newMetrics[key] = Math.min(Math.max(newMetrics[key], 0.7), 0.99);
        });
        
        // Update UI
        document.getElementById('new-accuracy').textContent = (newMetrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('new-precision').textContent = (newMetrics.precision * 100).toFixed(1) + '%';
        document.getElementById('new-recall').textContent = (newMetrics.recall * 100).toFixed(1) + '%';
        document.getElementById('new-f1').textContent = (newMetrics.f1 * 100).toFixed(1) + '%';
        
        // Update comparison chart
        updateComparisonChart();
        
        // Enable save button
        saveModelBtn.disabled = false;
        
        updateStatus('Training completed successfully! Review metrics and save the new model if desired.', 'success');
    }

    // Update comparison chart
    function updateComparisonChart() {
        if (comparisonChart) {
            comparisonChart.destroy();
        }
        
        comparisonChart = new Chart(comparisonChartCtx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                datasets: [
                    {
                        label: 'Current Model',
                        data: [
                            currentMetrics.accuracy,
                            currentMetrics.precision,
                            currentMetrics.recall,
                            currentMetrics.f1
                        ],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'New Model',
                        data: [
                            newMetrics.accuracy,
                            newMetrics.precision,
                            newMetrics.recall,
                            newMetrics.f1
                        ],
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(46, 204, 113, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + (context.raw * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Start training
    startTrainingBtn.addEventListener('click', function() {
        if (!selectedFile) {
            updateStatus('Please select a file first', 'error');
            return;
        }
        
        updateStatus('Training in progress... This may take several minutes', 'info');
        simulateTraining();
    });

    // Cancel training
    cancelTrainingBtn.addEventListener('click', function() {
        clearInterval(trainingInterval);
        trainingInProgress = false;
        startTrainingBtn.disabled = false;
        cancelTrainingBtn.disabled = true;
        trainingProgress.style.display = 'none';
        progressFill.style.width = '0%';
        progressText.textContent = 'Training: 0%';
        
        updateStatus('Training canceled. You can start a new training session.', 'error');
    });

    // Save model
    saveModelBtn.addEventListener('click', async function() {
        if (!newMetrics) {
            updateStatus('No new model to save', 'error');
            return;
        }
        
        saveStatus.textContent = 'Saving model...';
        saveStatus.className = 'save-status';
        
        try {
            // In a real app, you would send a request to your API
            // const response = await fetch(`${API_BASE_URL}/models/save`, {
            //     method: 'POST',
            //     headers: {
            //         'Content-Type': 'application/json'
            //     },
            //     body: JSON.stringify({
            //         model_type: selectedModel,
            //         metrics: newMetrics
            //     })
            // });
            
            // Simulate API delay
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Update current metrics
            currentMetrics = {...newMetrics};
            initCurrentMetrics();
            
            // Reset UI
            newMetrics = null;
            saveModelBtn.disabled = true;
            saveStatus.textContent = 'Model saved successfully!';
            saveStatus.className = 'save-status save-success';
            
            updateStatus('New model saved and now in use!', 'success');
        } catch (error) {
            console.error('Save error:', error);
            saveStatus.textContent = 'Error saving model: ' + error.message;
            saveStatus.className = 'save-status save-error';
        }
    });

    // Initialize
    initCurrentMetrics();
    updateComparisonChart();
});