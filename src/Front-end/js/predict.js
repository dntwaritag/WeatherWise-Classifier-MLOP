document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const singlePredictionForm = document.getElementById('singlePredictionForm');
    const bulkPredictionForm = document.getElementById('bulkPredictionForm');
    const fileInput = document.getElementById('dataset');
    const fileLabel = document.querySelector('.file-label');
    const fileNameDisplay = document.querySelector('.file-name');
    const resultContainer = document.getElementById('singlePredictionResult');
    const bulkResultContainer = document.getElementById('bulkPredictionResult');
    const spinner = document.getElementById('spinner');
    const bulkSpinner = document.getElementById('bulkPredictionSpinner');
    const API_BASE_URL = 'https://your-api-url.com/api';

    // Weather feature fields
    const weatherFields = [
        'precipitation',
        'temp_max',
        'temp_min',
        'wind',
        'lag_wind_1',
        'lag_precipitation_1',
        'lag_temp_max_1',
        'lag_temp_min_1'
    ];

    // Initialize form fields
    function initFormFields() {
        const formGrid = document.getElementById('formGridContainer');
        formGrid.innerHTML = '';

        weatherFields.forEach(field => {
            const div = document.createElement('div');
            div.className = 'form-item';

            const label = document.createElement('label');
            label.htmlFor = field;
            label.textContent = field.replace(/_/g, ' ') + ':';

            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.id = field;
            input.name = field;
            input.required = true;

            div.appendChild(label);
            div.appendChild(input);
            formGrid.appendChild(div);
        });
    }

    // Toggle spinner
    function toggleSpinner(element, show) {
        element.style.display = show ? 'block' : 'none';
    }

    // Format weather type
    function formatWeatherType(type) {
        const types = {
            'rain': 'Rainy',
            'sun': 'Sunny',
            'fog': 'Foggy',
            'drizzle': 'Drizzly',
            'snow': 'Snowy'
        };
        return types[type.toLowerCase()] || type;
    }

    // Show prediction result
    function showPredictionResult(prediction) {
        const weatherType = formatWeatherType(prediction.weather);
        const confidencePercent = Math.round(prediction.confidence * 100);
        
        // Set weather icon
        const weatherIcon = document.getElementById('weather-icon');
        weatherIcon.className = 'weather-icon';
        
        if (weatherType.toLowerCase().includes('rain')) {
            weatherIcon.classList.add('weather-rainy');
            weatherIcon.innerHTML = '<i class="fas fa-cloud-rain"></i>';
        } else if (weatherType.toLowerCase().includes('sun')) {
            weatherIcon.classList.add('weather-sunny');
            weatherIcon.innerHTML = '<i class="fas fa-sun"></i>';
        } else if (weatherType.toLowerCase().includes('fog')) {
            weatherIcon.classList.add('weather-foggy');
            weatherIcon.innerHTML = '<i class="fas fa-smog"></i>';
        } else if (weatherType.toLowerCase().includes('snow')) {
            weatherIcon.classList.add('weather-snowy');
            weatherIcon.innerHTML = '<i class="fas fa-snowflake"></i>';
        } else {
            weatherIcon.classList.add('weather-cloudy');
            weatherIcon.innerHTML = '<i class="fas fa-cloud"></i>';
        }
        
        // Update confidence meter
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = `${confidencePercent}%`;
        
        // Update text
        document.getElementById('weather-type').textContent = weatherType;
        document.getElementById('confidence-percent').textContent = `${confidencePercent}%`;
        document.getElementById('model-version').textContent = prediction.model_version;
        
        // Show probabilities
        const probabilitiesList = document.getElementById('probabilities-list');
        probabilitiesList.innerHTML = '';
        
        prediction.probabilities.forEach(prob => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${formatWeatherType(prob.weather)}:</span>
                <span>${Math.round(prob.probability * 100)}%</span>
            `;
            probabilitiesList.appendChild(li);
        });
        
        // Show result
        resultContainer.style.display = 'block';
    }

    // Handle single prediction form submission
    singlePredictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        toggleSpinner(spinner, true);
        resultContainer.style.display = 'none';
        
        try {
            // Collect form data
            const formData = {};
            weatherFields.forEach(field => {
                formData[field] = parseFloat(document.getElementById(field).value);
            });
            
            // Send prediction request
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const result = await response.json();
            showPredictionResult(result);
        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Prediction failed: ${error.message}`);
        } finally {
            toggleSpinner(spinner, false);
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileNameDisplay.textContent = this.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file selected';
        }
    });

    // Handle bulk prediction form submission
    bulkPredictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('Please select a file first');
            return;
        }
        
        toggleSpinner(bulkSpinner, true);
        bulkResultContainer.style.display = 'none';
        bulkResultContainer.innerHTML = '';
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const response = await fetch(`${API_BASE_URL}/predict/bulk`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const results = await response.json();
            
            // Create summary
            const summary = document.createElement('div');
            summary.className = 'bulk-summary';
            
            const totalPredictions = results.length;
            const predictionCounts = {
                'Rainy': 0,
                'Sunny': 0,
                'Foggy': 0,
                'Drizzly': 0,
                'Snowy': 0
            };
            
            results.forEach(prediction => {
                const weatherType = formatWeatherType(prediction.weather);
                predictionCounts[weatherType]++;
            });
            
            summary.innerHTML = `
                <h3>Batch Prediction Summary</h3>
                <p>Processed ${totalPredictions} records with the following distribution:</p>
                <div class="summary-stats">
                    ${Object.entries(predictionCounts).map(([type, count]) => `
                        <div class="stat-card">
                            <div class="value">${count}</div>
                            <div class="label">${type}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            bulkResultContainer.appendChild(summary);
            
            // Create table
            const table = document.createElement('table');
            table.className = 'results-table';
            
            // Table header
            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr>
                    <th>Record</th>
                    <th>Weather</th>
                    <th>Confidence</th>
                    <th>Details</th>
                </tr>
            `;
            table.appendChild(thead);
            
            // Table body
            const tbody = document.createElement('tbody');
            results.forEach((result, index) => {
                const weatherType = formatWeatherType(result.weather);
                const confidencePercent = Math.round(result.confidence * 100);
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>
                        <span class="weather-badge badge-${weatherType.toLowerCase()}">
                            ${weatherType}
                        </span>
                    </td>
                    <td>${confidencePercent}%</td>
                    <td>
                        <button class="details-btn" data-index="${index}">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            bulkResultContainer.appendChild(table);
            
            bulkResultContainer.style.display = 'block';
        } catch (error) {
            console.error('Bulk prediction error:', error);
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = `Error: ${error.message}`;
            bulkResultContainer.appendChild(errorDiv);
            bulkResultContainer.style.display = 'block';
        } finally {
            toggleSpinner(bulkSpinner, false);
        }
    });

    // Initialize the form
    initFormFields();
});