// X-ray Detection System JavaScript

// Global variables
let currentSection = 'dashboard';
let trainingInterval = null;
let trainingData = {
    epochs: [],
    losses: [],
    maps: []
};
let charts = {};

// Class mappings from the Python code
const CLASS_MAPPING = {
    1: 'firearm',
    2: 'knife', 
    3: 'explosive',
    4: 'drug_package',
    5: 'contraband_electronics',
    6: 'liquid_battery',
    7: 'prohibited_item_misc'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeDashboard();
    initializeFileUpload();
    startRealTimeUpdates();
});

// Section Navigation
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show selected section
    document.getElementById(sectionName + '-section').style.display = 'block';
    
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');
    
    currentSection = sectionName;
    
    // Initialize section-specific functionality
    if (sectionName === 'dataset') {
        initializeDatasetCharts();
    }
}

// Dashboard Initialization
function initializeDashboard() {
    populateAlertsTable();
    updateDashboardStats();
}

function updateDashboardStats() {
    // Simulate real-time stats updates
    const stats = {
        scansToday: Math.floor(Math.random() * 100) + 1200,
        threatsDetected: Math.floor(Math.random() * 10) + 20,
        modelAccuracy: (Math.random() * 5 + 92).toFixed(1),
        avgLatency: Math.floor(Math.random() * 100) + 300
    };
    
    document.getElementById('scans-today').textContent = stats.scansToday.toLocaleString();
    document.getElementById('threats-detected').textContent = stats.threatsDetected;
    document.getElementById('model-accuracy').textContent = stats.modelAccuracy + '%';
    document.getElementById('avg-latency').textContent = stats.avgLatency + 'ms';
}

function populateAlertsTable() {
    const alertsData = [
        {
            time: '14:32:15',
            threat: 'Firearm',
            confidence: '96.8%',
            location: 'Terminal A - Gate 12',
            status: 'Under Review',
            level: 'critical'
        },
        {
            time: '14:28:42',
            threat: 'Knife',
            confidence: '89.2%',
            location: 'Terminal B - Security',
            status: 'Resolved',
            level: 'high'
        },
        {
            time: '14:15:33',
            threat: 'Liquid Battery',
            confidence: '78.5%',
            location: 'Terminal C - Gate 8',
            status: 'False Positive',
            level: 'medium'
        },
        {
            time: '14:02:18',
            threat: 'Contraband Electronics',
            confidence: '92.1%',
            location: 'Terminal A - Security',
            status: 'Confiscated',
            level: 'high'
        }
    ];
    
    const tbody = document.getElementById('alerts-table');
    tbody.innerHTML = '';
    
    alertsData.forEach(alert => {
        const row = document.createElement('tr');
        row.className = `alert-${alert.level}`;
        row.innerHTML = `
            <td>${alert.time}</td>
            <td><span class="threat-level-${alert.level}">${alert.threat}</span></td>
            <td>${alert.confidence}</td>
            <td>${alert.location}</td>
            <td><span class="badge bg-${getStatusColor(alert.status)}">${alert.status}</span></td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="viewAlert('${alert.time}')">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="exportAlert('${alert.time}')">
                    <i class="fas fa-download"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function getStatusColor(status) {
    const colors = {
        'Under Review': 'warning',
        'Resolved': 'success',
        'False Positive': 'secondary',
        'Confiscated': 'danger'
    };
    return colors[status] || 'primary';
}

// Chart Initialization
function initializeCharts() {
    // Activity Chart
    const activityCtx = document.getElementById('activityChart').getContext('2d');
    charts.activity = new Chart(activityCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(24),
            datasets: [{
                label: 'Scans per Hour',
                data: generateRandomData(24, 30, 80),
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Threats Detected',
                data: generateRandomData(24, 0, 5),
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Threat Distribution Chart
    const threatCtx = document.getElementById('threatChart').getContext('2d');
    charts.threat = new Chart(threatCtx, {
        type: 'doughnut',
        data: {
            labels: Object.values(CLASS_MAPPING),
            datasets: [{
                data: [15, 12, 8, 10, 6, 4, 3],
                backgroundColor: [
                    '#dc3545', '#fd7e14', '#6f42c1', '#20c997',
                    '#0dcaf0', '#ffc107', '#6c757d'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function initializeDatasetCharts() {
    // Class Distribution Chart
    const classCtx = document.getElementById('classDistChart');
    if (classCtx && !charts.classDist) {
        charts.classDist = new Chart(classCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: Object.values(CLASS_MAPPING),
                datasets: [{
                    label: 'Number of Annotations',
                    data: [3245, 2876, 1987, 2134, 1654, 1432, 987],
                    backgroundColor: [
                        '#dc3545', '#fd7e14', '#6f42c1', '#20c997',
                        '#0dcaf0', '#ffc107', '#6c757d'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// File Upload Functionality
function initializeFileUpload() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: { files: files } });
        }
    });
    
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            displayImagePreview(e.target.result);
            document.getElementById('analyze-btn').disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

function displayImagePreview(imageSrc) {
    const previewContainer = document.getElementById('image-preview');
    previewContainer.innerHTML = `<img src="${imageSrc}" class="image-preview" alt="X-ray Preview">`;
}

function clearImage() {
    document.getElementById('file-input').value = '';
    document.getElementById('image-preview').innerHTML = '<p class="text-muted">No image selected</p>';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('detection-results').style.display = 'none';
}

// Image Analysis
function analyzeImage() {
    showLoading('Analyzing X-ray image...');
    
    // Simulate analysis delay
    setTimeout(() => {
        hideLoading();
        displayDetectionResults();
    }, 3000);
}

function displayDetectionResults() {
    // Simulate detection results
    const detections = [
        {
            class: 'firearm',
            confidence: 0.968,
            bbox: [150, 200, 120, 80],
            threat_level: 'critical'
        },
        {
            class: 'knife',
            confidence: 0.834,
            bbox: [300, 150, 60, 40],
            threat_level: 'high'
        }
    ];
    
    const resultsContainer = document.getElementById('results-content');
    let resultsHtml = '';
    
    if (detections.length === 0) {
        resultsHtml = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                No threats detected. Image appears safe.
            </div>
        `;
    } else {
        resultsHtml = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>${detections.length} threat(s) detected!</strong>
            </div>
            <div class="row">
        `;
        
        detections.forEach((detection, index) => {
            resultsHtml += `
                <div class="col-md-6 mb-3">
                    <div class="card border-${getThreatColor(detection.threat_level)}">
                        <div class="card-header bg-${getThreatColor(detection.threat_level)} text-white">
                            <h6 class="mb-0">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                ${detection.class.replace('_', ' ').toUpperCase()}
                            </h6>
                        </div>
                        <div class="card-body">
                            <p><strong>Confidence:</strong> ${(detection.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Threat Level:</strong> 
                                <span class="badge bg-${getThreatColor(detection.threat_level)}">${detection.threat_level}</span>
                            </p>
                            <p><strong>Location:</strong> [${detection.bbox.join(', ')}]</p>
                            <div class="d-flex gap-2">
                                <button class="btn btn-sm btn-outline-primary" onclick="highlightDetection(${index})">
                                    <i class="fas fa-crosshairs me-1"></i>Highlight
                                </button>
                                <button class="btn btn-sm btn-outline-secondary" onclick="exportDetection(${index})">
                                    <i class="fas fa-download me-1"></i>Export
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        resultsHtml += '</div>';
    }
    
    resultsContainer.innerHTML = resultsHtml;
    document.getElementById('detection-results').style.display = 'block';
}

function getThreatColor(level) {
    const colors = {
        'critical': 'danger',
        'high': 'warning',
        'medium': 'info',
        'low': 'success'
    };
    return colors[level] || 'secondary';
}

// Training Functions
function startTraining() {
    const config = getTrainingConfig();
    
    showLoading('Initializing training...');
    
    setTimeout(() => {
        hideLoading();
        updateTrainingStatus('running');
        simulateTraining(config);
        showNotification('Training started successfully!', 'success');
    }, 2000);
}

function getTrainingConfig() {
    return {
        modelType: document.getElementById('model-type').value,
        batchSize: parseInt(document.getElementById('batch-size').value),
        learningRate: parseFloat(document.getElementById('learning-rate').value),
        epochs: parseInt(document.getElementById('epochs').value),
        optimizer: document.getElementById('optimizer').value,
        augmentation: document.getElementById('augmentation').value
    };
}

function simulateTraining(config) {
    let currentEpoch = 0;
    const totalEpochs = config.epochs;
    
    document.getElementById('total-epochs').textContent = totalEpochs;
    
    trainingInterval = setInterval(() => {
        currentEpoch++;
        
        // Simulate training metrics
        const loss = Math.max(0.1, 2.0 - (currentEpoch / totalEpochs) * 1.5 + Math.random() * 0.2);
        const map = Math.min(0.95, (currentEpoch / totalEpochs) * 0.9 + Math.random() * 0.05);
        
        // Update UI
        document.getElementById('current-epoch').textContent = currentEpoch;
        document.getElementById('current-loss').textContent = loss.toFixed(4);
        document.getElementById('current-map').textContent = (map * 100).toFixed(1) + '%';
        
        const progress = (currentEpoch / totalEpochs) * 100;
        document.getElementById('training-progress').style.width = progress + '%';
        
        // Calculate ETA
        const remainingEpochs = totalEpochs - currentEpoch;
        const etaMinutes = Math.ceil(remainingEpochs * 2); // 2 minutes per epoch
        document.getElementById('training-eta').textContent = etaMinutes + ' min';
        
        // Update charts
        updateTrainingCharts(currentEpoch, loss, map);
        
        // Check if training is complete
        if (currentEpoch >= totalEpochs) {
            completeTraining();
        }
    }, 2000); // Update every 2 seconds for demo
}

function updateTrainingCharts(epoch, loss, map) {
    trainingData.epochs.push(epoch);
    trainingData.losses.push(loss);
    trainingData.maps.push(map);
    
    // Initialize training charts if not exists
    if (!charts.loss) {
        initializeTrainingCharts();
    }
    
    // Update loss chart
    charts.loss.data.labels = trainingData.epochs;
    charts.loss.data.datasets[0].data = trainingData.losses;
    charts.loss.update();
    
    // Update mAP chart
    charts.map.data.labels = trainingData.epochs;
    charts.map.data.datasets[0].data = trainingData.maps;
    charts.map.update();
}

function initializeTrainingCharts() {
    // Loss Chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    charts.loss = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // mAP Chart
    const mapCtx = document.getElementById('mapChart').getContext('2d');
    charts.map = new Chart(mapCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Validation mAP',
                data: [],
                borderColor: '#198754',
                backgroundColor: 'rgba(25, 135, 84, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function pauseTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
        updateTrainingStatus('paused');
        showNotification('Training paused', 'warning');
    }
}

function stopTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    updateTrainingStatus('stopped');
    resetTrainingUI();
    showNotification('Training stopped', 'info');
}

function completeTraining() {
    clearInterval(trainingInterval);
    trainingInterval = null;
    updateTrainingStatus('completed');
    document.getElementById('training-eta').textContent = 'Complete';
    showNotification('Training completed successfully!', 'success');
}

function updateTrainingStatus(status) {
    const statusElement = document.getElementById('training-status');
    const pauseBtn = document.getElementById('pause-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    statusElement.className = 'badge';
    
    switch (status) {
        case 'running':
            statusElement.classList.add('bg-success');
            statusElement.textContent = 'Running';
            pauseBtn.disabled = false;
            stopBtn.disabled = false;
            break;
        case 'paused':
            statusElement.classList.add('bg-warning');
            statusElement.textContent = 'Paused';
            break;
        case 'stopped':
            statusElement.classList.add('bg-danger');
            statusElement.textContent = 'Stopped';
            pauseBtn.disabled = true;
            stopBtn.disabled = true;
            break;
        case 'completed':
            statusElement.classList.add('bg-primary');
            statusElement.textContent = 'Completed';
            pauseBtn.disabled = true;
            stopBtn.disabled = true;
            break;
        default:
            statusElement.classList.add('bg-secondary');
            statusElement.textContent = 'Idle';
            pauseBtn.disabled = true;
            stopBtn.disabled = true;
    }
}

function resetTrainingUI() {
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('current-loss').textContent = '-';
    document.getElementById('current-map').textContent = '-';
    document.getElementById('training-progress').style.width = '0%';
    document.getElementById('training-eta').textContent = '-';
    
    // Reset training data
    trainingData = { epochs: [], losses: [], maps: [] };
}

// Dataset Functions
function uploadDataset() {
    const uploadType = document.getElementById('upload-type').value;
    const files = document.getElementById('dataset-files').files;
    
    if (files.length === 0) {
        showNotification('Please select files to upload', 'warning');
        return;
    }
    
    showLoading(`Uploading ${uploadType} dataset...`);
    
    setTimeout(() => {
        hideLoading();
        showNotification(`Dataset uploaded successfully! ${files.length} files processed.`, 'success');
        updateDatasetStats();
    }, 3000);
}

function updateDatasetStats() {
    // Simulate updated stats
    const stats = {
        totalImages: Math.floor(Math.random() * 1000) + 15000,
        trainImages: Math.floor(Math.random() * 800) + 12000,
        valImages: Math.floor(Math.random() * 300) + 2000,
        testImages: Math.floor(Math.random() * 200) + 1000,
        totalAnnotations: Math.floor(Math.random() * 5000) + 20000,
        avgObjects: (Math.random() * 0.5 + 1.2).toFixed(2)
    };
    
    document.getElementById('total-images').textContent = stats.totalImages.toLocaleString();
    document.getElementById('train-images').textContent = stats.trainImages.toLocaleString();
    document.getElementById('val-images').textContent = stats.valImages.toLocaleString();
    document.getElementById('test-images').textContent = stats.testImages.toLocaleString();
    document.getElementById('total-annotations').textContent = stats.totalAnnotations.toLocaleString();
    document.getElementById('avg-objects').textContent = stats.avgObjects;
}

function previewAugmentation() {
    const augType = document.getElementById('aug-type').value;
    const previewContainer = document.getElementById('augmentation-preview');
    
    showLoading('Generating augmentation preview...');
    
    setTimeout(() => {
        hideLoading();
        
        const augmentations = [
            'Original', 'Horizontal Flip', 'Rotation', 'Brightness/Contrast',
            'Gaussian Noise', 'CLAHE', 'Coarse Dropout', 'Perspective'
        ];
        
        let previewHtml = '';
        augmentations.forEach(aug => {
            previewHtml += `
                <div class="aug-preview-item">
                    <div style="width: 100%; height: 120px; background: linear-gradient(45deg, #f0f0f0, #e0e0e0); border-radius: 4px; margin-bottom: 8px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-image fa-2x text-muted"></i>
                    </div>
                    <div class="aug-label">${aug}</div>
                </div>
            `;
        });
        
        previewContainer.innerHTML = previewHtml;
        showNotification(`${augType} augmentation preview generated`, 'success');
    }, 2000);
}

// Utility Functions
function generateTimeLabels(hours) {
    const labels = [];
    const now = new Date();
    for (let i = hours - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        labels.push(time.getHours().toString().padStart(2, '0') + ':00');
    }
    return labels;
}

function generateRandomData(count, min, max) {
    return Array.from({ length: count }, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

function showLoading(message = 'Loading...') {
    document.getElementById('loading-text').textContent = message;
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification show`;
    notification.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span>${message}</span>
            <button type="button" class="btn-close" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function startRealTimeUpdates() {
    // Update dashboard stats every 30 seconds
    setInterval(updateDashboardStats, 30000);
    
    // Update activity chart every minute
    setInterval(() => {
        if (currentSection === 'dashboard' && charts.activity) {
            // Add new data point and remove old one
            const newScans = Math.floor(Math.random() * 50) + 30;
            const newThreats = Math.floor(Math.random() * 3);
            
            charts.activity.data.datasets[0].data.push(newScans);
            charts.activity.data.datasets[1].data.push(newThreats);
            
            if (charts.activity.data.datasets[0].data.length > 24) {
                charts.activity.data.datasets[0].data.shift();
                charts.activity.data.datasets[1].data.shift();
                charts.activity.data.labels.shift();
            }
            
            const now = new Date();
            charts.activity.data.labels.push(now.getHours().toString().padStart(2, '0') + ':' + 
                                           now.getMinutes().toString().padStart(2, '0'));
            
            charts.activity.update();
        }
    }, 60000);
}

// Event handlers for buttons
function viewAlert(time) {
    showNotification(`Viewing alert from ${time}`, 'info');
}

function exportAlert(time) {
    showNotification(`Exporting alert from ${time}`, 'success');
}

function highlightDetection(index) {
    showNotification(`Highlighting detection ${index + 1}`, 'info');
}

function exportDetection(index) {
    showNotification(`Exporting detection ${index + 1}`, 'success');
}