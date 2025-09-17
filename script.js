// script.js - Neural Network Prediction

// Global variables for model weights
let W1, b1, W2, b2, W3, b3;
let modelInitialized = false;

// Initialize the model
function initializeModel() {
    try {
        const np = window.numpy;
        
        // Convert to numpy arrays
        W1 = np.array(W1);
        b1 = np.array(b1);
        W2 = np.array(W2);
        b2 = np.array(b2);
        W3 = np.array(W3);
        b3 = np.array(b3);
        
        modelInitialized = true;
        console.log('✅ Model loaded successfully!');
        
    } catch (error) {
        console.error('❌ Model loading failed:', error);
        alert('Error: Could not load model weights. Check console for details.');
    }
}

// Activation functions
function relu(X) {
    return np.maximum(0, X);
}

function softmax(Z) {
    const Z_shifted = Z.subtract(np.max(Z, 0, true));
    const exp_Z = np.exp(Z_shifted);
    return exp_Z.divide(np.sum(exp_Z, 0, true));
}

// Get pixel data from grid
function getGridData() {
    const grid = document.getElementById("grid");
    const cells = grid.children;
    const pixelData = [];
    
    for (let i = 0; i < cells.length; i++) {
        const intensity = parseInt(cells[i].dataset.intensity) || 0;
        const normalized = intensity / 255.0;
        pixelData.push(normalized);
    }
    
    return pixelData;
}

// Preprocess grid data
function preprocessGridData() {
    const pixelData = getGridData();
    const np = window.numpy;
    const imageArray = np.array(pixelData).reshape(28, 28);
    const flattened = imageArray.flatten();
    return flattened.reshape(784, 1);
}

// Predict digit function
function predictDigit() {
    if (!modelInitialized) {
        alert('Model is still loading. Please wait...');
        return;
    }
    
    try {
        const np = window.numpy;
        const X = preprocessGridData();
        
        // Forward propagation
        const Z1 = W1.dot(X).add(b1);
        const A1 = relu(Z1);
        
        const Z2 = W2.dot(A1).add(b2);
        const A2 = relu(Z2);
        
        const Z3 = W3.dot(A2).add(b3);
        const A3 = softmax(Z3);
        
        // Get results
        const prediction = np.argmax(A3, 0).get(0);
        const confidence = np.max(A3).get(0);
        const probabilities = A3.flatten().tolist();
        
        // Show results
        showPredictionResult(prediction, confidence, probabilities);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Prediction failed: ' + error.message);
    }
}

// Show prediction results
function showPredictionResult(prediction, confidence, probabilities) {
    // Remove existing results if any
    const existingResult = document.getElementById('predictionResult');
    if (existingResult) {
        existingResult.remove();
    }
    
    // Create results display
    const resultDiv = document.createElement('div');
    resultDiv.id = 'predictionResult';
    resultDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        z-index: 1000;
        text-align: center;
        min-width: 300px;
        border: 3px solid #4CAF50;
    `;
    
    resultDiv.innerHTML = `
        <h2 style="color: #333; margin: 0 0 20px 0;">Prediction Result</h2>
        <div style="font-size: 72px; font-weight: bold; color: #4CAF50; margin: 20px 0;">
            ${prediction}
        </div>
        <div style="color: #666; font-size: 18px; margin: 10px 0;">
            Confidence: ${(confidence * 100).toFixed(1)}%
        </div>
        <div style="margin: 20px 0;">
            <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Probability Distribution</div>
            <div style="display: flex; gap: 3px; height: 40px; align-items: flex-end; justify-content: center;">
                ${probabilities.map((prob, digit) => `
                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; max-width: 30px;">
                        <div style="width: 100%; background-color: ${digit === prediction ? '#4CAF50' : '#ccc'}; 
                                  height: ${prob * 100}%; border-radius: 3px 3px 0 0; transition: height 0.3s;">
                        </div>
                        <div style="font-size: 10px; margin-top: 5px; color: #666;">${digit}</div>
                    </div>
                `).join('')}
            </div>
        </div>
        <button onclick="closeResult()" style="padding: 12px 24px; background: #f44336; color: white; 
                border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 15px;">
            Close
        </button>
    `;
    
    document.body.appendChild(resultDiv);
}

// Close result window
function closeResult() {
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.remove();
    }
}

// Clear grid function
function clearGrid() {
    const grid = document.getElementById("grid");
    const cells = grid.children;
    
    for (let i = 0; i < cells.length; i++) {
        cells[i].dataset.intensity = "0";
        cells[i].style.background = "rgb(255, 255, 255)";
    }
    
    closeResult();
}

// Initialize when page loads
window.addEventListener('load', function() {
    // Add event listeners to buttons
    document.getElementById('predictBtn').addEventListener('click', predictDigit);
    document.getElementById('clearBtn').addEventListener('click', clearGrid);
    
    // Initialize model when numpy is ready
    if (window.numpy) {
        initializeModel();
    } else {
        const checkNumpy = setInterval(() => {
            if (window.numpy) {
                clearInterval(checkNumpy);
                initializeModel();
            }
        }, 100);
    }
});

// Make functions available globally
window.predictDigit = predictDigit;
window.clearGrid = clearGrid;
window.closeResult = closeResult;

console.log('✅ Script loaded successfully!');