// script.js - Fixed complete version
console.log('✅ script.js loaded!');

// Global variables for model weights
let W1, b1, W2, b2, W3, b3;
let modelInitialized = false;

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
    try {
        const pixelData = getGridData();
        const np = window.numpy;
        const imageArray = np.array(pixelData).reshape(28, 28);
        const flattened = imageArray.flatten();
        return flattened.reshape(784, 1);
        
    } catch (error) {
        console.error('Preprocessing error:', error);
        throw error;
    }
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
    console.log('🧹 Grid cleared');
}

// Debug function to check if weights are loaded
function checkWeightsLoaded() {
    console.log('=== DEBUG: Checking Weights ===');
    console.log('window.modelWeights exists:', typeof window.modelWeights !== 'undefined');
    console.log('numpy loaded:', typeof window.numpy !== 'undefined');
    console.log('modelInitialized:', modelInitialized);
    
    if (window.modelWeights) {
        console.log('W1 sample:', window.modelWeights.W1?.slice(0, 2)?.map(row => row?.slice(0, 3)));
    }
}

// Initialize the model
function initializeModel() {
    try {
        console.log('🔄 Initializing model...');
        
        const np = window.numpy;
        if (!np) {
            throw new Error('numpy not loaded');
        }
        
        // Check if weights are available
        if (!window.modelWeights) {
            throw new Error('Weights not found in window.modelWeights');
        }
        
        const weights = window.modelWeights;
        
        // Convert to numpy arrays
        W1 = np.array(weights.W1);
        b1 = np.array(weights.b1);
        W2 = np.array(weights.W2);
        b2 = np.array(weights.b2);
        W3 = np.array(weights.W3);
        b3 = np.array(weights.b3);
        
        modelInitialized = true;
        console.log('✅ Model loaded successfully!');
        console.log('W1 shape:', W1.shape);
        console.log('W2 shape:', W2.shape);
        console.log('W3 shape:', W3.shape);
        
    } catch (error) {
        console.error('❌ Model loading failed:', error);
        alert('Error: Could not load model weights. Check console for details.');
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

// Predict digit function
function predictDigit() {
    console.log('🎯 Predict button clicked!');
    
    if (!modelInitialized) {
        alert('Model is not initialized. Please check console for errors.');
        checkWeightsLoaded(); // Show debug info
        return;
    }
    
    // Check if any drawing exists
    const grid = document.getElementById("grid");
    const hasDrawing = Array.from(grid.children).some(cell => 
        parseInt(cell.dataset.intensity) > 0
    );
    
    if (!hasDrawing) {
        alert('Please draw a digit first!');
        return;
    }
    
    try {
        console.log('🧠 Starting prediction...');
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
        
        console.log('📊 Prediction:', prediction, 'Confidence:', confidence);
        
        // Show results
        showPredictionResult(prediction, confidence, probabilities);
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        alert('Prediction failed: ' + error.message);
    }
}

// Initialize when page loads
window.addEventListener('load', function() {
    console.log('📄 Page loaded, initializing...');
    
    // Add event listeners to buttons
    document.getElementById('predictBtn').addEventListener('click', predictDigit);
    document.getElementById('clearBtn').addEventListener('click', clearGrid);
    
    // Add debug button
    const debugBtn = document.createElement('button');
    debugBtn.textContent = 'Debug Weights';
    debugBtn.style.cssText = 'padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 6px; cursor: pointer; margin-left: 10px;';
    debugBtn.onclick = checkWeightsLoaded;
    document.querySelector('.buttons').appendChild(debugBtn);
    
    // Check if numpy is loaded
    function checkNumpy() {
        if (window.numpy) {
            console.log('✅ numpy loaded successfully');
            initializeModel();
        } else {
            console.log('⏳ Waiting for numpy...');
            setTimeout(checkNumpy, 100);
        }
    }
    
    // Start checking for numpy
    setTimeout(checkNumpy, 1000);
});

// Make functions available globally
window.predictDigit = predictDigit;
window.clearGrid = clearGrid;
window.closeResult = closeResult;
window.checkWeightsLoaded = checkWeightsLoaded;