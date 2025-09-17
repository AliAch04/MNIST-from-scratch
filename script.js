// script.js - COMPLETE numpy-free version
console.log('✅ script.js loaded without numpy!');

// Global variables for model weights
let W1, b1, W2, b2, W3, b3;
let modelInitialized = false;

// ==================== PURE JAVASCRIPT MATH FUNCTIONS ====================

// Matrix multiplication: A * B
function matMultiply(A, B) {
    const aRows = A.length, aCols = A[0].length;
    const bRows = B.length, bCols = B[0].length;
    
    if (aCols !== bRows) {
        throw new Error(`Matrix multiplication error: ${aCols} != ${bRows}`);
    }
    
    const result = new Array(aRows);
    for (let i = 0; i < aRows; i++) {
        result[i] = new Array(bCols).fill(0);
        for (let j = 0; j < bCols; j++) {
            for (let k = 0; k < aCols; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Matrix addition: A + B
function matAdd(A, B) {
    const rows = A.length, cols = A[0].length;
    const result = new Array(rows);
    
    for (let i = 0; i < rows; i++) {
        result[i] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// ReLU activation function
function relu(X) {
    const rows = X.length, cols = X[0].length;
    const result = new Array(rows);
    
    for (let i = 0; i < rows; i++) {
        result[i] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            result[i][j] = Math.max(0, X[i][j]);
        }
    }
    return result;
}

// Softmax activation function
function softmax(X) {
    const rows = X.length, cols = X[0].length;
    const result = new Array(rows);
    
    for (let j = 0; j < cols; j++) {
        // Find max value in this column for numerical stability
        let maxVal = -Infinity;
        for (let i = 0; i < rows; i++) {
            if (X[i][j] > maxVal) maxVal = X[i][j];
        }
        
        // Compute exponentials
        let sum = 0;
        const expValues = new Array(rows);
        for (let i = 0; i < rows; i++) {
            expValues[i] = Math.exp(X[i][j] - maxVal);
            sum += expValues[i];
        }
        
        // Normalize
        for (let i = 0; i < rows; i++) {
            if (!result[i]) result[i] = new Array(cols);
            result[i][j] = expValues[i] / sum;
        }
    }
    return result;
}

// Get maximum value and index from matrix
function argmax(matrix) {
    const rows = matrix.length, cols = matrix[0].length;
    const predictions = new Array(cols);
    const confidences = new Array(cols);
    
    for (let j = 0; j < cols; j++) {
        let maxVal = -Infinity;
        let maxIndex = -1;
        
        for (let i = 0; i < rows; i++) {
            if (matrix[i][j] > maxVal) {
                maxVal = matrix[i][j];
                maxIndex = i;
            }
        }
        
        predictions[j] = maxIndex;
        confidences[j] = maxVal;
    }
    
    return { predictions, confidences };
}

// ==================== NEURAL NETWORK FUNCTIONS ====================

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

// Preprocess grid data for neural network
function preprocessGridData() {
    const pixelData = getGridData();
    // Reshape to 28x28 and then to 784x1
    const image2D = [];
    for (let i = 0; i < 28; i++) {
        image2D.push(pixelData.slice(i * 28, (i + 1) * 28));
    }
    // Transpose to 784x1 (each pixel as a separate row)
    const result = [];
    for (let i = 0; i < 784; i++) {
        result.push([pixelData[i]]);
    }
    return result;
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

// Debug function
function checkWeightsLoaded() {
    console.log('=== DEBUG: Checking Weights ===');
    console.log('window.modelWeights exists:', typeof window.modelWeights !== 'undefined');
    console.log('modelInitialized:', modelInitialized);
    
    if (window.modelWeights) {
        console.log('W1 shape:', window.modelWeights.W1.length, 'x', window.modelWeights.W1[0].length);
        console.log('W1 sample:', window.modelWeights.W1[0].slice(0, 3));
    }
}

// Initialize the model
function initializeModel() {
    try {
        console.log('🔄 Initializing model...');
        
        // Check if weights are available
        if (!window.modelWeights) {
            throw new Error('Weights not found in window.modelWeights');
        }
        
        const weights = window.modelWeights;
        
        // Assign weights (they're already in correct format)
        W1 = weights.W1;
        b1 = weights.b1;
        W2 = weights.W2;
        b2 = weights.b2;
        W3 = weights.W3;
        b3 = weights.b3;
        
        modelInitialized = true;
        console.log('✅ Model loaded successfully!');
        console.log('W1 shape:', W1.length, 'x', W1[0].length);
        console.log('W2 shape:', W2.length, 'x', W2[0].length);
        console.log('W3 shape:', W3.length, 'x', W3[0].length);
        
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
                                  height: ${prob * 100}%; border-radius: 3px 3px 0 0;">
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
        checkWeightsLoaded();
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
        const X = preprocessGridData();
        
        // Forward propagation through the neural network
        console.log('➡️ Forward propagation...');
        const Z1 = matAdd(matMultiply(W1, X), b1);
        const A1 = relu(Z1);
        
        const Z2 = matAdd(matMultiply(W2, A1), b2);
        const A2 = relu(Z2);
        
        const Z3 = matAdd(matMultiply(W3, A2), b3);
        const A3 = softmax(Z3);
        
        // Get results
        const { predictions, confidences } = argmax(A3);
        const prediction = predictions[0]; // First (and only) prediction
        const confidence = confidences[0];
        const probabilities = A3.map(row => row[0]); // Flatten to 1D array
        
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
    
    // Initialize model immediately (no waiting for numpy)
    initializeModel();
});

// Make functions available globally
window.predictDigit = predictDigit;
window.clearGrid = clearGrid;
window.closeResult = closeResult;
window.checkWeightsLoaded = checkWeightsLoaded;

console.log('🚀 Neural network ready without numpy!');