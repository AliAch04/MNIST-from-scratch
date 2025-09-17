console.log('✅ script.js loaded!');

// Global variables for model weights
let W1, b1, W2, b2, W3, b3;
let modelInitialized = false;

// Debug function to check if weights are loaded
function checkWeightsLoaded() {
    console.log('=== DEBUG: Checking Weights ===');
    console.log('window.modelWeights exists:', typeof window.modelWeights !== 'undefined');
    
    if (window.modelWeights) {
        console.log('W1 sample:', window.modelWeights.W1?.slice(0, 2)?.map(row => row?.slice(0, 3)));
        console.log('b1 sample:', window.modelWeights.b1?.slice(0, 3));
    }
    
    console.log('numpy loaded:', typeof window.numpy !== 'undefined');
    console.log('modelInitialized:', modelInitialized);
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
    
    console.log('Grid data sample:', pixelData.slice(0, 10));
    return pixelData;
}

// Preprocess grid data
function preprocessGridData() {
    try {
        const pixelData = getGridData();
        const np = window.numpy;
        const imageArray = np.array(pixelData).reshape(28, 28);
        const flattened = imageArray.flatten();
        const result = flattened.reshape(784, 1);
        
        console.log('Preprocessed data shape:', result.shape);
        return result;
        
    } catch (error) {
        console.error('Preprocessing error:', error);
        throw error;
    }
}

// Predict digit function
function predictDigit() {
    console.log('🎯 Predict button clicked!');
    checkWeightsLoaded(); // Debug info
    
    if (!modelInitialized) {
        alert('Model is not initialized. Please check console for errors.');
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
    
    // Initialize model when everything is ready
    setTimeout(initializeModel, 1000);
});

// Make functions available globally
window.predictDigit = predictDigit;
window.clearGrid = clearGrid;
window.closeResult = closeResult;
window.checkWeightsLoaded = checkWeightsLoaded;

console.log('✅ Script loaded successfully!');