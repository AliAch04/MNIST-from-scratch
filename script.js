const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const predictionDiv = document.getElementById('prediction');
const confidenceDiv = document.getElementById('confidence');
const probabilitiesDiv = document.getElementById('probabilities');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 20;
ctx.lineCap = 'round';

// Drawing functions
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', predictDigit);

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.style.display = 'none';
}

// Preprocess image for the neural network
function preprocessImage() {
    // Create a temporary canvas to resize to 28x28
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    
    // Draw and resize
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convert to grayscale and normalize
    const processedData = [];
    for (let i = 0; i < data.length; i += 4) {
        // Extract RGB and convert to grayscale (invert since canvas is black background)
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const grayscale = 255 - (0.299 * r + 0.587 * g + 0.114 * b); // Invert for black background
    
        processedData.push(grayscale / 255.0);
    }
    
    return processedData;
}

// Predict digit using your neural network
async function predictDigit() {
    try {
        // Preprocess the drawn image
        const imageData = preprocessImage();
        
        // Convert to numpy array (using browser numpy)
        const np = window.numpy;
        const inputArray = np.array(imageData).reshape(28, 28);
        
        // Flatten and add batch dimension
        const flattened = inputArray.flatten();
        const X = flattened.reshape(784, 1);
        
        // Normalize (if not already done in preprocessing)
        const X_normalized = X.divide(255.0);
        
        // Run through your neural network functions
        const Z1 = W1.dot(X_normalized).add(b1);
        const A1 = relu(Z1);
        
        const Z2 = W2.dot(A1).add(b2);
        const A2 = relu(Z2);
        
        const Z3 = W3.dot(A2).add(b3);
        const A3 = softmax(Z3);
        
        // Get prediction
        const prediction = np.argmax(A3, 0).get(0);
        const confidence = np.max(A3).get(0);
        const probabilities = A3.flatten().tolist();
        
        // Display results
        displayResults(prediction, confidence, probabilities);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please check the console.');
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

// Display prediction results
function displayResults(prediction, confidence, probabilities) {
    predictionDiv.textContent = prediction;
    confidenceDiv.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    
    // Create probability bars
    probabilitiesDiv.innerHTML = '';
    probabilities.forEach((prob, digit) => {
        const barContainer = document.createElement('div');
        barContainer.style.display = 'flex';
        barContainer.style.flexDirection = 'column';
        barContainer.style.alignItems = 'center';
        
        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.height = `${prob * 100}px`;
        bar.style.backgroundColor = digit === prediction ? '#2e7d32' : '#4CAF50';
        bar.style.opacity = digit === prediction ? '1' : '0.6';
        
        const label = document.createElement('div');
        label.className = 'prob-label';
        label.textContent = digit;
        
        barContainer.appendChild(bar);
        barContainer.appendChild(label);
        probabilitiesDiv.appendChild(barContainer);
    });
    
    resultDiv.style.display = 'block';
}

// Load your trained model weights (you'll need to replace these with your actual weights)
// In a real application, you'd load these from a file or server
const W1 = /* Your W1 numpy array */;
const b1 = /* Your b1 numpy array */;
const W2 = /* Your W2 numpy array */;
const W3 = /* Your W3 numpy array */;
const b2 = /* Your b2 numpy array */;
const b3 = /* Your b3 numpy array */;

// For testing purposes, you can use random weights initially
// Replace these with your actual trained weights
function initializeDemoWeights() {
    const np = window.numpy;
    
    // Demo weights - replace with your actual trained weights
    W1 = np.random.rand(128, 784).multiply(0.01);
    b1 = np.zeros([128, 1]);
    
    W2 = np.random.rand(64, 128).multiply(0.01);
    b2 = np.zeros([64, 1]);
    
    W3 = np.random.rand(10, 64).multiply(0.01);
    b3 = np.zeros([10, 1]);
    
    console.log('Demo weights initialized. Replace with your actual trained weights.');
}

// Initialize when numpy is loaded
if (window.numpy) {
    initializeDemoWeights();
} else {
    window.addEventListener('load', initializeDemoWeights);
}