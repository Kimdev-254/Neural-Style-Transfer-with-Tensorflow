let styleNet, transformNet;

const contentImage = document.getElementById('content-preview');
const styleImage = document.getElementById('style-preview');
const outputCanvas = document.getElementById('output-canvas');

// Load image previews
document.getElementById('content-image').addEventListener('change', (e) => {
  contentImage.src = URL.createObjectURL(e.target.files[0]);
});

document.getElementById('style-image').addEventListener('change', (e) => {
  styleImage.src = URL.createObjectURL(e.target.files[0]);
});

// Load TensorFlow.js model
async function loadModel() {
  try {
    styleNet = await tf.loadGraphModel(
      'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2',
      { fromTFHub: true }
    );
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading the model:', error);
    alert('Failed to load the model. Please check your internet connection and try again.');
  }
}

async function applyStyleTransfer() {
  try {
    if (!contentImage.src || !styleImage.src) {
      alert('Please select both content and style images first.');
      return;
    }

    if (!styleNet) {
      console.log('Loading model...');
      await loadModel();
    }

    if (!styleNet) {
      throw new Error('Model failed to load');
    }

  const contentTensor = tf.browser.fromPixels(contentImage).toFloat().expandDims();
  const styleTensor = tf.browser.fromPixels(styleImage).toFloat().expandDims();

    console.log('Processing images...');
    const result = await styleNet.executeAsync({
      'content_image': contentTensor,
      'style_image': styleTensor
    });

    console.log('Rendering result...');
    await tf.browser.toPixels(result[0].squeeze(), outputCanvas);
    console.log('Style transfer complete!');

    // Cleanup tensors
    tf.dispose([contentTensor, styleTensor, result]);
  } catch (error) {
    console.error('Error during style transfer:', error);
    alert('An error occurred during style transfer. Please try again.');
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  console.log('Application started');
  loadModel();
});
