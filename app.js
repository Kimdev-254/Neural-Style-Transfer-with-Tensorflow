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
  styleNet = await tf.loadGraphModel(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2',
    { fromTFHub: true }
  );
  console.log('Model loaded.');
}

async function applyStyleTransfer() {
  if (!styleNet) {
    await loadModel();
  }

  const contentTensor = tf.browser.fromPixels(contentImage).toFloat().expandDims();
  const styleTensor = tf.browser.fromPixels(styleImage).toFloat().expandDims();

  const result = await styleNet.executeAsync({
    'content_image': contentTensor,
    'style_image': styleTensor
  });

  await tf.browser.toPixels(result[0].squeeze(), outputCanvas);

  // Cleanup tensors
  tf.dispose([contentTensor, styleTensor, result]);
}

loadModel();
