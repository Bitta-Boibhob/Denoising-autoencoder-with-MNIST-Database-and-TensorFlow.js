const originalImg = document.getElementById("originalImg");
const noisyImg = document.getElementById("noisyImg");
const restoredImg = document.getElementById("restoredImg");
const newDigit = document.getElementById("newDigit");
const newNoise = document.getElementById("newNoise");
const mseResult = document.getElementById("mseResult");

newDigit.onclick = () => updateInputTensor();
newNoise.onclick = () => updateNoiseTensor();

var denoising_ae_model;
tf.loadLayersModel("res/model/trained_denoising_autoencoder_model.json").then(
  (model) => {
    denoising_ae_model = model;
    initializeTensors();
  }
);

const initializeTensors = () => {
  noiseTensor = tf.randomNormal([28, 28, 1], 0, 0.5);
  updateInputTensor();
};

const updateInputTensor = () => {
  let randInt = Math.floor(Math.random() * 1002);
  let temp = new Image();
  temp.src = "res/digits_img/" + randInt + ".png";
  temp.onload = () => {
    if (this.inputTensor) {
      inputTensor.dispose();
    }
    inputTensor = tf.tidy(() =>
      tf.browser.fromPixels(temp, 1).toFloat().div(tf.scalar(255))
    );
    tf.browser.toPixels(inputTensor, originalImg);
    updateDistortedTensor();
  };
};

const updateNoiseTensor = () => {
  noiseTensor.dispose();
  noiseTensor = tf.randomNormal([28, 28, 1], 0, 0.5);
  updateDistortedTensor();
};

const updateDistortedTensor = () => {
  if (this.distortedTensor) {
    distortedTensor.dispose();
  }
  distortedTensor = tf.tidy(() => {
    return noiseTensor.add(inputTensor).clipByValue(0, 1);
  });
  tf.browser.toPixels(distortedTensor, noisyImg);
  updateFixedTensor();
};

const updateFixedTensor = () => {
  if (this.fixedTensor) {
    fixedTensor.dispose();
  }
  fixedTensor = tf.tidy(() => {
    return denoising_ae_model.predict(distortedTensor.expandDims()).squeeze();
  });
  tf.browser.toPixels(fixedTensor, restoredImg);

  mseResult.innerText =
    "Mean squared error : " + calculateMSE(inputTensor, fixedTensor) + "%";
};

const calculateMSE = (originalTensor, reconstructedTensor) => {
  // Calculate the MSE between two tensors
  const squaredDifference = tf.squaredDifference(
    originalTensor,
    reconstructedTensor
  );
  const mse = tf.mean(squaredDifference).dataSync()[0];
  return (mse * 100).toFixed(2);
};

// Usage example:
// const mse = calculateMSE(inputTensor, fixedTensor);
// console.log('Mean Squared Error:', mse);
