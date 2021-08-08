
imageSize=32
channels=3

// Load image.
async function runModel() {
    // create onnx session
    const session = new onnx.InferenceSession();
    await session.loadModel("pretrained/onnx_model.onnx");

    // load image
    image_url = document.getElementById('imageSource').src
    const imageLoader = new ImageLoader(imageSize, imageSize);
    const imageData = await imageLoader.getImageData(image_url);
    console.log(imageData)

    // Preprocess the image data to match input dimension requirement, which is 1*3*32*32.
    const width = imageSize;
    const height = imageSize;
    const preprocessedData = preprocess(imageData.data, width, height);

    console.log(preprocessedData)
    const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, channels, width, height]);

    // Run model with Tensor inputs and get the result.
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;

    console.log(outputData)
    infernence(outputData)
}

function preprocess(data, width, height) {
  // ndarray gives channels as RGBA, so you need to use 4
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  // Normalize 
  // The pixel values are normalized between 0 and 1 by dividing 255. 
  ndarray.ops.divseq(dataFromImage, 255);
  // RGB channels are normalized with the mean values [0.5, 0.5, 0.5] 
  // and the standard deviations [0.5, 0.5, 0.5].
  ndarray.ops.subseq(dataFromImage.pick(null, null, 0), 0.5);
  ndarray.ops.divseq(dataFromImage.pick(null, null, 0), 0.5);

  ndarray.ops.subseq(dataFromImage.pick(null, null, 1), 0.5);
  ndarray.ops.divseq(dataFromImage.pick(null, null, 1), 0.5);

  ndarray.ops.subseq(dataFromImage.pick(null, null, 2), 0.5);
  ndarray.ops.divseq(dataFromImage.pick(null, null, 2), 0.5);

  // create new nd array of size [1*3*32*32]
  const dataProcessed = ndarray(new Float32Array(width * height * channels), [1, channels, height, width]);
  // Realign imageData from [32*32*3] to the correct dimension [1*3*32*32].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

  return dataProcessed.data;
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function infernence(data) {
  let outputClasses = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
  const probs = Array.from(data)
  class_idx = argMax(probs)
  console.log(class_idx)
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = 'Model predicted Image as - '+outputClasses[class_idx];
}

runModel()

/*
let fileTag = document.getElementById("filetag"),
    preview = document.getElementById("preview");
    
fileTag.addEventListener("change", function() {
  changeImage(this);
});

function changeImage(input) {
  var reader;

  if (input.files && input.files[0]) {
    reader = new FileReader();

    reader.onload = function(e) {
      preview.setAttribute('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]);
  }
}
*/