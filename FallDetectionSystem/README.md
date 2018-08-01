# Fall Detection System

Keras implementation of video classifiers 
3D CNN + LSTM 


# Usage

### Train Deep Learning model

To train a deep learning model, run the following commands:

```bash
python cnn_bidirectional_lstm_train.py 
```

To test the model: 
```bash
python cnn_bidirectional_lstm_predict.py 

```

The default dataset used is UCF101
To change to another dataset, go to utility/UCF101_loader and change the data_set_name parameter in its fit() method to other dataset name instead of UCF-101 will allow it to be trained on other video datasets

# Evaluation Report

20 classes from UCF101 is used to train the video classifier. 30 epochs are set for the training

### Evaluate Convolutional Network

```bash

python cnn_train.py 

```

The Convolutional Network: (accuracy around 22.73% for training and 28.75% for validation)

### Evaluate CNN+LSTM

```bash

python cnn_lstm_train.py 

```

The LSTM with 3D CNN feature extractor: (accuracy around 68.9% for training and 75% for validation)

### Evaluate CNN+Bidirectional-LSTM

```bash

python cnn_bidirectional_lstm_train.py 

```

The bidirectional LSTM with 3D CNN feature extractor: (accuracy around 91.97% for training and 80% for validation)

# Note 

### Configure Keras to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
