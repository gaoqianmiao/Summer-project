# Fall Detection System

Keras implementation of video classifiers 
3D CNN + LSTM 


# Usage

### Train Deep Learning model

To train a deep learning model, say VGG16BidirectionalLSTMVideoClassifier, run the following commands:

```bash
pip install -r requirements.txt

python vgg16_lstm_train.py 
```
The default dataset used is UCF101
To change to another dataset, go to utility/UCF101_loader and change the line: 

# Evaluation Report

20 classes from UCF101 is used to train the video classifier. 50 epochs are set for the training

### Evaluate Convolutional Network

Below is the train history for the Convolutional Network:

![cnn-history](demo/reports/UCF-101/cnn-history.png)

The Convolutional Network: (accuracy around 22.73% for training and 28.75% for validation)

### Evaluate VGG16+LSTM (top included for VGG16)

Below is the train history for the VGG16+LSTM (top included for VGG16):

![vgg16-lstm-history](demo/reports/UCF-101/vgg16-lstm-history.png)

The LSTM with VGG16 (top included) feature extractor: (accuracy around 87.07% for training and 85.68% for validation)


# Note 

### Configure Keras to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
