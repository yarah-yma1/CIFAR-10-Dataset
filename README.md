# CIFAR-10-Dataset
The CIFAR-10 dataset is utlized in machine learning containing 60,000 labeled 32×32 color images across 10 classes, commonly used to train and evaluate image classification algorithms. This repository includes a media folder where outputs + visuals are found, a documentation folder where statistical information is found along side any other important documentations, and source code folder where codes are found. 
# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [References](#references)
# Implementation
The models implementation consists of an input of the MNIST dataset in which it involves a collection of 70,000 handwritten digits (0-9), with each image being 28x28 pixels. The model is trained using 25 EPOCHs in which it is passed through 25 times, with a validation split of 0.2 to prevent overfitting and validates on a specific portion of the data. 
# Requirments 
This project requires tensorflow, keras, and scikit-learn. It was developed using a Python environment through VSCode.
Use 'pip install -r requirements.txt' to install the following dependencies:
```
absl-py==2.3.1
astunparse==1.6.3
certifi==2025.8.3
charset-normalizer==3.4.3
contourpy==1.3.3
cycler==0.12.1
flatbuffers==25.9.23
fonttools==4.60.1
gast==0.6.0
google-pasta==0.2.0
grpcio==1.75.1
h5py==3.14.0
idna==3.10
keras==3.11.3
kiwisolver==1.4.9
libclang==18.1.1
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.6
mdurl==0.1.2
ml_dtypes==0.5.3
namex==0.1.0
numpy==2.3.3
opt_einsum==3.4.0
optree==0.17.0
packaging==25.0
pillow==11.3.0
protobuf==6.32.1
Pygments==2.19.2
pyparsing==3.2.5
python-dateutil==2.9.0.post0
requests==2.32.5
rich==14.1.0
scipy==1.16.2
setuptools==80.9.0
six==1.17.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.1.0
typing_extensions==4.15.0
urllib3==2.5.0
Werkzeug==3.1.3
wheel==0.45.1
wrapt==1.17.3
```
# How to Use
To run this code, you will need to have a Python environment installed on your computer. You can download "cifar_10_dataset.py" into a folder, and open the folder within VS Code.
The CIFAR-10 dataset is automatically downloaded and loaded in the script, so no external dataset is required.

# References 
[1]GeeksforGeeks, “CIFAR10 Image Classification in TensorFlow,” GeeksforGeeks, Apr. 29, 2021. https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/
‌
