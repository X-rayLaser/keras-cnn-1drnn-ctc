# A toolkit for training neural networks to perform line-level Handwritten Text Recognition (HTR)

The toolkit is built on top of TensorFlow/Keras.
It comes with ready-to-train models, evaluation metrics, automatic data preparation and more.

## Key features
- built-in model implementations
- automatic data pre-processing
- training on your own data
- built-in performance metrics: LER (Label Error Rate)
- handwriting language independence

## Built-in models
- CNN-1DRNN-CTC [1]

# Pre-requisites
- Python >= 3.6
- TensorFlow >= 2.0
- tested on Ubuntu

# Installation
```
git clone https://github.com/X-rayLaser/Keras-HTR.git
cd Keras-HTR
```
Optionally, create and activate a Python virtual environment:
```
virtualenv --python=/path/to/python3/executable venv
. venv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```

# Quick start

Create working line-level HTR system in just 4 steps:
- Create a subclass of Source class representing raw data examples
- Use the data source to build a dataset
- Train model with a particular architecture on that dataset
- Take trained model and use it to perform recognition

You only need to focus on the first step. Once you implement a class 
for a data source, the steps that follow will automatically pre-process 
the data, train a neural network and save it.

Below is example of training CNN-1DRNN-CTC model on synthetic images using SyntheticSource class. 

## Build a dataset using synthetic words data source, store it in temp_ds folder
```
python build_lines_dataset.py --source='keras_htr.data_source.synthetic.SyntheticSource' --destination=temp_ds --size=1000
```
Note that the source argument expects a fully-qualified name of a class representing a data source.

## Train a model
```
python train.py temp_ds --units=64 --epochs=80 --model_path=conv_lstm_model
```
The script will save a model at the end of each training epoch. Therefore, you may interrupt (Ctrl+C) the script when
a loss becomes small enough. A self-contained model for inference will be saved in ```conv_lstm_model/inference_model.h5```.
You can load the model later like so:
```
import tensorflow as tf
tf.keras.models.load_model('conv_lstm_model/inference_model.h5', custom_objects={'tf': tf})
```

## Run demo script
```
python demo.py conv_lstm_model temp_ds/test
```
## Recognize handwriting
Recognize an image taken from a test dataset after necessary preprocessing was already applied
```
python htr.py conv_lstm_model temp_ds/character_table.txt temp_ds/test/0.png
```
To recognize an arbitrary raw image, pass an argument --raw=True 
(this will ensure that all necessary preprocessing steps will be applied such as binarization, resizing, etc.):
```
python htr.py conv_lstm_model temp_ds/character_table.txt /path/to/unseen_image.png --raw=True
```


# Data sources

A data source is a Python generator that yields raw examples in the form of tuples 
(line_image, text). The Keras-HTR toolkit uses data sources to construct a train/val/test split,
build a character table, collect useful meta-information about the data set such as 
average image height, width and more.

## SyntheticSource

It is generator of printed text examples

## IAMSource
It is generator of handwritings taken from IAM handwriting database.
Before you can use this source, you have to download the actual database.

- create a directory named iam_database in the repository directory
- register an account on http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
- download xml.tgz archive file  (you will be prompted to enter your password)
```
curl -o iam_database/xml.tgz -u <user_name> http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
```
- download lines.tgz archive file (you will be prompted to enter your password)
```
curl -o iam_database/lines.tgz -u <user_name> http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
```
- extract both archives

The project directory structure should look as follows:
```
.gitignore
build_lines_dataset.py
demo.py
...
keras_htr\
iam_database\
    lines\
        a01\
        a02\
        a03\
        ...
    xml\
        a01-000u.xml
        a01-000x.xml
        a01-003.xml
        ...
```

Create 1000 training examples using IAM database:
```
python build_lines_dataset.py --source='iam' --destination=temp_ds --size=1000
```

## Custom Source

To train a model on your data, you need to create a subclass of Source class and 
implement an iterator method that yields a pair (line_image, text) at each step.
Here line_image is either a path to an image file or Pillow image object, the text 
is a corresponding transcription.

Let's create a dummy source that produces a total of 100 pairs of random images with some text.
- create a python file mysource.py in keras_htr/data_source directory
- create a subclass of Source class and implement its ```__iter__``` method.
```
import tensorflow as tf
import numpy as np
from keras_htr.data_source.base import Source


class MySource(Source):
    def __iter__(self):
        for i in range(100):
            a = np.round(np.random.random((300, 500, 1)) * 255)
            image = tf.keras.preprocessing.image.array_to_img(a)
            yield image, "Line of text {}".format(i)
```
- use this source by providing it's fully-qualified class name
```
python build_lines_dataset.py --source='keras_htr.data_source.mysource.MySource' --destination=temp_ds --size=100
```

# Training on IAM dataset

Pre-requisite: you have to setup IAMSource first (see the section on IAMSource above).

Prepare a dataset by extracting 8000 examples from IAM database and preprocessing them 
(it might take a few minutes)
```
python build_lines_dataset.py --source='iam' --destination=temp_ds --size=8000
```

Begin training a cnn-1drnn-ctc model for 80 epochs using 256 hidden units in LSTM layers.
When validation loss stops decreasing, press Ctrl+C to stop the script execution. 
```
python train.py temp_ds --units=256 --epochs=80 --model_path=conv_lstm_model
```

# References

[1] [Joan Puigcerver. Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)

# Support
If you find this repository useful, consider starring it by clicking at the ★ button.
It would be much appreciated.
