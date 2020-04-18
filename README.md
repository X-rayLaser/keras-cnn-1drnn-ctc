# A toolkit for developing handwritten text recognition (HTR) pipelines

Easy to use toolkit for the rapid development of offline handwritten text recognition (HTR) system. 
The toolkit provides a set of useful utilities and scripts for training and evaluating 
models and performing line recognition. It is shipped with ready-to-train model 
implementations for HTR tasks.

## Key features
- built-in model implementations
- automatic data pre-processing
- built-in performance metrics: LER (label error rate)
- data-set independence
- handwriting language independence

## Built-in models
- CNN-1DRNN-CTC [1]

# Quick start

Create working HTR system in just 4 steps:
- Subclass Source class representing raw data examples
- Use the data source to build a dataset
- Train model with a particular architecture on the dataset
- Take trained model and use it to perform recognition

You only need to focus on the first step. Once you implement a class 
for a data source, the steps that follow will automatically pre-process 
the data,  train a neural network and save it.

Below is example of training 1D-LSTM model on synthetic images using SyntheticSource class. 

## Build a dataset using synthetic words data source, store it in temp_ds folder
```
python build_lines_dataset.py --source='keras_htr.data_source.synthetic.SyntheticSource' --destination=temp_ds --size=1000
```
Note that the source argument expects a fully-qualified name of a class representing a data source.

## Train a model
```
python train.py temp_ds --units=32 --epochs=35 --model_path=conv_lstm_model
```
## Run demo script
```
python demo.py conv_lstm_model temp_ds/test
```
## Recognize handwriting
```
python htr.py conv_lstm_model temp_ds/character_table.txt temp_ds/test/0.png
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

# References

[1] [Joan Puigcerver. Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)
