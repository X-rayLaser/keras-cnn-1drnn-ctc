# A toolkit for developing handwritten text recognition (HTR) pipelines

Easy to use toolkit for the rapid development of offline handwritten text recognition (HTR) system. 
The toolkit provides a set of useful utilities and scripts for training and evaluating 
models and performing recognition. It is shipped with ready-to-train model 
implementations for HTR tasks.

## Key features
- built-in model implementations
- automatic data pre-processing
- built-in performance metrics: LER (label error rate)
- data-set independence
- handwriting language independence

## Built-in models
- 1D-LSTM [1] by Joan Puigcerver

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
python build_lines_dataset.py --source='keras_htr.data_source.synthetic.SyntheticSource' --destination=temp_ds
```
Note that the source argument expects a fully-qualified name of a class representing a data source.

## Train a model
```
python train.py temp_ds --units=32 --epochs=35 --model_path=conv_lstm_model.h5
```
## Run demo script
```
python demo.py conv_lstm_model.h5 temp_ds/test
```
## Recognize handwriting
```
python htr.py conv_lstm_model.h5 temp_ds/character_table.txt temp_ds/test/0.png
```

# Data sources

A data source is a Python generator that yields raw examples in the form of tuples 
(text, line_image). The Keras-HTR toolkit uses data sources to construct a train/val/test split, 
build a character table, collect useful meta-information about the data set such as 
average image height, width and more.

To train a model on your data, you need to create your subclass of Source class and 
implement an iterator method that yields a pair (line_image, text) at each step.
Here line_image is either a path to an image file or Pillow image object, the text 
is a corresponding transcription.

## SyntheticSource

It is generator of printed text examples

## IAMSource
It is generator of handwritings taken from IAM handwriting database.
Before you can use this source, you have to download the actual database:
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database.

# References

[1] [Joan Puigcerver. Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)
