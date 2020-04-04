# A toolkit for developing handwritten text recognition (HTR) pipelines

# Usage

- Build/compile a dataset using synthetic words data source, store it in temp_ds folder
```
python build_lines_dataset.py --source='keras_htr.data_source.synthetic.SyntheticSource' --destination=temp_ds
```
- Train a model
```
python train.py temp_ds --units=32 --epochs=35 --model_path=conv_lstm_model.h5
```
- Run demo script to look at the predicted text given image 
```
python demo.py conv_lstm_model.h5 temp_ds/test
```
- Recognize image containing handwriting
```
python htr.py conv_lstm_model.h5 temp_ds/character_table.txt temp_ds/test/0.png
```

# Data sources

Data source is a Python generator that yields raw examples in form of tuples (text, line_image).
Keras-HTR uses data source to construct a train/val/test split, build a character table,
collect useful meta-information about the data set such as average image height, width and more.

To train a model on your own data, you need to create your own subclass of Source class and implement a
__iter__ method that yields a pair (text, line_image) at each step. Here line_image is either
 a path to image file or Pillow image object, text is a corresponding transcription

# Built-in data sources

## SyntheticSource

It is generator of printed text examples

## IAMSource
It is generator of handwritings taken from IAM handwriting database.
