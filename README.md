# A toolkit for developing handwritten text recognition (HTR) pipelines

# Usage

- Build/compile a dataset using synthetic words data source, store it in temp_ds folder
```
python build_lines_dataset.py --source='keras_htr.data_source.synthetic.SyntheticSource' --destination=temp_ds
```
- Train a model
```
python train.py temp_ds --units=32 --epochs=35 --model_path='conv_lstm_model.h5
```
- Run demo script to look at the predicted text given image 
```
python demo.py conv_lstm_model.h5 temp_ds/test
```
- Recognize image containing handwriting
```
python htr.py conv_lstm_model.h5 temp_ds/test/0.png
```
