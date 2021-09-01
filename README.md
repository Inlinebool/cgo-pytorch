#  (WIP) Intention Oriented Image Captions with Guiding Objects

This is a (work in progress) pytorch implementation of _Intention Oriented Image Captions with Guiding Objects_ by Zheng et al.

Currently the model architectures are mocked up, and I attempted to reproduce table 1 (only CGO results). However, reproduced table 1 results (especially METEOR scores) are suboptimal. There might be bugs in the code.

## Table 1

### Original Paper

| Model            | METEOR | Avg.Num | Avg.R |
| ---------------- | ------ | ------- | ----- |
| CGO (k=1)        | 24.4   | 1.62    | 0.50  |
| CGO (k=3)        | 24.4   | 2.43    | 0.67  |
| CGO (k=5)        | 24.2   | 2.77    | 0.73  |
| CGO (k=10)       | 24.2   | 2.92    | 0.75  |
| Caption GT label | -      | 2.01    | 0.61  |
| CGO (caption GT) | 28.0   | -       | -     |
| CGO (det GT)     | 24.2   | 3.06    | 1.00  |

### This Implementation

| Model            | METEOR | Avg.Num | Avg.R |
| ---------------- | ------ | ------- | ----- |
| CGO (k=1)        | 18.7   | 1.28    | 0.41  |
| CGO (k=3)        | 18.3   | 2.04    | 0.66  |
| CGO (k=5)        | 17.7   | 2.35    | 0.76  |
| CGO (k=10)       | 17.1   | 2.65    | 0.86  |
| Caption GT label | -      | 1.12    | 0.49  |
| CGO (caption GT) | 22.3   | 1.19    | 0.51  |
| CGO (det GT)     | 19.0   | 2.89    | 1.00  |

METEOR scores are consistantly lower than the original paper. The most confusing is that "Caption GT label" results are different. These should be the same as they are ground truth results. 

## Instructions

Run `git clone --recursive git@github.com:Inlinebool/cgo-pytorch.git` to clone the repo as well as evaluation tools from `Maluuba/nlg-eval`.

### Requirements

See `environment.yml`. `java > 1.8` is required for reproducing table 1.

With `conda`, simply run `conda env create -f environment.yml` to create a conda virtualenv with all required packages named `cgo`.

### Data Preparation

Download [bottom up features](https://storage.googleapis.com/up-down-attention/trainval_36.zip) ([Andersen el al. 2018](https://github.com/peteanderson80/bottom-up-attention)), and extract to `data/trainval_36`.

Download [COCO 2014 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and extract to `data/annotations_trainval2014`

Since we are using pre-trained bottom up features, we do not need the original images. You are free to download them for debug purposes. A convienient script `inspect_coco_with_image_id.py` is provided for quick inspection of the images.

Next, download the [Karpathy's split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) (Karpathy et al. 2015), and extract to `data/caption_datasets`

Finally, run 

`python prepare_features.py` to convert the `.tsv` image features to `hdf5` format for quick indexing.

`python prepare_input_classifier.py` to prepare datasets for the classifier, and run

`python prepare_input_lstm.py` to prepare datasets for the LSTMs.

### Training

Run `python train_classifier.py` to train the classifier with default settings. Model (and checkpoints) will be saved to `models/classifier.pkl`.


Next, run `python train_lstm.py left` to train the `LSTM-left` with default settings. Model (and checkpoints) will be saved to `models/lstm-left.pkl`.

Lastly, run `python train_lstm.py right` to train the `LSTM-right` with default settings. Model (and checkpoints) will be saved to `models/lstm-right.pkl`.

### Evaluation

Run `python eval_classifier.py` to evaluate the classifier. Results will be saved to `results/results_classifier_thresh{x}` and `results/results_classifier_top{k}`, where x is the confidence threshold and k is the top k setting.

To reproduce table 1, first run the following to install necessary tools.

```
pip install -e nlg-eval
nlg-eval --set-up
``` 

Then, run `python eval_cgo_table1.py` to reproduce table 1. Results will be saved to `results/table_1.json`.

Models for the current results:

[classifier](https://drive.google.com/file/d/12rHjAKwOBygR4GeCIdO_XTwGo_JIs7TW/view?usp=sharing), [classifier parameters](https://drive.google.com/file/d/1QNjOQy7hZY6c1sFVVEHkQHpzgGrrxsVO/view?usp=sharing)

[lstm-left](https://drive.google.com/file/d/1xxIBT4Xv0lgp3UqO0TZhFMAXJ2tcF0Wt/view?usp=sharing), [lstm-left parameters](https://drive.google.com/file/d/1a8rcUIpWNroDYwjMb5yIKkvt68RDYEcy/view?usp=sharing)

[lstm-right](https://drive.google.com/file/d/1wvwV_DvT5_fYRwqZziiEtfw4_FvCFhzN/view?usp=sharing), [lstm-right parameters](https://drive.google.com/file/d/1eG2Van3fwxxRJ88m7veEti5vOHZRQ2lj/view?usp=sharing)

