#  Intention Oriented Image Captions with Guiding Objects

This is a pytorch implementation of [_Intention Oriented Image Captions with Guiding Objects_](https://arxiv.org/abs/1811.07662) by Zheng et al.

Currently the model architectures are mocked up, and I attempted to reproduce table 1 (only CGO results and base with beam=1). Object average numbers and recalls are on par with the original paper. Note that for base model and "Caption GT label", the numbers are lower because I used exact match when counting if an object appears in the sentence, while the original paper converted the words to their roots. METEOR scores are lower but comparable to the original paper.

### Original Paper

| Model            | METEOR | Avg.Num | Avg.R |
| ---------------- | ------ | ------- | ----- |
| Base (b=1)       | 26.6   | 1.50    | 0.55  |
| CGO (k=1)        | 24.4   | 1.62    | 0.50  |
| CGO (k=3)        | 24.4   | 2.43    | 0.67  |
| CGO (k=5)        | 24.2   | 2.77    | 0.73  |
| CGO (k=10)       | 24.2   | 2.92    | 0.75  |
| Caption GT label | -      | 2.01    | 0.61  |
| CGO (caption GT) | 28.0   | -       | -     |
| CGO (det GT)     | 24.2   | 3.06    | 1.00  |

## This Implementation

| Model            | METEOR | Avg.Num | Avg.R |
| ---------------- | ------ | ------- | ----- |
| Base (b=1)       | 25.2   | 0.76    | 0.35  |
| CGO (k=1)        | 24.1   | 1.40    | 0.46  |
| CGO (k=3)        | 23.2   | 2.05    | 0.66  |
| CGO (k=5)        | 22.7   | 2.35    | 0.76  |
| CGO (k=10)       | 22.1   | 2.65    | 0.86  |
| Caption GT label | -      | 1.12    | 0.49  |
| CGO (caption GT) | 26.9   | 1.18    | 0.51  |
| CGO (det GT)     | 24.1   | 2.89    | 1.00  |

# Instructions

Run `git clone --recursive git@github.com:Inlinebool/cgo-pytorch.git` to clone the repo as well as evaluation tools from `Maluuba/nlg-eval`.

## Requirements

See `environment.yml`. `java > 1.8` is required for reproducing table 1.

With `conda`, simply run `conda env create -f environment.yml` to create a conda virtualenv with all required packages named `cgo`.

## Data Preparation

Download [bottom up features](https://storage.googleapis.com/up-down-attention/trainval_36.zip) ([Andersen el al. 2018](https://github.com/peteanderson80/bottom-up-attention)), and extract to `data/trainval_36`.

Download [COCO 2014 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and extract to `data/annotations_trainval2014`

Since we are using pre-trained bottom up features, we do not need the original images. You are free to download them for debug purposes. A convienient script `inspect_coco_with_image_id.py` is provided for quick inspection of the images.

Next, download the [Karpathy's split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) (Karpathy et al. 2015), and extract to `data/caption_datasets`

Finally, run 

`python prepare_features.py` to convert the `.tsv` image features to `hdf5` format for quick indexing.

`python prepare_input_classifier.py` to prepare datasets for the classifier, and run

`python prepare_input_lstm.py` to prepare datasets for the LSTMs.

## Training

Run `python train_classifier.py` to train the classifier with default settings. Model (and checkpoints) will be saved to `models/classifier.pkl`.


Next, run `python train_lstm.py left` to train the `LSTM-left` with default settings. Model (and checkpoints) will be saved to `models/lstm-left.pkl`.

Lastly, run `python train_lstm.py right` to train the `LSTM-right` with default settings. Model (and checkpoints) will be saved to `models/lstm-right.pkl`.

## Evaluation

Run `python eval_classifier.py` to evaluate the classifier. Results will be saved to `results/results_classifier_thresh{x}` and `results/results_classifier_top{k}`, where x is the confidence threshold and k is the top k setting.

To reproduce table 1, first run the following to install necessary tools.

```
pip install -e nlg-eval
nlg-eval --set-up
``` 

Then, run `python eval_cgo_table1.py` to reproduce table 1. Results will be saved to `results/table_1.json`.

Models for the current results:

[models and params](https://drive.google.com/drive/folders/1XluVIfmxOnKXzPBSnR3oD3_20H_bDYhK?usp=sharing)
