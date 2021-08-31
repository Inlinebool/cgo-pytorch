import argparse
import json
import os

import torch
from loguru import logger
from nlgeval import NLGEval
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import ClassifierDataset, LSTMDataset
from models import LanguageModel, ObjectClassifier


def cgo_decode(image_features: torch.Tensor,
               guiding_objs: list[str],
               word_map: dict,
               reversed_word_map: list[str],
               lstm_left: LanguageModel,
               lstm_right: LanguageModel,
               beam: int = 3):
    guiding_left = []
    guiding_left.append(word_map['<start>'])
    guiding_obj_tokens = guiding_objs[0].split(' ')
    for go in reversed(guiding_objs):
        tokens = go.split(' ')
        for token in reversed(tokens):
            guiding_left.append(word_map[token])
    # no need for padding
    guiding_left_length = len(guiding_left) - 1  # only count actual seq
    guiding_left = torch.tensor(guiding_left,
                                device=lstm_left.device).view(1, -1)
    guiding_left_length = torch.tensor(guiding_left_length,
                                       device=lstm_left.device).view(1, -1)
    left_seq = lstm_left.decode(
        (image_features, guiding_left, guiding_left_length), word_map['<end>'],
        beam)
    guiding_right = left_seq[::-1]
    guiding_right[0] = word_map['<start>']
    for token in guiding_obj_tokens:
        guiding_right.append(word_map[token])
    guiding_right_length = len(guiding_right) - 1
    guiding_right = torch.tensor(guiding_right,
                                 device=lstm_right.device).view(1, -1)
    guiding_right_length = torch.tensor(guiding_right_length,
                                        device=lstm_right.device).view(1, -1)
    right_seq = lstm_right.decode(
        (image_features, guiding_right, guiding_right_length),
        word_map['<end>'], beam)

    full_seq = guiding_right.view(-1).tolist() + right_seq
    return [reversed_word_map[x] for x in full_seq]


def cgo_decode_list(image_features: torch.Tensor,
                    guiding_obj_list: list[int],
                    category_names: list[str],
                    word_map: dict,
                    reversed_word_map: list[str],
                    lstm_left: LanguageModel,
                    lstm_right: LanguageModel,
                    beam: int = 3):
    decoded_list = []
    for guiding_obj in guiding_obj_list:
        decoded = cgo_decode(image_features, [category_names[guiding_obj]],
                             word_map, reversed_word_map, lstm_left,
                             lstm_right, beam)
        assert decoded[0] == '<start>'
        if decoded[-1] != '<end>':
            logger.warning('decoded sentence not ending with <end>.')
            logger.warning('image_id: {0}'.format(image_id))
            logger.warning('guiding_obj: {0}'.format(
                category_names[guiding_obj]))
            logger.warning('decoded: {0}'.format(decoded))
            decoded_list.append(decoded[1:])
        else:
            decoded_list.append(decoded[1:-1])
    return decoded_list


def sentence_obj_hit(sentence, categories):
    hit = set()
    for word in sentence:
        if word in categories:
            hit.add(categories.index(word))
    bigrams = list(zip(sentence, sentence[1:]))
    for bigram in bigrams:
        word = bigram[0] + ' ' + bigram[1]
        if word in categories:
            hit.add(categories.index(word))
    return hit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate LSTM-right plain as baseline.')

    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--classifier', type=str, default='classifier.pkl')
    parser.add_argument('--lstm_left', type=str, default='lstm-left.pkl')
    parser.add_argument('--lstm_right', type=str, default='lstm-right.pkl')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--lstm_data_dir', type=str, default='lstm')
    parser.add_argument('--classifier_data_dir',
                        type=str,
                        default='classifier')
    parser.add_argument('--feature_filename',
                        type=str,
                        default='image_features.h5')
    parser.add_argument('--featuremap_filename',
                        type=str,
                        default='feature_map.json')
    parser.add_argument('--category_names_filename',
                        type=str,
                        default='category_names.json')
    parser.add_argument('--word_map_filename',
                        type=str,
                        default='word_map.json')
    parser.add_argument('--label_right_filename',
                        type=str,
                        default='label_right.json')
    parser.add_argument('--label_detection_filename',
                        type=str,
                        default='label_detection.json')
    parser.add_argument('--label_caption_filename',
                        type=str,
                        default='label_caption.json')
    parser.add_argument('--beam', type=int, default=3)

    args = parser.parse_args()

    classifier_path = os.path.join(args.model_dir, args.classifier)
    lstm_left_path = os.path.join(args.model_dir, args.lstm_left)
    lstm_right_path = os.path.join(args.model_dir, args.lstm_right)
    feature_path = os.path.join(args.data_dir, args.feature_filename)
    featuremap_path = os.path.join(args.data_dir, args.featuremap_filename)
    caption_label_path = os.path.join(args.data_dir, args.lstm_data_dir,
                                      args.label_right_filename)
    detection_label_path = os.path.join(args.data_dir,
                                        args.classifier_data_dir,
                                        args.label_detection_filename)
    cap_detection_label_path = os.path.join(args.data_dir,
                                            args.classifier_data_dir,
                                            args.label_caption_filename)
    word_map_path = os.path.join(args.data_dir, args.lstm_data_dir,
                                 args.word_map_filename)
    category_names_path = os.path.join(args.data_dir, args.classifier_data_dir,
                                       args.category_names_filename)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    ref_path = os.path.join(args.result_dir, 'ref.json')
    hyp_path = os.path.join(args.result_dir, 'hyp-right.json')
    table1_path = os.path.join(args.result_dir, 'result_lstm-right.json')

    with open(word_map_path, 'r') as fp:
        word_map = json.load(fp)
    with open(category_names_path, 'r') as fp:
        category_names = json.load(fp)

    reverse_word_map = [(word_map[k], k) for k in word_map]
    reverse_word_map = sorted(reverse_word_map)
    reverse_word_map = [x[1] for x in reverse_word_map]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_caption = LSTMDataset(feature_path, caption_label_path,
                                  featuremap_path, 'test')
    dataset_detection = ClassifierDataset(feature_path, detection_label_path,
                                          featuremap_path, 'test')
    dataset_cap_detection = ClassifierDataset(feature_path,
                                              cap_detection_label_path,
                                              featuremap_path, 'test')

    lstm_left: LanguageModel = torch.load(lstm_left_path)
    lstm_right: LanguageModel = torch.load(lstm_right_path)
    classifier: ObjectClassifier = torch.load(classifier_path)
    lstm_left.eval()
    lstm_right.eval()
    classifier.eval()

    loader_caption = DataLoader(dataset_caption, shuffle=False)
    loader_detection = DataLoader(dataset_detection, shuffle=False)
    loader_cap_detection = DataLoader(dataset_cap_detection, shuffle=False)

    logger.info('preparing references...')
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as fp:
            ref = json.load(fp)
    else:
        ref = {}
        for cap_label in tqdm(loader_caption):
            image_id = str(cap_label['image_id'][0])
            if image_id not in ref:
                ref[image_id] = []
            seq, seq_length = cap_label['label']
            ref[image_id].append(
                [reverse_word_map[x] for x in seq[0][1:seq_length[0] + 1]])
        with open(ref_path, 'w') as fp:
            json.dump(ref, fp)

    logger.info('total test images: {0}'.format(len(ref)))

    table_k = [1, 3, 5, 10]
    hyp = {}

    logger.info('preparing hypothesis...')
    if os.path.exists(hyp_path):
        with open(hyp_path, 'r') as fp:
            hyp = json.load(fp)
    else:
        hyp = {}
        with torch.no_grad():
            for cap_det_label in tqdm(loader_cap_detection):
                image_id = str(cap_det_label['image_id'][0])
                classifier_inputs = [
                    x.to(device) for x in cap_det_label['inputs']
                ]
                image_features = classifier_inputs[0]
                seq = torch.tensor([word_map['<start>']]).view(1,
                                                               -1).to(device)
                seq_length = torch.tensor([0]).view(1, -1).to(device)
                decoded = lstm_right.decode((image_features, seq, seq_length),
                                            word_map['<end>'],
                                            beam=3)
                decoded = [reverse_word_map[x] for x in decoded]
                if decoded[-1] != '<end>':
                    logger.warning('decoded sentence not ending with <end>.')
                    logger.warning('image_id: {0}'.format(image_id))
                    logger.warning('decoded: {0}'.format(decoded))
                    hyp[image_id] = decoded
                else:
                    hyp[image_id] = decoded[:-1]

        with open(hyp_path, 'w') as fp:
            json.dump(hyp, fp)

    print(len(hyp))

    logger.info('preparing detection ground truths...')
    table_1 = []
    det_gt_objs = {}
    for det_label in tqdm(loader_detection):
        image_id = str(det_label['image_id'][0])
        det_gt = torch.nonzero(
            det_label['label'][0].view(-1)).view(-1).tolist()
        det_gt_objs[image_id] = det_gt

    nlg = NLGEval(False, True, True, ['Bleu_1', 'ROUGE_L', 'CIDEr'])
    title = 'LSTM-right'
    logger.info('computing metrics for {0}'.format(title))
    k_hyp = []
    k_ref = [[] for _ in range(5)]
    avg_num = 0
    true_positive = [0 for _ in category_names]
    det_gt_positive = [0 for _ in category_names]
    for image_id in tqdm(hyp):
        img_hyp = [hyp[image_id]]
        k_hyp += [' '.join(x) for x in img_hyp]
        img_ref = [' '.join(x) for x in ref[image_id]]
        if len(img_ref) != 5:
            logger.warning('caption reference more than 5! ({0})'.format(
                len(img_ref)))
            logger.warning('only using first 5 reference.')
        for i, r in enumerate(img_ref):
            if i < 5:
                k_ref[i].append(r)
        hit_set = set([])
        for sentence in img_hyp:
            hit = sentence_obj_hit(sentence, category_names)
            hit_set = hit_set.union(hit)
        if image_id in det_gt_objs:
            det_gt = set(det_gt_objs[image_id])
            true_positives = hit_set.intersection(det_gt)
            avg_num += len(true_positives)
            for obj in true_positives:
                true_positive[obj] += 1
            for obj in det_gt:
                det_gt_positive[obj] += 1
    avg_num /= len(det_gt_objs)
    tp = torch.tensor(true_positive, dtype=torch.float32)
    det_p = torch.tensor(det_gt_positive, dtype=torch.float32)
    avg_recall = torch.mean(tp / det_p).item()  # macro avg, per paper
    assert len(k_ref) == 5
    assert len(k_hyp) == len(k_ref[0])
    logger.info('computing caption metrics...')
    metrics = nlg.compute_metrics(k_ref, k_hyp)
    meteor = metrics['METEOR']
    result = {
        'title': title,
        'meteor': meteor,
        'avg_num': avg_num,
        'avg_recall': avg_recall
    }
    logger.info('result: {0}'.format(result))
    table_1.append(result)
