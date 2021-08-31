import argparse
import json
import os
from typing import Counter

from loguru import logger
from tqdm import tqdm

data_dir = 'data/'
classifier_dir = os.path.join(data_dir, 'classifier')
lstm_dir = os.path.join(data_dir, 'lstm')
dataset = 'coco'
split_path = os.path.join(data_dir, 'caption_datasets', 'dataset_coco.json')
feature_map_path = os.path.join(data_dir, 'feature_map.json')


def append_go_to_seq(go, seq):
    if not isinstance(go, list):
        seq.append(go)
    else:
        for w in go:
            seq.append(w)


def prepare_input_lstm():
    parser = argparse.ArgumentParser(description='Prepare input for LSTMs.')

    parser.add_argument('--min_word_freq', type=int, default=5)

    args = parser.parse_args()

    with open(split_path) as fp:
        split_data = json.load(fp)
    with open(feature_map_path, 'r') as fp:
        feature_map = json.load(fp)
    with open(os.path.join(classifier_dir, 'category_names.json'), 'r') as fp:
        categories = json.load(fp)

    word_counter = Counter()
    max_len = 0

    for item in tqdm(split_data['images']):
        image_id = str(item['cocoid'])
        if not image_id in feature_map:
            logger.warning('Not found in features! id: {0}'.format(image_id))
        else:
            for sentence in item['sentences']:
                word_counter.update(sentence['tokens'])
                sentence_len = len(sentence['tokens'])
                max_len = sentence_len if sentence_len > max_len else max_len

    logger.info('maximum sentence length: {0}'.format(max_len))
    vocab = [w for w in word_counter if word_counter[w] > args.min_word_freq]
    word_map = {v: i + 1 for i, v in enumerate(vocab)}
    pad_token = '<pad>'
    start_token = '<start>'
    end_token = '<end>'
    unk_token = '<unk>'
    word_map[pad_token] = 0
    word_map[start_token] = len(word_map)
    word_map[end_token] = len(word_map)
    word_map[unk_token] = len(word_map)
    logger.info('vocab size: {0}'.format(len(word_map)))
    logger.info('max index (unk_token): {0}'.format(word_map[unk_token]))

    label_left_splited = {'train': [], 'val': [], 'test': []}
    label_right_splited = {'train': [], 'val': [], 'test': []}

    for item in tqdm(split_data['images']):
        image_id = str(item['cocoid'])
        if not image_id in feature_map:
            logger.warning('Not found in features! id: {0}'.format(image_id))
        else:
            split = item['split']
            if split == 'restval':
                split = 'train'
            assert split in ['train', 'val', 'test']

            for sentence in item['sentences']:
                seq_length = len(sentence['tokens'])
                full_seq = [start_token]
                full_seq += [w for w in sentence['tokens']]
                full_seq += [end_token]
                full_seq += [pad_token for _ in range(max_len - seq_length)]
                assert len(full_seq) == max_len + 2
                full_seq_onehot = [
                    word_map.get(w, word_map[unk_token]) for w in full_seq
                ]

                label_right_splited[split].append({
                    'image_id': image_id,
                    'seq': full_seq_onehot,
                    'seq_length': seq_length
                })

                reversed_sentence = sentence['tokens'][::-1]
                candidate_objs = []
                for i, word in enumerate(reversed_sentence):
                    if word in categories:
                        candidate_objs.append({'tokens': [word], 'index': i})
                bigrams = list(zip(reversed_sentence, reversed_sentence[1:]))
                for i, bigram in enumerate(bigrams):
                    word = bigram[1] + ' ' + bigram[0]
                    if word in categories:
                        candidate_objs.append({
                            'tokens': bigram,
                            'index': i + 1
                        })

                for i, obj in enumerate(candidate_objs):
                    guiding_objs = candidate_objs[:i + 1]
                    guiding_seq = []
                    for go in guiding_objs:
                        guiding_seq += [w for w in go['tokens']]
                    left_seq_length = len(
                        reversed_sentence[obj['index'] +
                                          1:]) + len(guiding_seq)
                    left_seq = [start_token]
                    left_seq += guiding_seq
                    left_seq += [
                        w for w in reversed_sentence[obj['index'] + 1:]
                    ]
                    left_seq += [end_token]
                    left_seq += [
                        pad_token for _ in range(max_len - left_seq_length)
                    ]
                    assert len(left_seq) == max_len + 2

                    left_seq_onehot = [
                        word_map.get(w, word_map[unk_token]) for w in left_seq
                    ]

                    label_left_splited[split].append({
                        'image_id':
                        image_id,
                        'seq':
                        left_seq_onehot,
                        'seq_length':
                        left_seq_length,
                        'guiding_objs':
                        guiding_objs,
                    })

    if not os.path.exists(lstm_dir):
        os.mkdir(lstm_dir)
    with open(os.path.join(lstm_dir, 'word_counter.json'), 'w') as fp:
        json.dump(word_counter, fp)
    with open(os.path.join(lstm_dir, 'word_map.json'), 'w') as fp:
        json.dump(word_map, fp)
    with open(os.path.join(lstm_dir, 'label_right.json'), 'w') as fp:
        json.dump(label_right_splited, fp)
    with open(os.path.join(lstm_dir, 'label_left.json'), 'w') as fp:
        json.dump(label_left_splited, fp)


if __name__ == '__main__':
    prepare_input_lstm()
