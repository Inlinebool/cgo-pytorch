import argparse
import json
import os

import torch
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import trainer
from datasets import ClassifierDataset, LSTMDataset
from models import LanguageModel


def loss_fn_lstm(preds, label):
    label_seq, label_seq_length = label
    label_seq_length += 1
    # remove <start>
    label_seq = label_seq[:, 1:]
    label_seq_length = label_seq_length.cpu().to(torch.int64)
    label_packed = pack_padded_sequence(label_seq,
                                        label_seq_length,
                                        batch_first=True,
                                        enforce_sorted=False)
    return nn.CrossEntropyLoss()(preds, label_packed.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train language models.')

    parser.add_argument('direction', type=str)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data')
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
    parser.add_argument('--label_filename', type=str, default='')
    parser.add_argument('--word_map_filename',
                        type=str,
                        default='word_map.json')
    parser.add_argument('--label_caption_filename',
                        type=str,
                        default='label_caption.json')

    parser.add_argument('--att_dim', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--img_embed_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--decay_every', type=int, default=20)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--load_param', type=str, default='')
    parser.add_argument('--no-logfile', dest='logfile', action='store_false')
    parser.set_defaults(log_file=True)

    args = parser.parse_args()

    if args.direction == 'left':
        model_name = 'lstm-left' if not args.model_name else args.model_name
        label_filename = 'label_left.json'
    elif args.direction == 'right':
        model_name = 'lstm-right' if not args.model_name else args.model_name
        label_filename = 'label_right.json'
    else:
        logger.error('direction has to be either left or right!')
        exit(1)
    if args.label_filename:
        label_filename = args.label_filename

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if args.log_file:
        logger.add(os.path.join(
            args.model_dir,
            '{0}_log.txt',
        ).format(model_name),
                   mode='w')

    model_path = os.path.join(args.model_dir, model_name)
    feature_path = os.path.join(args.data_dir, args.feature_filename)
    featuremap_path = os.path.join(args.data_dir, args.featuremap_filename)
    label_path = os.path.join(args.data_dir, args.lstm_data_dir,
                              label_filename)
    word_map_path = os.path.join(args.data_dir, args.lstm_data_dir,
                                 args.word_map_filename)
    cap_detection_label_path = os.path.join(args.data_dir,
                                            args.classifier_data_dir,
                                            args.label_caption_filename)

    if args.load_param:
        with open(args.load_opt, 'r') as fp:
            params = json.load(fp)
    else:
        params = {
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'decay_every': args.decay_every,
            'decay_rate': args.decay_rate,
            'lr': args.lr,
        }
    with open(
            os.path.join(args.model_dir, '{0}_param.json'.format(model_name)),
            'w') as fp:
        json.dump(params, fp)
    with open(os.path.join(args.model_dir, '{0}_args.json'.format(model_name)),
              'w') as fp:
        json.dump(vars(args), fp)
    with open(word_map_path, 'r') as fp:
        word_map = json.load(fp)

    reversed_word_map = [(word_map[k], k) for k in word_map]
    reversed_word_map = sorted(reversed_word_map)
    reversed_word_map = [x[1] for x in reversed_word_map]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Training {0}...".format(model_name))

    train_dataset = LSTMDataset(feature_path, label_path, featuremap_path,
                                'train')
    val_dataset = LSTMDataset(feature_path, label_path, featuremap_path, 'val')
    val_cap_detection = ClassifierDataset(feature_path,
                                          cap_detection_label_path,
                                          featuremap_path, 'val')

    model = LanguageModel(len(word_map), args.embed_dim, args.hidden_dim,
                          (36, 2048), args.img_embed_dim, args.att_dim, device)

    if args.direction == 'left':
        trainer.train_val_loss(model=model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               num_workers=args.num_workers,
                               params=params,
                               loss_fn=loss_fn_lstm,
                               model_save_path=model_path,
                               save_every=5,
                               device=device)
    else:
        trainer.train_val_meteor(model=model,
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 val_cap_dataset=val_cap_detection,
                                 word_map=word_map,
                                 reversed_word_map=reversed_word_map,
                                 num_workers=args.num_workers,
                                 params=params,
                                 loss_fn=loss_fn_lstm,
                                 model_save_path=model_path,
                                 save_every=5,
                                 device=device)
