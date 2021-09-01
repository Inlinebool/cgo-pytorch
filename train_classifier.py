import argparse
import json
import os

import torch
from loguru import logger
from torch import nn

import trainer
from datasets import ClassifierDataset
from models import ObjectClassifier


def loss_fn_classifier(preds, label):
    return nn.BCEWithLogitsLoss()(preds, label[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train object classifier.')

    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--model_name', type=str, default='classifier')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--classifier_data_dir',
                        type=str,
                        default='classifier')
    parser.add_argument('--feature_filename',
                        type=str,
                        default='image_features.h5')
    parser.add_argument('--featuremap_filename',
                        type=str,
                        default='feature_map.json')
    parser.add_argument('--label_filename',
                        type=str,
                        default='label_detection.json')

    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--linear_dims',
                        type=int,
                        nargs='+',
                        default=[512, 512, 512])

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--decay_every', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--load_param', type=str, default='')
    parser.add_argument('--no-logfile', dest='logfile', action='store_false')
    parser.set_defaults(log_file=True)

    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if args.log_file:
        logger.add(os.path.join(
            args.model_dir,
            '{0}_log.txt',
        ).format(args.model_name),
                   mode='w')

    model_path = os.path.join(args.model_dir, args.model_name)
    feature_path = os.path.join(args.data_dir, args.feature_filename)
    featuremap_path = os.path.join(args.data_dir, args.featuremap_filename)
    label_path = os.path.join(args.data_dir, args.classifier_data_dir,
                              args.label_filename)

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
            os.path.join(args.model_dir,
                         '{0}_param.json'.format(args.model_name)), 'w') as fp:
        json.dump(params, fp)
    with open(
            os.path.join(args.model_dir,
                         '{0}_args.json'.format(args.model_name)), 'w') as fp:
        json.dump(vars(args), fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Training object classifier...")

    train_dataset = ClassifierDataset(feature_path, label_path,
                                      featuremap_path, 'train')
    val_dataset = ClassifierDataset(feature_path, label_path, featuremap_path,
                                    'val')

    model = ObjectClassifier((36, 2048), args.att_dim, args.linear_dims, 80)

    trainer.train_val_loss(model=model,
                           train_dataset=train_dataset,
                           val_dataset=val_dataset,
                           num_workers=args.num_workers,
                           params=params,
                           loss_fn=loss_fn_classifier,
                           model_save_path=model_path,
                           save_every=5,
                           device=device)
