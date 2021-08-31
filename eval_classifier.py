import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ClassifierDataset


def init_result(categories):
    eval_result = {}
    for c in categories:
        eval_result[c] = {
            'true_pos': 0,
            'false_pos': 0,
            'true_neg': 0,
            'false_neg': 0,
        }
    return eval_result


def compute_metrics(result):
    for c in result:
        all_pos = result[c]['true_pos'] + result[c]['false_pos']
        all_relevant = result[c]['true_pos'] + result[c]['false_neg']
        result[c]['recall'] = result[c][
            'true_pos'] / all_relevant if all_relevant else 0
        result[c][
            'precision'] = result[c]['true_pos'] / all_pos if all_pos else 0
        result[c]['f1'] = 2 * result[c]['true_pos'] / (
            2 * result[c]['true_pos'] + result[c]['false_pos'] +
            result[c]['false_neg'])


def hit_fn_topk(k: int):
    return lambda predict: torch.argsort(predict, descending=True)[0:k].view(
        -1).tolist()


def hit_fn_thresh(thresh: float):
    return lambda predict: torch.nonzero(torch.ge(predict, thresh)).view(
        -1).tolist()


def hit_to_names(hit, categories):
    names = [categories[x] for x in hit]
    return names


def evaluate_model(model, test, category_names, hit_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        data_loader = DataLoader(test, shuffle=True)
        eval_result = init_result(category_names)
        model.eval()
        for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = [x.to(device) for x in batch['inputs']]
            label = [x.to(device) for x in batch['label']]
            predict = model(inputs)
            predict = torch.sigmoid(predict).squeeze()
            hit = hit_fn(predict)
            # label_hit = hit_fn_thresh(0.9)(label)
            # predicted_categories = hit_to_names(hit, category_names)
            # label_categories = hit_to_names(label_hit, category_names)
            for c in category_names:
                if label[category_names.index(c)] == 1:
                    if category_names.index(c) in hit:
                        eval_result[c]['true_pos'] += 1
                    else:
                        eval_result[c]['false_neg'] += 1
                else:
                    if category_names.index(c) in hit:
                        eval_result[c]['false_pos'] += 1
                    else:
                        eval_result[c]['true_neg'] += 1

        compute_metrics(eval_result)
        return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate classifier.')

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
    parser.add_argument('--category_name_filename',
                        type=str,
                        default='category_names.json')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name)
    feature_path = os.path.join(args.data_dir, args.feature_filename)
    featuremap_path = os.path.join(args.data_dir, args.featuremap_filename)
    label_path = os.path.join(args.data_dir, args.classifier_data_dir,
                              args.label_filename)
    category_name_path = os.path.join(args.data_dir, args.classifier_data_dir,
                                      args.category_name_filename)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('{0}.pkl'.format(model_path)).to(device)
    test = ClassifierDataset(feature_path, label_path, featuremap_path, 'test')
    with open(category_name_path, 'r') as fp:
        category_names = json.load(fp)

    results_topk = evaluate_model(model, test, category_names,
                                  hit_fn_topk(args.top_k))
    topk_filename = 'results_{0}_top{1}.json'.format(args.model_name,
                                                     args.top_k)
    with open(os.path.join(args.result_dir, topk_filename), 'w') as fp:
        json.dump(results_topk, fp, indent=4)

    results_thresh = evaluate_model(model, test, category_names,
                                    hit_fn_thresh(args.thresh))
    thresh_filename = 'results_{0}_thresh{1}.json'.format(
        args.model_name, args.thresh)
    with open(os.path.join(args.result_dir, thresh_filename), 'w') as fp:
        json.dump(results_thresh, fp, indent=4)
