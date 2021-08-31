import json
import os

from loguru import logger
from tqdm import tqdm

data_dir = 'data/'
classifier_dir = os.path.join(data_dir, 'classifier')
dataset = 'coco'
coco_dir = os.path.join(data_dir, 'annotations_trainval2014/annotations')
coco_train_filename = 'instances_train2014.json'
coco_val_filename = 'instances_val2014.json'
split_path = os.path.join(data_dir, 'caption_datasets', 'dataset_coco.json')
feature_map_path = os.path.join(data_dir, 'feature_map.json')


def prepare_input_classifier():
    assert dataset in {'coco'}

    with open(os.path.join(coco_dir, coco_train_filename)) as fp:
        detection_data_train = json.load(fp)
    with open(os.path.join(coco_dir, coco_val_filename)) as fp:
        detection_data_val = json.load(fp)
    with open(split_path) as fp:
        split_data = json.load(fp)
    with open(feature_map_path, 'r') as fp:
        feature_map = json.load(fp)

    categories = []
    category_indices = {}

    for c in detection_data_train['categories']:
        category_indices[c['id']] = len(categories)
        categories.append(c['name'])

    assert len(categories) == 80

    if not os.path.exists(classifier_dir):
        os.mkdir(classifier_dir)
    with open(os.path.join(classifier_dir, 'category_names.json'), 'w') as fp:
        json.dump(categories, fp)

    label_detection = {}
    category_count_detection = {}
    for category in categories:
        category_count_detection[category] = 0

    for annotation in tqdm(detection_data_train['annotations'] + \
                      detection_data_val['annotations']):
        image_id = str(annotation['image_id'])
        if not image_id in feature_map:
            logger.warning('Not found in features! id: {0}'.format(image_id))
        else:
            if image_id in label_detection:
                label_detection[image_id].append(annotation['category_id'])
            else:
                label_detection[image_id] = [annotation['category_id']]

    for image in label_detection:
        label = label_detection[image]
        label_detection[image] = [0 for _ in range(len(categories))]
        for label in label:
            label_detection[image][category_indices[label]] = 1
            category_count_detection[categories[category_indices[label]]] += 1

    label_detection = {'train': [], 'val': [], 'test': []}

    label_caption = {'train': [], 'val': [], 'test': []}

    filemap = {}

    category_count_caption = {}
    for category in categories:
        category_count_caption[category] = 0

    for item in tqdm(split_data['images']):
        image_id = str(item['cocoid'])
        if not image_id in feature_map:
            logger.warning('Not found in features! id: {0}'.format(image_id))
        else:
            split = item['split']
            if split == 'restval':
                split = 'train'
            assert split in ['train', 'val', 'test']

            label = [0 for _ in range(len(categories))]
            for sentence in item['sentences']:
                for word in sentence['tokens']:
                    if word in categories:
                        label[categories.index(word)] = 1
                        category_count_caption[word] += 1
                bigrams = list(zip(sentence['tokens'], sentence['tokens'][1:]))
                for bigram in bigrams:
                    word = bigram[0] + ' ' + bigram[1]
                    if word in categories:
                        label[categories.index(word)] = 1
                        category_count_caption[word] += 1

            label_caption[split].append({
                'image_id': image_id,
                'label': label,
            })
            filemap[image_id] = {
                'file_path': item['filepath'],
                'file_name': item['filename']
            }
            if image_id in label_detection:
                label_detection[split].append({
                    'image_id':
                    image_id,
                    'label':
                    label_detection[image_id],
                })
            else:
                logger.warning(
                    "Not found in coco detection! id: {0}".format(image_id))

    with open(os.path.join(classifier_dir, 'label_caption.json'), 'w') as fp:
        json.dump(label_caption, fp)

    with open(os.path.join(classifier_dir, 'label_detection.json'), 'w') as fp:
        json.dump(label_detection, fp)

    with open(os.path.join(classifier_dir, 'category_count_detection.json'),
              'w') as fp:
        json.dump(category_count_detection, fp)

    with open(os.path.join(classifier_dir, 'category_count_caption.json'),
              'w') as fp:
        json.dump(category_count_caption, fp)
    with open(os.path.join(data_dir, 'image_file_map.json'), 'w') as fp:
        json.dump(filemap, fp)


if __name__ == '__main__':
    prepare_input_classifier()
