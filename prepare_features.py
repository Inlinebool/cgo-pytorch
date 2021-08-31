"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format. 
Hierarchy of HDF5 file:
[
    image_id: { 
        'features': num_boxes x 2048 array of features
        'bboxes': num_boxes x 4 array of bounding boxes 
    }
]
"""

import base64
import csv
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

csv.field_size_limit(2**31 - 1)

FIELDNAMES = [
    'image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'
]
data_dir = 'data/'
trainval_dir = os.path.join(data_dir, 'trainval_36')
infile = os.path.join(trainval_dir,
                      'trainval_resnet101_faster_rcnn_genome_36.tsv')
feature_file = os.path.join(data_dir, 'image_features.h5')
indexmap = {}

feature_length = 2048
num_fixed_boxes = 36
num_images = 123287

if __name__ == '__main__':
    h_feature = h5py.File(feature_file, "w")
    features = h_feature.create_dataset('image_feature',
                                        shape=(num_images, num_fixed_boxes,
                                               feature_length),
                                        dtype='<f4')

    print("reading tsv...")
    with open(infile, newline='') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file,
                                delimiter='\t',
                                fieldnames=FIELDNAMES)
        image_count = 0
        for item in tqdm(reader, total=123287):
            num_boxes = int(item['num_boxes'])
            image_id = str(item['image_id'])
            image_feature = np.frombuffer(base64.b64decode(item['features']),
                                          dtype=np.float32).reshape(
                                              num_boxes, -1)
            assert image_feature.shape == (num_fixed_boxes, feature_length)
            indexmap[image_id] = image_count
            features[image_count, :, :] = image_feature
            image_count += 1

    h_feature.close()
    with open(os.path.join(data_dir, 'feature_map.json'), 'w') as fp:
        json.dump(indexmap, fp)
    print("done!")
