import argparse
import json
import os

from PIL import Image

parser = argparse.ArgumentParser(description='Train object classifier.')

parser.add_argument('image_id', type=str)
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--model_name', type=str, default='classifier')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--classifier_data_dir', type=str, default='classifier')
parser.add_argument('--feature_filename',
                    type=str,
                    default='image_features.h5')
parser.add_argument('--indexmap_filename', type=str, default='indexmap.json')
parser.add_argument('--filemap_filename', type=str, default='filemap.json')
parser.add_argument('--label_filename',
                    type=str,
                    default='label_detection_splited.json')

args = parser.parse_args()

model_path = os.path.join(args.model_dir, args.model_name)
feature_path = os.path.join(args.data_dir, args.feature_filename)
indexmap_path = os.path.join(args.data_dir, args.indexmap_filename)
label_path = os.path.join(args.data_dir, args.classifier_data_dir,
                          args.label_filename)
filemap_path = os.path.join(args.data_dir, args.filemap_filename)

with open(filemap_path, 'r') as fp:
    filemap = json.load(fp)

image_path = os.path.join(args.data_dir, filemap[args.image_id]['file_path'],
                          filemap[args.image_id]['file_name'])

image = Image.open(image_path)
image.show()
