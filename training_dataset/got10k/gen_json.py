from os.path import join
import json
from typing import Dict
import os
from got10k.datasets import GOT10k

input_dir = 'got10k'
output_dir = "./"

for subset in ['val', 'train']:
    js = dict()
    dataset = GOT10k(root_dir=input_dir, subset=subset, return_meta=True)
    n_videos = len(dataset)
    
    for s, (img_files, anno, meta) in enumerate(dataset):
        seq_name = dataset.seq_names[s]
        video_crop_base_path = join(subset, seq_name)

        js[video_crop_base_path] = dict()

        for idx, img_file in enumerate(img_files):
            if meta['absence'][idx] == 1:
                continue
            rect = anno[idx, :]
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
            js[video_crop_base_path]['{:02d}'.format(idx)] = {'000000': bbox}

    print('save json (dataset), please wait 20 seconds~')
    output_path = os.path.join(output_dir, '{}.json'.format(subset))
    json.dump(js, open(output_path, 'w'), indent=4, sort_keys=True)
    print('done!')