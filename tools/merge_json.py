import json
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default=None, help='the raw data file')
    args = parser.parse_args()
    dirname = os.path.dirname(args.input_data)
    data = json.load(open(args.input_data, 'r'))
    output_dir =os.path.join(dirname, 'tmp_process/output')
    all_output = {}
    imgid2count = defaultdict(int)
    for output_fn in os.listdir(output_dir):
        out = json.load(open(os.path.join(output_dir, output_fn), 'r'))
        for output_item in out:
            out_tuples = [tuple(p['tuple']) for p in output_item['test_tuples']]
            all_output[output_item['image_id']] = out_tuples
    
    for item in data:
        c = imgid2count[item['image_id']]
        imgid2count[item['image_id']] += 1
        item['phrases'] = all_output['{}_{}'.format(item['image_id'], c)]
    
    with open(os.path.join(dirname, 'processed_{}'.format(os.path.basename(args.input_data))), 'w') as wf:
        json.dump(data, wf)

if __name__=='__main__':
    main() 

