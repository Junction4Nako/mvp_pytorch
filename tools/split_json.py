import json
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default=None)
    args = parser.parse_args()
    dirname = os.path.dirname(args.input_json)
    assert(args.input_json.endswith('.json'))
    data = json.load(open(args.input_json, 'r'))
    num_len = len(data)
    num_splits = (num_len // 5000) + 1
    imgid2count = defaultdict(int)
    print('loaded {} texts and split to {} parts'.format(num_len, num_splits))
    for i in range(num_splits):
        to_save_data = []
        if i == num_splits - 1:
            end = len(data)
        else:
            end = (i+1)*5000
        for j in range(i*5000, end):
            item = data[j]
            c = imgid2count[item['image_id']]
            imgid2count[item['image_id']] += 1
            to_save_data.append({'image_id':'{}_{}'.format(item['image_id'], c), 'test': item['text'].lower(), 'refs':[]})
        with open(os.path.join(dirname, 'tmp_process', 'tmp_split{}.json'.format(i+1)), 'w') as wf:
            json.dump(to_save_data, wf)

if __name__=='__main__':
    main()
