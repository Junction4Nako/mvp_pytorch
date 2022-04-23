import torch, json
import argparse
from PIL import Image
from tools.transform_utils import build_transforms
import os, tqdm
import numpy as np
import torchvision
import base64
import os.path as op

def cpu_numpy(input_tensor):
    return input_tensor.cpu().numpy()

def tsv_writer(values, tsv_file, sep='\t'):
    # mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    imageid2index = {}
    i = 0
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type. 
            imageid2index[value[0]] = i
            i += 1
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)
    with open(op.join(op.dirname(tsv_file), 'imageid2idx.json'), 'w') as wf:
        json.dump(imageid2index, wf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vinvl_od_path', type=str, default=None, help='path to the torchscript vinvl object detector')
    parser.add_argument('--image_dir', type=str, default=None, help='image directory to process')
    parser.add_argument('--target_dir', type=str, default=None, help='the target path to store the results')
    parser.add_argument('--od_config_dir', type=str, default=None, help='the directory for object detector configs')
    parser.add_argument('--device', type=str, default=None, help='the device to perform inference')
    args = parser.parse_args()

    if args.target_dir is None:
        args.target_dir = args.image_dir

    # load the vinvl object detector configs
    try:
        print('loading configs from the directory: {}'.format(args.od_config_dir))
        transform_cfg = json.load(open(os.path.join(args.od_config_dir, 'vinvl_transform.json'), 'r'))
        label_map = json.load(open(os.path.join(args.od_config_dir, 'VG-SGG-dicts-vgoi6-clipped.json'), 'r'))['label_to_idx']
        label_map = {v:k for k,v in label_map.items()}
    except:
        print('config files not found, please check https://github.com/Junction4Nako/mvp_pytorch/tree/master/tools/configs/')
        return None
    
    # make the image transformations
    img_transform = build_transforms(transform_cfg)

    # load the object detector model
    device = torch.device(args.device)
    try:
        print('loading torchscripted object detector from {}'.format(args.vinvl_od_path))
        od_model = torch.jit.load(args.vinvl_od_path).to(device)
        od_model.eval()
    except:
        print('fail to load the obejct detector torchscript model, please check https://github.com/Junction4Nako/mvp_pytorch/')
        return None

    valid_suffix = ['.jpg', '.png', '.jpeg', '.webp']
    image_filenames = os.listdir(args.image_dir)
    image_filenames = [fn for fn in image_filenames if any([fn.lower().endswith(suffix) for suffix in valid_suffix])]
    print('{} valid images found in {}'.format(len(image_filenames), args.image_dir))

    # processing the images
    def gen_rows():
        i = 0
        img2idx = {}
        for img_fn in tqdm.tqdm(image_filenames):
            img = Image.open(os.path.join(args.image_dir, img_fn))
            img = img.convert('RGB')
            raw_fn = '.'.join(img_fn.split('.')[:-1]) # remove the suffix
            img2idx[raw_fn] = i
            i += 1
            img_feat = img_transform(img, None)[0]
            img_h, img_w = img_feat.shape[-2:]
            with torch.no_grad():
                img_feat = img_feat.unsqueeze(0).to(od_model.device)
                bboxes, od_tags, obj_feats = [cpu_numpy(t) for t in od_model(img_feat)]
            # od_tags = ' '.join([label_map[ot] for ot in od_tags[0]])
            od_tags = [label_map[ot] for ot in od_tags[0]]
            bboxes = np.copy(bboxes[0])
            obj_feats = obj_feats[0]
            bboxes[:, 0] = bboxes[:, 0]/img_w
            bboxes[:, 2] = bboxes[:, 2]/img_w
            bboxes[:, 1] = bboxes[:, 1]/img_h
            bboxes[:, 3] = bboxes[:, 3]/img_h
            box_width = np.expand_dims(bboxes[:, 2] - bboxes[:, 0], 1)
            box_height = np.expand_dims(bboxes[:, 3] - bboxes[:, 1], 1)
            full_feats = np.concatenate([obj_feats, bboxes, box_width, box_height], axis=1)
            encoded_feat = base64.b64encode(full_feats).decode('utf-8')
            yield raw_fn, json.dumps({'feature': encoded_feat, 'predictions': od_tags})
    
    tsv_writer(gen_rows(), os.path.join(args.target_dir, 'predictions.tsv'))
    # with open(os.path.join(args.target_dir, 'imageid2idx.json'), 'w') as wf:
    #     json.dump(img2idx, wf)
    

if __name__=='__main__':
    main()
