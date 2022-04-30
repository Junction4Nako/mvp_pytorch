import torch
import json, os
from oscar.modeling.modeling_vlbert import BiImageBertForRetrieval, BiImageBertForVQA, BiImageBertRep, BiBertImgForMLM
from transformers.pytorch_transformers import BertTokenizer
from tools.transform_utils import build_transforms
from PIL import Image
import numpy as np


model_name_mapping = {
    'mlm':       BiBertImgForMLM,
    'embedding': BiImageBertRep
}

def cpu_numpy(input_tensor):
    return input_tensor.cpu().numpy()

class InferencePipeline(object):
    def __init__(self, model_name, model_path, object_detector_path, od_config_dir='tools/configs/', parser_path='tools/spice', id2phrase='datasets/mvp/id2phrase_new.json', max_seq_length=30, max_img_seq_length=50,
                max_tag_length=20, max_phrases=5, device='cuda:0'):
        print("creating a MVPTR inference pipeline...")
        print('loading a {} model from {}'.format(model_name, model_path))
        assert model_name in model_name_mapping, "{} not in valida names: {}".format(model_name, ', '.join(model_name_mapping.keys()))
        model_class = model_name_mapping[model_name]
        self.device = torch.device(device)
        self.model = model_class.from_pretrained(model_path).to(self.device)
        self.model_name = model_name
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        print('loading the torchscripted object detector from {}'.format(object_detector_path))
        self.od_model = torch.jit.load(object_detector_path).to(self.device)
        self.od_model.eval()
        print('loading the object detector configs in {}'.format(od_config_dir))
        try:
            self.transform_cfg = json.load(open(os.path.join(od_config_dir, 'vinvl_transform.json'), 'r'))
            self.img_transform = build_transforms(self.transform_cfg)
            label_map = json.load(open(os.path.join(od_config_dir,'VG-SGG-dicts-vgoi6-clipped.json')))['label_to_idx']
            self.label_map = {v:k for k,v in label_map.items()}
        except:
            print('object detector configs not found in {}, it is supposed be tools/configs/, please specify the correct path \
                if you are in another directory'.format(od_config_dir))

        print('check the scene graph parser in {}'.format(parser_path))
        assert os.path.exists(os.path.join(parser_path, 'spice-1.0.jar')), 'SPICE parser not found in {}, \
            it is supposed to be installed in ./tools/spice by tools/prepare_spice.sh, please specify the correct path \
                if you are in another directory'.format(parser_path)
        self.parser_path = parser_path
        self.phrase_cache_dir = os.path.expanduser('~/.cache/mvptr')

        print('loading phrases to id mapping in {}'.format(id2phrase))
        try:
            self.id2sg = json.load(open(id2phrase, 'r'))
        except:
            print('id2phrase mapping not found in {}, it is supposed be datasets/mvp/id2phrase_new.json, please specify the correct path \
                if you are in another directory'.format(id2phrase))
            raise ValueError
        self.sg2id = {tuple(v):int(k) for k,v in self.id2sg.items()}
        self.phrase_vocab_size = len(self.sg2id)

        self.max_seq_length = max_seq_length
        self.max_img_seq_len = max_img_seq_length
        self.max_tag_length = max_tag_length
        self.max_phrases = max_phrases

    def preprocess_img(self, img_path):
        img = Image.open(os.path.join(img_path))
        img = img.convert('RGB')
        img_feat = self.img_transform(img, None)[0]
        img_h, img_w = img_feat.shape[-2:]
        with torch.no_grad():
            img_feat = img_feat.unsqueeze(0).to(self.od_model.device)
            bboxes, od_tags, obj_feats = [cpu_numpy(t) for t in self.od_model(img_feat)]
        # od_tags = ' '.join([label_map[ot] for ot in od_tags[0]])
        od_tags = [self.label_map[ot] for ot in od_tags[0]]
        bboxes = np.copy(bboxes[0])
        obj_feats = obj_feats[0]
        bboxes[:, 0] = bboxes[:, 0]/img_w
        bboxes[:, 2] = bboxes[:, 2]/img_w
        bboxes[:, 1] = bboxes[:, 1]/img_h
        bboxes[:, 3] = bboxes[:, 3]/img_h
        box_width = np.expand_dims(bboxes[:, 2] - bboxes[:, 0], 1)
        box_height = np.expand_dims(bboxes[:, 3] - bboxes[:, 1], 1)
        full_feats = np.concatenate([obj_feats, bboxes, box_width, box_height], axis=1)
        if not full_feats.flags['WRITEABLE']:
            full_feats = np.copy(full_feats)
        t_features = torch.from_numpy(full_feats)
        return t_features, od_tags

    def phrase_extract(self, text):
        os.makedirs(self.phrase_cache_dir, exist_ok=True)
        tmp_input = [{'image_id':'null', 'test': text.lower(), 'refs':[]}]
        with open(os.path.join(self.phrase_cache_dir, 'tmp_input.json'), 'w') as wf:
            json.dump(tmp_input, wf)
        os.system('export PATH=$PATH:$JAVA_HOME/bin/')
        os.system('java -Xmx8G -jar {} {} -out {} -threads 20 -detailed -silent'.format(os.path.join(self.parser_path, 'spice-1.0.jar'), \
            os.path.join(self.phrase_cache_dir, 'tmp_input.json'), os.path.join(self.phrase_cache_dir, 'tmp_output.json')))
        tmp_output = json.load(open(os.path.join(self.phrase_cache_dir, 'tmp_output.json'), 'r'))[0]['test_tuples']
        phrase_nodes = [tuple(t['tuple']) for t in tmp_output]
        phrase_nodes = [self.sg2id[t] for t in phrase_nodes if t in self.sg2id]
        return phrase_nodes

    def inference(self, img_path, text):
        img_feat, od_tags = self.preprocess_img(img_path)
        text_b = ' '.join(od_tags)
        phrases = self.phrase_extract(text)
        example = self.tensorize_example(text, img_feat, text_b, phrases)
        input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, img_feat = [t.unsqueeze(0).to(self.device) for t in example]
        with torch.no_grad():
            if self.model_name == 'mlm':
                mlm_score, seq_score = self.model(input_ids_a=input_ids_a, \
                    attention_mask_a=input_mask_a, token_type_ids_a=segment_ids_a, input_ids_b=input_ids_b, \
                    attention_mask_b=input_mask_b, token_type_ids_b=segment_ids_b, img_feats=img_feat, \
                    max_tag_length=self.max_tag_length)
                mlm_pred = mlm_score.argmax(dim=-1).squeeze().cpu().numpy().tolist()
                new_res = self.tokenizer.convert_ids_to_tokens(mlm_pred)
                new_res = new_res if isinstance(new_res, str) else ','.join(new_res)
                return 'recovered [MASK] tokens: {}'.format(self.tokenizer.convert_ids_to_tokens(mlm_pred))
            elif self.model_name == 'embedding':
                sequence_output, pooled_output, single_stream_output = self.model(input_ids_a=input_ids_a, \
                    attention_mask_a=input_mask_a, token_type_ids_a=segment_ids_a, input_ids_b=input_ids_b, \
                    attention_mask_b=input_mask_b, token_type_ids_b=segment_ids_b, img_feats=img_feat, \
                    max_tag_length=self.max_tag_length)
                res = {'cross_modal_output':sequence_output, 'pooled_output': pooled_output, \
                    'vis_encoder_output': single_stream_output[1], 'txt_encoder_output': single_stream_output[0]}
                res = {k:v.squeeze() for k,v in res.items()}
            else:
                raise ValueError

    def tensorize_example(self, text_a, img_feat, text_b=None, phrase_nodes=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        num_extra_tokens = 2
        num_phrases = self.max_phrases
        if len(tokens_a) > self.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.max_seq_length - num_extra_tokens)]

        if len(phrase_nodes) >= num_phrases+self.max_seq_length-2-len(tokens_a):
            phrase_nodes = phrase_nodes[:(num_phrases+self.max_seq_length-2-len(tokens_a))]

        seq_tokens_a = [self.tokenizer.cls_token] + tokens_a #  + [self.tokenizer.sep_token]
        phrase_index = [len(seq_tokens_a), len(seq_tokens_a)+len(phrase_nodes)]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(seq_tokens_a) + phrase_nodes + [self.tokenizer.vocab[self.tokenizer.sep_token]]
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + len(phrase_nodes) + 1)
        seq_a_len = len(input_ids_a)
        input_mask_a = [1] * len(input_ids_a)

        tokens_b = self.tokenizer.tokenize(text_b)
        if len(tokens_b) > self.max_tag_length - 2:
            # num_left_tokens = max(0, self.max_seq_len - len(tokens_b) - 2) # to avoid -1
            # assert(num_left_tokens >= 0)
            tokens_b = tokens_b[: (self.max_tag_length - 2)]
        seq_tokens_b = ['[CLS]'] + tokens_b + [self.tokenizer.sep_token]
        input_ids_b = self.tokenizer.convert_tokens_to_ids(seq_tokens_b)
        segment_ids_b = [sequence_b_segment_id] * len(seq_tokens_b)
        input_mask_b = [1] * len(input_ids_b)
        seq_b_len = len(input_ids_b)

        seq_len_a = len(input_ids_a)
        tmp_max_seq_len = self.max_seq_length + self.max_phrases
        seq_padding_len_a = tmp_max_seq_len - seq_len_a
        input_ids_a += seq_padding_len_a * [0,]
        input_mask_a += seq_padding_len_a * [0,]
        segment_ids_a += seq_padding_len_a * [pad_token_segment_id,]

        seq_padding_len_b = self.max_tag_length - seq_b_len
        input_ids_b += seq_padding_len_b * [0, ]
        input_mask_b += seq_padding_len_b * [0, ]
        segment_ids_b += seq_padding_len_b * [pad_token_segment_id, ]

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
            input_mask_b += [1]*self.max_img_seq_len
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            input_mask_b += [1]*img_len + [0]*img_padding_len
        image_start_index = len(input_ids_a) # input_ids_a here for the concated sequence
        image_end_index = image_start_index + img_len
        img_index = [image_start_index, image_end_index]

        input_ids_a = torch.tensor(input_ids_a, dtype=torch.long)
        input_mask_a = torch.tensor(input_mask_a, dtype=torch.long)
        segment_ids_a = torch.tensor(segment_ids_a, dtype=torch.long)
        input_ids_b = torch.tensor(input_ids_b, dtype=torch.long)
        input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)
        segment_ids_b = torch.tensor(segment_ids_b, dtype=torch.long)
        phrase_index = torch.tensor(phrase_index, dtype=torch.long)
        image_index = torch.tensor(img_index, dtype=torch.long)
        return (input_ids_a, input_mask_a, segment_ids_a, input_ids_b,
            input_mask_b, segment_ids_b, img_feat)


if __name__=='__main__':
    inference = InferencePipeline('mlm', '/opt/tiger/mvp/pretrained_models/base/', '/opt/tiger/mvp/tools/od_model.pt',
    od_config_dir='/opt/tiger/mvp/tools/configs/', parser_path='/opt/tiger/mvp/tools/spice', id2phrase='/opt/tiger/mvp/datasets/mvp/id2phrase_new.json')
    res = inference.inference('/opt/tiger/tmp_dir/test_mvp/coco_test.jpg', 'two [MASK] are playing on a ground')
    print(res)
    # print({k:v.shape for k,v in res.items()})
