# MVPTR step by step

This is a detailed instruction on how to use MVPTR as the backbone for datasets that are not involved in our experiments.

### Input Preprocess

MVPTR is mainly designed for the input of image-text pairs, for image-only or text-only inputs, you may use the uni-modal encoders of MVPTR.

#### Object Detection

On the vision side, we use the VinVL object detector to extract object-level features (region features and object tags) of images. 

1.  Before you start to extract features by yourself, please check [here](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) to see if the image features of your target dataset are provided by VinVL;

2.  Please refer to the "VinVL Feature extraction" part in [scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark) to extract features from images by yourself:

   - Follow [here](https://github.com/microsoft/scene_graph_benchmark/blob/main/INSTALL.md) to install the environment of VinVL object detector;

   - Download the pre-trained object detector and label map;

   - Extract the tsv-format features from images in your data directory;

   - Each line in the tsv file contains the extracted object tags and base64-encoded dense features, decode it into numpy arrays, we provide a simple example here
     
     ```python
     import numpy as np
     import base64, json
     from oscar.utils.tsv_file import TSVFile
     
     tsv_file = TSVFile('YOUR_RESULT_TSV_FILENAME')
     img2size = json.load(open('YOUR_IMAGE_TO_SIZE_FILENAME'))
     line = tsv_file[0] # you can iterate through all lines
     img_id = line[0]
     img_w, img_h = img2size[img_id] # width and height of the image
     obj_info = json.loads(line[1])
     for obj in obj_info['objects']:
         od_tags = obj['class'] # object tags
         obj_feat = obj['feature'] # region features (2048-dim)
         obj_feat = np.frombuffer(base64.b64decode(obj_feat), dtype=np.float32)
         # augment obj feature with its position
         box = obj['rect']
         # normalized box coordinates
         pos_feat = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
         # normalized box width and height
         pos_feat.extend([pos_feat[2]-pos_feat[0], pos_feat[3]-pos_feat[1]])
         pos_feat = np.array(pos_feat, dtype=float32) # ensure numpy.float32
         input_obj_feat = np.concatenate([obj_feat, pos_feat]) # 2054-dim
     ```

     You can store the processed features and tags in the tsv-format like COCO (encode with base64) or the pt-format like flickr30k (image_id: feature mapping).

#### Phrase Extraction

To extract phrases in the text, we utilize [SPICE](https://github.com/peteanderson80/SPICE) to extract tuples in scene graphs as phrases:

- Prepare the SPICE parser from [here](https://panderson.me/images/SPICE-1.0.zip)

- Unzip this file, download Stanford CoreNLP using the included download script

- Prepare your captions in a json file like:

  ```json
  [{"image_id": "1406296515.jpg#2r5c", "test": "a girl is touring germany.", "refs": []}, {"image_id": "1406296515.jpg#2r1e", "test": "the girl is outside.", "refs": []}]
  ```

- Running the following command to extract phrases from captions:

  ```bash
  java -Xmx8G -jar spice-1.0.jar YOUR_JSON -out YOUR_OURPUT -threads 100 -detailed
  ```

- Please check the phrase format in the [id2phrase.json](https://github.com/Junction4Nako/mvp_pytorch/blob/master/datasets/mvp/id2phrase_new.json) json file (Tuple[str]).

#### Tensorization & Encode

We need to tensorize an image-text pair to fit the model input (this procedure is similar in different tasks), you can refer to the dataset definition in run scripts, we provide an example here:

```python
from transformers.pytorch_transformers import BertTokenizer

max_seq = YOUR_MAX_STRING_SETTING
max_tag = YOUR_MAX_TAGS_SETTING
max_img_seq = YOUR_MAX_IMG_SETTING
tokenizer = BertTokenizer.from_pretrained('CHECKPOINT_DIR')
id2phrase = json.load(open('./datasets/mvp/id2phrase_new.json', 'r')) # id2phrase mapping
phrase2id = {tuple(v):int(k) for k,v in id2phrase.items()} # phrase tuple to id
phrase_ids = [phrase2id[tuple(t)] for t in phrases if tuple(t) in phrase2id] # phrases: the extracted phrases of your text
tokens_a = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']  # text: your input text
input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a) + phrase_ids
segment_ids_a = [0] * len(max_seq) # segment embedding
attention_mask_a = [1] * len(input_ids_a) + [0] * (max_seq - len(input_ids_a))# attention mask
# Padding to maximal length
input_ids_a += tokenizer.convert_tokens_to_ids(['[PAD]']*(max_seq - len(input_ids_a)))

tokens_b = tokenizer.tokenize(object_tags) # a space-separated object tags string
input_ids_b = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_b + ['[SEP]'])
segment_ids_b = [1] * len(max_tag) # segment embedding
attention_mask_b = [1] * len(input_ids_b) + [0] * (max_tag - len(input_ids_b)) # attention mask
# Padding to maximal length
input_ids_a += tokenizer.convert_tokens_to_ids(['[PAD]']*(max_tag - len(input_ids_b)))

img_len = img_feat.shape[0] # img_feat is the [num_objects, 2054] feature
if img_len > max_img_seq:
    # truncating
    img_feat = img_feat[0 : max_img_seq, :]
    img_len = img_feat.shape[0]
    img_padding_len = 0
    input_mask_b += [1]*self.max_img_seq
else:
    # padding
    img_padding_len = max_img_seq - img_len
    padding_matrix = torch.zeros((img_padding, img_feat.shape[1]))
    img_feat = torch.cat((img_feat, padding_matrix), 0)
    input_mask_b += [1]*img_len + [0]*img_padding_len
```

In this example, we ignore the situation that the string is too long and needs to be truncated.

To encode the representation of the image-text pair, please check the BiImageBertRep class in [modeling_vlbert](https://github.com/Junction4Nako/mvp_pytorch/blob/master/oscar/modeling/modeling_vlbert.py), you can build your model on top of it or use it to extract representations. 

