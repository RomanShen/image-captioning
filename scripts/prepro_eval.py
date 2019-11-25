import json

dataset = json.load(open("/root/PycharmProjects/dataset_rsicd/dataset_rsicd.json", 'r'))
imgs = dataset['images']

cap_map = {}
img_map = {}
cap_list = []
img_list = []
for i, img in enumerate(imgs):
    if img['split'] == 'val':
        img_map['liscense'] = 1
        img_map['file_name'] = img['filename']
        img_map['id'] = img['imgid']
        img_list.append(img_map)
        img_map = {}
        for c in img['sentences']:
            cap_map['image_id'] = c['imgid']
            cap_map['id'] = c['sentid']
            cap_map['caption'] = c['raw']
            cap_list.append(cap_map)
            cap_map = {}

final_map = {}
final_map['info'] = {'description': 'this is a created file.'}
final_map['images'] = img_list
final_map['type'] = 'annotations'
final_map['licenses'] = 'noncommercial'
final_map['annotations'] = cap_list

result = "/root/PycharmProjects/dataset_rsicd/captions_val2014.json"
with open(result, 'w') as j:
    json.dump(final_map, j)
