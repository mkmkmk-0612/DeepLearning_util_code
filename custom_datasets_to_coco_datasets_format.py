import json
import os
from collections import OrderedDict

c = 0

path = "C:\\Users\\user\\Desktop\\hair\\test_anno\\"
file_data = OrderedDict()
file_images = []
file_annotations = []
bbox = []

file_data["info"] = {"description": "", "url": "", "version": "", "year": 2020, "contributor": "", "data_created": "2021-01-12"}
file_data["licenses"] = [{"id": 1, "name": None, "url": None}]
file_data["categories"] = [{"id": 1, "name": "1", "supercategory": "None"}, {"id": 2, "name": "2", "supercategory": "None"},{"id": 3, "name": "3", "supercategory": "None"}, {"id": 4, "name": "4", "supercategory": "None"}]

for i, filename in enumerate(os.listdir(path)):
  with open(path+filename, 'r') as f:
    j = json.load(f)
  #print(json.dumps(j["labels"]))
  image_info = {
    "id": int(i),
    "file_name": filename,
    "width": int(j["resolution"][0]),
    "height": int(j["resolution"][1]),
    "date_captured": "2021/01/12",
    "license": int(1),
    "coco_url": "",
    "flickr_url": ""
  }
  file_images.append(image_info)

  for k in range(len(j["labels"])):
    bbox.clear()
    size = (j["labels"][k]["width"]) * (j["labels"][k]["height"])
    bbox.append(int(j["labels"][k]["x"]))
    bbox.append(int(j["labels"][k]["y"]))
    bbox.append(int(j["labels"][k]["width"]))
    bbox.append(int(j["labels"][k]["height"]))
    anno_info = {
      "id": int(c),
      "image_id": int(i),
      "category_id": int(j["labels"][k]["class"]),
      "iscrowd": 0,
      "area": float(size),
      "bbox": bbox,
      "segmentation": []
    }
    c = c+1
    file_annotations.append(anno_info)

file_data["images"] = file_images
file_data["annotations"] = file_annotations

print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

with open('./instances_test_hair.json', 'w') as output:
  json.dump(file_data, output)