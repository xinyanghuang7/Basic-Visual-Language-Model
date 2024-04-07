import json
import numpy as np
from argparse import ArgumentParser
import yaml

def read_json(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    return data

def process(caption_data, image_path, start_ID = 0):
    image_map = {}
    captions = {}
    for idx, D in enumerate(caption_data):
        image_name = D["image_id"]
        caption = D["caption"]
  
        image_map[image_name] = {"iamge_file":image_name, "ID":idx+start_ID, "path":image_path}
        if np.random.random() > 0.5:
            captions[idx+start_ID] = {"q":"这副图像描述了什么？<|extra_0|>", "a":caption}
        else:
            captions[idx+start_ID] = {"q":"<|extra_0|>这幅图像中有什么？", "a":caption}

    return image_map, captions

def process_chat(chat_data, image_path, start_ID = 0):
    image_map = {}
    captions = {}
    for idx, D in enumerate(chat_data):
        image_name = D["image"]
        caption = D["conversations"]
        if len(caption) > 2:
            continue
        answer = caption[1]["value"]
        question = caption[0]["value"].replace("<image>", "<|extra_0|>").replace("\n", "")
        image_map[image_name] = {"iamge_file":image_name, "ID":idx+start_ID, "path":image_path}
        captions[idx+start_ID] = {"q":question, "a":[answer]}

    return image_map, captions

def process_coco(caption_data, image_path, start_ID = 0):
    image_map = {}
    captions = {}
    for idx, D in enumerate(caption_data):
        image_name = D["image"]
        caption = D["conversations"]
  
        answer = caption[1]["value"]
        question = caption[0]["value"].replace("<image>", "<|extra_0|>").replace("\n", "")
        image_map[image_name] = {"iamge_file":image_name, "ID":idx+start_ID, "path":image_path}
        captions[idx+start_ID] = {"q":question, "a":[answer]}

    return image_map, captions

def main():
    config = yaml.load(open("config.yaml", 'r', encoding="utf-8"), Loader=yaml.FullLoader)
 

    caption_data_ali = read_json(config["ali_image_path_label_file"])
    caption_data_chat = read_json(config["coco_image_path_label_file"])
    image_map_ali, captions_ali = process(caption_data_ali[:50000], config["ali_image_path"])
    print(len(image_map_ali))
    image_map_chat, captions_chat = process_coco(caption_data_chat, config["coco_image_path"], len(image_map_ali))
    image_map = image_map_ali
    image_map.update(image_map_chat)
    captions = captions_ali
    captions.update(captions_chat)
    print(len(image_map))   
    with open(config["output_image_map"], 'w', encoding="utf-8") as f:
        f.write(json.dumps(image_map, ensure_ascii=False))
    with open(config["output_captions"], 'w', encoding="utf-8") as f:
        f.write(json.dumps(captions, ensure_ascii=False))


if __name__ == "__main__":
    main()