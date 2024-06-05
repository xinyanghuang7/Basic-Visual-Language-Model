import json
import numpy as np
import yaml

def read_json(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def process(caption_data, image_path, start_ID=0):
    image_map = {}
    captions = {}
    for idx, D in enumerate(caption_data):
        # Debugging: Print the first few elements to understand the structure
        if idx < 5:
            print(f"Processing element {idx}: {D}")
        
        image_name = D.get("image_id", D.get("image"))
        caption = D.get("caption", D.get("conversations"))
        
        if isinstance(caption, list):
            # If caption is a list, randomly select one caption from the list
            selected_caption = np.random.choice(caption)
            if np.random.random() > 0.5:
                captions[idx + start_ID] = {"q": "这副图像描述了什么？<|extra_0|>", "a": selected_caption}
            else:
                captions[idx + start_ID] = {"q": "<|extra_0|>这幅图像中有什么？", "a": selected_caption}
        else:
            if isinstance(caption, list):
                if len(caption) > 2:
                    continue
                answer = caption[1]["value"]
                question = caption[0]["value"].replace("<image>", "<|extra_0|>").replace("\n", "")
                captions[idx + start_ID] = {"q": question, "a": [answer]}
            else:
                if np.random.random() > 0.5:
                    captions[idx + start_ID] = {"q": "这副图像描述了什么？<|extra_0|>", "a": caption}
                else:
                    captions[idx + start_ID] = {"q": "<|extra_0|>这幅图像中有什么？", "a": caption}
        
        image_map[image_name] = {"image_file": image_name, "ID": idx + start_ID, "path": image_path}

    return image_map, captions

def main():
    config = yaml.load(open("config.yaml", 'r', encoding="utf-8"), Loader=yaml.FullLoader)
    
    caption_data_ali = read_json(config["ali_image_path_label_file"])
    caption_data_chat = read_json(config["coco_image_path_label_file"])
    
    print("First few elements of caption_data_ali:")
    for i in range(min(5, len(caption_data_ali))):
        print(caption_data_ali[i])
    
    image_map_ali, captions_ali = process(caption_data_ali[:50000], config["ali_image_path"])
    print(f"Processed {len(image_map_ali)} images from ali dataset.")
    
    image_map_chat, captions_chat = process(caption_data_chat, config["coco_image_path"], len(image_map_ali))
    print(f"Processed {len(image_map_chat)} images from coco dataset.")
    
    image_map = {**image_map_ali, **image_map_chat}
    captions = {**captions_ali, **captions_chat}
    
    print(f"Total images processed: {len(image_map)}")
    
    with open(config["output_image_map"], 'w', encoding="utf-8") as f:
        json.dump(image_map, f, ensure_ascii=False, indent=4)
    
    with open(config["output_captions"], 'w', encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
