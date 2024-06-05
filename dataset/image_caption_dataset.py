import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm

from qwen.qwen_generation_utils import make_context

def readJson(filePath):
    with open(filePath, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def data_collate(example, tokenizer, black_token_length):
    images = []
    captions = []
    labels = []
    max_length = np.max([len(e[1]) for e in example]) + 1
    for e in example:
        img, caption, L = e
        L = L + 1
        caption = caption + [tokenizer.eod_id]
        images.append(img)
        caption_labels = [-100]*(black_token_length + (len(caption)-L) - 1) + caption[-L:] + [-100]*(max_length - len(caption))
        captions.append(torch.tensor(caption + [tokenizer.eod_id]*(max_length - len(caption))))
        labels.append(torch.tensor(caption_labels))

    labels = torch.stack(labels, dim=0).long()
    captions = torch.stack(captions, dim=0).long()
    images = torch.stack(images, dim=0).to(torch.float16)

    return {"images": images, "input_ids": captions, "labels": labels}

class ImageCaptionDataset(Dataset):
    def __init__(self, tokenizer, image_map_file, captions_file, Vconfig, return_caption_num=1, max_train_data_item=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.return_caption_num = return_caption_num
        self.max_train_data_item = max_train_data_item

        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]  # RGB

        self.tran = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize([224, 224])
        ])

        self.image_map = readJson(image_map_file)
        self.captions = readJson(captions_file)

        self.image_processor = CLIPProcessor.from_pretrained(Vconfig.model_path)

        self.readImage()  # 一次性读入内存

    def readImage(self):
        self.data_list = []
        number = 0
        image_map_keys = list(self.image_map.keys())
        np.random.shuffle(image_map_keys)
        for IM in tqdm(image_map_keys):
            number += 1
            if self.max_train_data_item is not None and number > self.max_train_data_item:
                return
            try:
                image_file_path = self.image_map[IM]["path"] + self.image_map[IM]["image_file"]
                self.data_list.append([image_file_path, self.image_map[IM]["ID"]])
            except Exception as e:
                print(f"Error loading image {IM}: {e}")
                continue

        # Debug information
        print(f"Total images loaded: {len(self.data_list)}")

    def __getitem__(self, index):
        image_path, ID = self.data_list[index]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise

        captions_data = self.captions.get(str(ID), {})
        captions = captions_data.get("a", [])
        
        # Ensure captions is a list
        if isinstance(captions, str):
            captions = [captions]
        elif isinstance(captions, dict):
            # Handle the case where captions is a dictionary
            captions = [captions.get("value", "")]
        
        if not isinstance(captions, list):
            raise ValueError(f"Captions for ID {ID} are not in the expected format: {captions}")
        
        if not captions:
            raise ValueError(f"No captions found for ID {ID}")
        
        prompt = captions_data.get("q", "")
        
        # Debug information
        # print(f"Captions for ID {ID}: {captions}")
        
        select_idx = np.random.choice(len(captions))
        
        # More debug information
        # print(f"Selected index: {select_idx}, Selected caption: {captions[select_idx]}")
        
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

        prompt_raw, context_tokens = make_context(
            self.tokenizer,
            prompt,
            history=[],
            system="你是一位图像理解助手。"
        )

        choice_captions = self.tokenizer(prompt_raw)["input_ids"]
        answer = self.tokenizer(captions[select_idx])["input_ids"]
        choice_captions = choice_captions + answer

        return image, choice_captions, len(answer)

    def __len__(self):
        return len(self.data_list)

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    class Vconfig:
        model_path = "openai/clip-vit-base-patch32"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = ImageCaptionDataset(
        tokenizer=tokenizer,
        image_map_file="path/to/image_map.json",
        captions_file="path/to/captions.json",
        Vconfig=Vconfig,
        return_caption_num=1,
        max_train_data_item=1000
    )

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: data_collate(x, tokenizer, black_token_length=10))

    for batch in dataloader:
        print(batch)
