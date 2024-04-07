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

    labels = torch.stack(labels, dim = 0).long()
    captions = torch.stack(captions, dim = 0).long()
    images = torch.stack(images, dim = 0).to(torch.float16)

    return {"images":images, "input_ids":captions, "labels":labels}

class ImageCaptionDataset(Dataset):
    def __init__(self, tokenizer, image_map_file, captions_file, Vconfig, return_caption_num = 1, max_train_data_item = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.return_caption_num = return_caption_num
        self.max_train_data_item = max_train_data_item

        mean=[0.485, 0.456, 0.406]  # RGB
        std=[0.229, 0.224, 0.225]  # RGB

        self.tran = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize([224, 224])
        ])

        self.image_map = readJson(image_map_file)
        self.captions = readJson(captions_file)

        self.image_processor = CLIPProcessor.from_pretrained(Vconfig.model_path)

        self.readImage()   # 一次性读入内存

    def readImage(self):
        self.data_list = []
        number = 0
        image_map_keys = list(self.image_map.keys())
        np.random.shuffle(image_map_keys)
        for IM in tqdm(image_map_keys):
            number+=1
            if self.max_train_data_item is not None and number > self.max_train_data_item:
                return 
            try:
                image_file_path = self.image_map[IM]["path"] + self.image_map[IM]["iamge_file"]
                # image = Image.open(image_file_path)
                # image = image.convert("RGB")
                # self.data_list.append([self.tran(image), self.image_map[IM]["ID"]])  # 一次性加载内存消耗较大
                self.data_list.append([image_file_path, self.image_map[IM]["ID"]])
            except:
                continue


    def __getitem__(self, index):  
        # image, ID = self.data_list[index]
        image_path, ID = self.data_list[index]
        # print(image_path)
        image = Image.open(image_path)           # 动态加载每次迭代都有较大的IO开销
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        captions = self.captions[str(ID)]["a"]
        prompt = self.captions[str(ID)]["q"]
        select_idx = np.random.choice(len(captions))
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
