import torch
from transformers import AutoTokenizer, ChineseCLIPProcessor
from torchvision import transforms
from PIL import Image

from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig
from qwen.qwen_generation_utils import make_context


def image_process(image):
    mean=[0.485, 0.456, 0.406]  # RGB
    std=[0.229, 0.224, 0.225]  # RGB

    tran = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize([224, 224])
    ])

    return tran(image)

def main():
    base_language_model = "F:/huggingface_model/qwen/Qwen-7B-chat/"
    base_value_model = "F:/huggingface_model/clip-vit-large-patch14"

    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
    
    model = MMultiModal(LanguageConfig(model_path=base_language_model), VisualConfig(model_path=base_value_model), 
                        MultiModalConfig(replace_token_id=replace_token_id),train=False).cuda()
    model.load("./weights/train_V1_5/checkpoint-27000/")

    prompt = "使用语言描述一下图中出现了那些颜色<|extra_0|>"


    image = Image.open("./data/fb221fa2fe34da45f489a81aa8e94f16.jpeg")
    image = image.convert("RGB")
    # image_processer = ChineseCLIPProcessor.from_pretrained(VModelConfig.model_path)
    # image_pt = image_processer(images = image, return_tensors = "pt").pixel_values.cuda().to(torch.bfloat16)
    image_pt = image_process(image).unsqueeze(0).cuda().to(torch.bfloat16)
    messages = [{"role": "system", "content": "你是一位图像理解助手。"}, {"role": "user", "content": "用中文回答："+prompt}]
    # raw_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw_text, context_tokens = make_context(
            tokenizer,
            "用中文回答："+prompt,
            history=[],
            system="你是一位图像理解助手。"
        )
    question_ids = tokenizer.encode(raw_text)

    result = model.generate(image_pt, question_ids)
    result = tokenizer.decode(result[0])
    print(result)


if __name__ == "__main__":
    main()

