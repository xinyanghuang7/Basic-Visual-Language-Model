import argparse
import torch
from transformers import AutoTokenizer, SiglipProcessor
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
    transforms.Resize([384, 384])
    ])

    return tran(image)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
    
    model = MMultiModal(LanguageConfig(model_path=args.base_language_model), 
                        VisualConfig(model_path=args.base_value_model), 
                        MultiModalConfig(replace_token_id=replace_token_id), 
                        train=False).cuda()
    model.load(args.model_weights)

    prompt = args.prompt

    image_processor = SiglipProcessor.from_pretrained(args.base_value_model)
    image = Image.open(args.image_path).convert("RGB")
    image_pt = image_processor(images=image, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
    # image_pt = image_process(image).unsqueeze(0).cuda().to(torch.bfloat16)
    # print(image_pt1.shape, image_pt.shape)
    messages = [{"role": "system", "content": "你是一位图像理解助手。"}, {"role": "user", "content": "用中文回答："+prompt}]
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
    parser = argparse.ArgumentParser(description="Image and Text Processing with MultiModal Model")
    parser.add_argument("--base_language_model", type=str, required=True, help="Path to the base language model")
    parser.add_argument("--base_value_model", type=str, required=True, help="Path to the base value model")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model")

    args = parser.parse_args()
    main(args)
