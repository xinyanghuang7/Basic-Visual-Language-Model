import json
import gradio as gr
import torch
from transformers import AutoTokenizer, ChineseCLIPProcessor
from torchvision import transforms
from PIL import Image

from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig

base_language_model = "F:/huggingface_model/qwen/Qwen-7B-chat/"
base_value_model = "F:/huggingface_model/clip-vit-large-patch14"

tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")

model = MMultiModal(LanguageConfig(model_path=base_language_model), VisualConfig(model_path=base_value_model), 
                    MultiModalConfig(replace_token_id=replace_token_id),train=False).cuda()
model.load("./weights/train_V1_5/checkpoint-18000/")

def image_process(image):
    mean=[0.485, 0.456, 0.406]  # RGB
    std=[0.229, 0.224, 0.225]  # RGB

    tran = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize([224, 224])
    ])

    return tran(image)

def chat(image, messages):
    if image is None:
        image_pt = None
    else:
        image_pt = image_process(image).unsqueeze(0).cuda().to(torch.bfloat16)
    raw_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_ids = tokenizer.encode(raw_text)
    result = model.generate(image_pt, question_ids)[0]
    result = tokenizer.decode(result)
    return result

def chatbot_(input_text, chat_history, image):
    SP_token = "<|extra_0|>"
    send_history = [{"role": "system", "content": "你是一位图像理解助手。"}]
    for CH in chat_history:
        send_history.append({"role":"user", "content":CH[0]})
        send_history.append({"role":"assistant", "content":CH[1]})
    if image is not None:
        send_history.append({"role":"user", "content":"用中文回答："+input_text+SP_token})
    else:
        send_history.append({"role":"user", "content":"用中文回答："+input_text})
    bot_message = chat(image, send_history)
    chat_history.append((input_text, bot_message))
    return "", chat_history

def clear_history():
    return "", []


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="image")
            clear = gr.ClearButton()
        with gr.Column():
            chatbot = gr.Chatbot()
            input_text = gr.Textbox()

    input_text.submit(chatbot_, [input_text, chatbot, image], [input_text, chatbot])
    clear.click(clear_history, [], [input_text, chatbot])

def main():
    demo.launch(server_port=23200)

if __name__ == "__main__":
    main()
