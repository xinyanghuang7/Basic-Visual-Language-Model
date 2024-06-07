# Building Your Own Multimodal Large Model from Scratch

For the Chinese version of the README, please refer to [中文文档](README_zh.md).

## Code Explanation 💻

- **Data Preprocessing**: Relevant code is located in the `dataprocess` folder, and dataset-related code is in the `dataset` folder. Data preprocessing mainly includes path merging, QA data concatenation, feature insertion token processing, etc.
- **LLM Model**: Uses Qwen-7B as the main model, with related code in the `qwen` folder. By overriding the `forward` method of `QWenModel`, multi-modal features are injected.
- **Visual Model**: Uses `CLIP_VIT` and `SIGLIP_VIT`, with related code in the `visual` folder, which also includes other backbone networks.
- **VLM Model**: Relevant code is in the `model.py` file under the `model` folder.

## Dataset 🌏

We use a multilingual dataset, mainly including the COCO2017 dataset and the AI Challenger image Chinese description dataset:
- COCO dataset annotations use LLAVA's `detail_23k` and `complex_reasoning_77k`, which can effectively enhance the richness of model descriptions.
- AI Challenger dataset uses original annotations and fixed prompts.

## Model Architecture 🤖

In VLM, the visual part uses the `CLIP` or `SIGLIP` model, which has initially achieved semantic alignment, and uses a two-layer MLP for feature mapping. By overriding the `forward` method of `QWenModel`, the corresponding `image` tokens are replaced with visual features.

If you wish to replace the model architecture, please modify [this part](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L41).

## How to Start Deployment 🔧

### Download Relevant Data

| AI Challenger | COCO | complex_reasoning_77k.json | detail_23k.json |
| --- | --- | --- | --- |
| [AI Challenger](https://tianchi.aliyun.com/dataset/145781) | [COCO 2017](http://images.cocodataset.org/zips/train2017.zip) | [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json) | [detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json) |

Please store the datasets according to the paths in the [configuration file](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/dataprocess/config.yaml). Of course, the paths can be customized.

Please note that this path needs to be consistent with [data/](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L29) for the model to read.

After downloading the data, use `process_image.py` for preprocessing.

### Install the Runtime Environment

Use `pip install` to install `requirements.txt`:

```shell
pip install -r requirements.txt
```

### Start Training

The model training adopts the method of freezing the image model, and LLM uses LoRA for training to reduce training pressure. The parameters to be trained include the visual feature mapping layer and the LoRA parameters in LLM. Since the mapping layer is initialized with untrained parameters, a larger learning rate is set for the mapping layer compared to the LoRA part to balance the optimization speed of the model parameters.

Run `train.sh` in the root directory, and you can configure relevant parameters for experiments.

```shell
sh train.sh
```

Through the above steps, you can start the training process and train the multi-modal model.

The model weights will be saved in the `--output_dir`, and this path can also be customized.

#### `train.sh` Script Explanation

```sh
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=25642 train.py \
    --lora_rank 128 \
    --lora_dropout 0.10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 3e-5 \
    --seed 42 \
    --ddp_find_unused_parameters False \
    --feature_proj_lr 1e-4 \
    --remove_unused_columns false \
    --logging_steps 100 \
    --output_dir ./weights/train_V1_5 \
    --target_modules "c_attn|w1|w2" \
    --image_map /home/u2023111315/Basic-Vision-Language-Model/data/image_map_b.json \
    --captions_file /home/u2023111315/Basic-Vision-Language-Model/data/captions_b.json
```

#### Explanation

1. **CUDA_VISIBLE_DEVICES=0**: Use GPU with ID 0.
2. **torchrun**: PyTorch's distributed training tool.
3. **--nproc_per_node=1**: Run 1 process per node.
4. **--master_port=25642**: Set the communication port between processes.
5. **train.py**: Main training script.

#### Parameters Passed to `train.py`

1. **--lora_rank 128**: LoRA layer rank is 128.
2. **--lora_dropout 0.10**: LoRA layer dropout rate is 10%.
3. **--per_device_train_batch_size 4**: Training batch size per device is 4.
4. **--gradient_accumulation_steps 1**: Gradient accumulation steps are 1.
5. **--num_train_epochs 2**: Train for 2 epochs.
6. **--save_steps 1000**: Save the model every 1000 steps.
7. **--save_total_limit 5**: Save up to 5 checkpoints.
8. **--learning_rate 3e-5**: Learning rate is 3e-5.
9. **--seed 42**: Random seed is 42.
10. **--ddp_find_unused_parameters False**: Disable DDP finding unused parameters.
11. **--feature_proj_lr 1e-4**: Learning rate for the feature projection layer is 1e-4.
12. **--remove_unused_columns false**: Keep unused columns.
13. **--logging_steps 100**: Log every 100 steps.
14. **--output_dir ./weights/train_V1_5**: Output directory.
15. **--target_modules "c_attn|w1|w2"**: Target modules for LoRA adaptation.
16. **--image_map /home/u2023111315/Basic-Vision-Language-Model/data/image_map_b.json**: Path to the image map file.
17. **--captions_file /home/u2023111315/Basic-Vision-Language-Model/data/captions_b.json**: Path to the captions file.

### Test the Model

Run `test.sh` in the root directory, and you can configure relevant parameters for experiments.

```sh
sh test.sh
```

The code will read images from the folder for Q&A.

#### `test.sh` Script Explanation

```sh
python test.py --base_language_model Qwen/Qwen-7B-Chat --base_value_model openai/clip-vit-large-patch14 --model_weights ./weights/train_V1_5/checkpoint-10000/ --image_path ./test_img/1.jpg --prompt "Describe the colors in the image using language<|extra_0|>"
```

#### Parameters Passed to `test.py`

1. **--base_language_model Qwen/Qwen-7B-Chat**: Specify the path to the base language model, here using `Qwen/Qwen-7B-Chat`.
2. **--base_value_model openai/clip-vit-large-patch14**: Specify the path to the base visual model, here using `openai/clip-vit-large-patch14`.
3. **--model_weights ./weights/train_V1_5/checkpoint-10000/**: Specify the path to the model weights, here using the checkpoint `checkpoint-10000` saved during training.
4. **--image_path ./test_img/1.jpg**: Specify the path to the input image, here using `./test_img/1.jpg`.
5. **--prompt "Describe the colors in the image using language<|extra_0|>"**: Specify the prompt for the model, here asking the model to describe the colors in the image using language.

## References 📚

Thanks to the great work of the following projects 🙌:

- https://github.com/WatchTower-Liu/VLM-learning/tree/main
- https://github.com/QwenLM/Qwen
- https://github.com/haotian-liu/LLaVA

## Contact ✉

If you have any questions or ideas, feel free to contact me 😊:

hsinyanghuang7@gmail.com

I will reply as soon as I see the email!
