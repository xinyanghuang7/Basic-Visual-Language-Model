# Building Your Own Multimodal Large Model from Scratch

For the Chinese version of the README, please refer to [中文文档](README_zh.md).

## Code Explanation 💻

- **Data Preprocessing**: The relevant code is located in the `dataprocess` folder, with dataset-related code in the `dataset` folder. Data preprocessing mainly includes path merging, QA data concatenation, and token processing for feature insertion.
- **LLM Model**: We use Qwen-7B as the main model, with relevant code in the `qwen` folder. By overriding the `forward` method of `QWenModel`, we inject multimodal features.
- **Vision Model**: We use `CLIP_VIT` and `SIGLIP_VIT`, with relevant code in the `visual` folder, which also includes other backbone networks.
- **VLM Model**: The relevant code is in the `model.py` file within the `model` folder.

## Datasets 🌏

We use multilingual datasets, primarily including the COCO2017 dataset and the AI Challenger image Chinese description dataset:
- The COCO dataset annotations use LLAVA's `detail_23k` and `complex_reasoning_77k`, which can effectively enhance the richness of the model's descriptions.
- The AI Challenger dataset uses the original annotations with fixed prompts.

## Model Architecture

In the VLM, the vision part uses the `CLIP` or `SIGLIP` models, which have already achieved preliminary semantic alignment, and employs a two-layer MLP for feature mapping. By overriding the `forward` method of `QWenModel`, the corresponding `image` tokens are replaced with visual features.

If you wish to modify the model architecture, please change [this part](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L41).

## How to Start Training

### Download the Relevant Data

| AI Challenger | COCO | complex_reasoning_77k.json | detail_23k.json |
| --- | --- | --- | --- |
| [AI Challenger](https://tianchi.aliyun.com/dataset/145781) | [COCO 2017](http://images.cocodataset.org/zips/train2017.zip) | [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json) | [detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json) |

Please store the datasets according to the paths specified in the [configuration file](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/dataprocess/config.yaml). Of course, the paths can be customized.

After downloading the data, use `process_image.py` for preprocessing.

### Install the Runtime Environment

Use `pip install` to install the dependencies listed in `requirements.txt`:

```shell
pip install -r requirements.txt
```

### Start Training

The model training adopts a strategy where the image model is frozen, and the LLM is trained using the Lora method to reduce training pressure. The parameters to be trained include the visual feature mapping layer and the Lora parameters in the LLM. Since the mapping layer consists of untrained initialization parameters, a larger learning rate is set for the mapping layer compared to the Lora part to balance the optimization speed of the model parameters.

Run `train.sh` in the root directory, and you can configure the relevant parameters for experimentation.

```shell
sh train.sh
```

By following the above steps, you can start the training process and train the multimodal model.

The model weights will be saved in the [data/](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L29) directory. This path can also be customized.

## Testing the Model

Run `test.sh` in the root directory, and you can configure the relevant parameters for experimentation.

```shell
sh test.sh
```

The code will read images from the folder and perform Q&A.

## References

Thanks to the following projects for their great work 🙌:

- https://github.com/WatchTower-Liu/VLM-learning/tree/main
- https://github.com/QwenLM/Qwen
- https://github.com/haotian-liu/LLaVA

## Contact

If you have any questions or ideas, feel free to contact me 😊:

hsinyanghuang7@gmail.com

I will respond as soon as I see the email!
