# Building Your Own Multimodal Large Model

For the Chinese version of this README, please refer to [中文文档](README_zh.md).

## Code Explanation

- **LLM**: Utilizes Qwen-7B as the main body. The related code is in the `qwen` folder. The `forward` method of `QWenModel` is overridden to inject multimodal features.
- **Visual Backbone**: Uses `CLIP_VIT`. The related code is in the `visual` folder, which also contains other backbone networks.
- **VLM Model**: Located in the `model.py` file within the `model` folder.
- **Data Preprocessing**: Located in the `dataprocess` folder. Dataset-related code is in the `dataset` folder.

## Construction

In the VLM, the visual part adopts the `CLIP` model, which has already achieved preliminary semantic alignment. Specifically, it uses `clip-vit-large-patch14` and employs a two-layer MLP for feature mapping (a single layer is also feasible, as the visual model that has been aligned mainly requires adjustment of the mapping center). By overriding the `forward` method of `QWenModel`, the corresponding `image` tags are replaced with visual features.

## Training

The data used includes multilingual data, primarily the COCO2017 dataset and the AI Challenger image Chinese description dataset. The COCO dataset annotations use LLAVA's `complex_reasoning_77k`, which can effectively enhance the richness of the model's descriptions. The AI Challenger uses the original annotations with a fixed prompt.

### Data Preprocessing

Data preprocessing mainly includes path merging, QA data concatenation, feature insertion token processing, etc.

### Environment Setup

Install PyTorch and use `pip install` to install the dependencies from `requirements.txt`:

```shell
pip install -r requirements.txt
```

### Download Data

Download the relevant data:

| AIC | COCO | complex_reasoning_77k.json |
| --- | --- | --- |
| [AI Challenger](https://tianchi.aliyun.com/dataset/145781) | [COCO 2017](http://images.cocodataset.org/zips/train2017.zip) | [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json) |

### Data Preprocessing

Configure the data paths using `config.yaml` and preprocess using `process_image.py`.

### Start Training

The model training adopts a method where the image model is frozen, and the LLM is trained using the Lora method to reduce training pressure. The parameters to be trained include the visual feature mapping layer and the Lora parameters in the LLM. Since the mapping layer is initialized with untrained parameters, to balance the optimization speed of the model parameters, a larger learning rate is set for the mapping layer compared to the Lora part.

Run `train.sh` in the root directory, and you can configure the relevant parameters for experimentation.

By following the above steps, you can start the training process and train the multimodal model. If you have any questions or need further assistance, please feel free to contact us.

### Reference

Thanks to their work:

https://github.com/WatchTower-Liu/VLM-learning/tree/main
