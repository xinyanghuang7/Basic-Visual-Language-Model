# 搭建自己的多模态大模型

如需英文版README，请参考 [README.md](README.md)。

这是一个视觉多模态大模型构建以及训练的笔记，用于学习以及理解多模态大模型。

## 代码说明

- **LLM**：使用了 Qwen-7B 作为主体，相关的代码在 `qwen` 文件夹下，通过重写 `QWenModel` 的 `forward` 来实现多模态特征的注入。
- **视觉主干**：使用 `CLIP_VIT`，相关代码在 `visual` 文件夹下，其中还包含其他主干网络。
- **VLM模型**：在 `model` 文件夹下的 `model.py` 文件中。
- **数据预处理**：在 `dataprocess` 文件夹下，数据集相关代码在 `dataset` 文件夹下。

## 构建

VLM 中视觉部分采用已经实现初步语义对齐的 `CLIP` 模型，具体为：`clip-vit-large-patch14`，使用两层 MLP 进行特征映射（实际上一层也是可以的，已经对齐过的视觉模型更多需要的是调整映射中心）。通过重写 `QWenModel` 的 `forward`，将对应的 `image` 标记替换为视觉特征。

## 训练

数据使用了多语言数据，这里主要为 COCO2017 数据集以及 AI Challenger 图像中文描述数据集。COCO 数据集的标注使用了 LLAVA 的 `complex_reasoning_77k`，该标注可以有效提升模型的描述丰富度。AI Challenger 使用原始标注，并使用固定 prompt。

### 数据预处理

数据预处理主要包括路径合并，QA 数据拼接，特征插入 token 处理等。

### 运行环境

安装 PyTorch 并使用 `pip install` 安装 `requirements.txt`：

```shell
pip install -r requirements.txt
```

### 下载数据

下载相关数据：

| AIC | COCO | complex_reasoning_77k.json |
| --- | --- | --- |
| [AI Challenger](https://tianchi.aliyun.com/dataset/145781) | [COCO 2017](http://images.cocodataset.org/zips/train2017.zip) | [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json) |

### 数据预处理

使用 `config.yaml` 配置数据路径，并使用 `process_image.py` 进行预处理。

### 开始训练

模型训练采用 image model 冻结的方式进行，LLM 使用 Lora 方式训练来减少训练压力。需要训练的参数包括视觉特征映射层以及 LLM 中 Lora 的参数。由于映射层是未训练的初始化参数，所以为了平衡模型参数优化速度，这里为映射层设定了比 Lora 部分更大的学习率。

运行根目录的`train.sh`，可自行配置相关参数进行试验。

通过上述步骤，您可以启动训练过程并进行多模态模型的训练。如果有任何问题或需要进一步的帮助，请随时联系。

### 参考

感谢他们的工作：

https://github.com/WatchTower-Liu/VLM-learning/tree/main
