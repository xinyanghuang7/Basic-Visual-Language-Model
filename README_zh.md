# 从零搭建自己的多模态大模型

For the English version of the README, please refer to [README.md](README.md).

## 代码说明 💻

- **数据预处理**：相关代码位于 `dataprocess` 文件夹下，数据集相关代码在 `dataset` 文件夹中。数据预处理主要包括路径合并、QA 数据拼接、特征插入 token 处理等。
- **LLM模型**：使用 Qwen-7B 作为主体，相关代码在 `qwen` 文件夹中。通过重写 `QWenModel` 的 `forward` 方法，实现多模态特征的注入。
- **视觉模型**：使用 `CLIP_VIT` 和 `SIGLIP_VIT`，相关代码在 `visual` 文件夹中，其中还包含其他主干网络。
- **VLM模型**：相关代码在 `model` 文件夹下的 `model.py` 文件中。

## 数据集 🌏

我们使用了多语言数据集，主要包括 COCO2017 数据集和 AI Challenger 图像中文描述数据集：
- COCO 数据集的标注使用了 LLAVA 的 `detail_23k` 和 `complex_reasoning_77k`，这些标注可以有效提升模型的描述丰富度。
- AI Challenger 数据集使用原始标注，并使用固定的 prompt。

## 模型架构 🤖

在 VLM 中，视觉部分采用已经实现初步语义对齐的 `CLIP` 或 `SIGLIP` 模型，并使用两层 MLP 进行特征映射。通过重写 `QWenModel` 的 `forward` 方法，将对应的 `image` 标记替换为视觉特征。

如果你希望替换模型架构，请修改[这部分](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L41)。

## 如何开始部署 🔧

### 下载相关数据

| AI Challenger | COCO | complex_reasoning_77k.json | detail_23k.json |
| --- | --- | --- | --- |
| [AI Challenger](https://tianchi.aliyun.com/dataset/145781) | [COCO 2017](http://images.cocodataset.org/zips/train2017.zip) | [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json) | [detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json) |

请按照[配置文件](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/dataprocess/config.yaml)中的路径存放数据集。当然，路径可以自定义。

请注意，此路径需要与[data/](https://github.com/xinyanghuang7/Basic-Vision-Language-Model/blob/main/train.py#L29)保持一致，以便模型进行读取。

数据下载完毕后，使用 `process_image.py` 进行预处理。

### 安装运行环境

使用 `pip install` 安装 `requirements.txt`：

```shell
pip install -r requirements.txt
```

### 开始训练

模型训练采用 image model 冻结的方式进行，LLM 使用 Lora 方式训练以减少训练压力。需要训练的参数包括视觉特征映射层以及 LLM 中 Lora 的参数。由于映射层是未训练的初始化参数，为了平衡模型参数优化速度，这里为映射层设定了比 Lora 部分更大的学习率。

运行根目录的 `train.sh`，可自行配置相关参数进行试验。

```shell
sh train.sh
```

通过上述步骤，您可以启动训练过程并进行多模态模型的训练。

模型权重将会保存在`--output_dir`中，同样，这个路径可以进行自定义。

### 测试模型 

运行根目录的 `test.sh`，可自行配置相关参数进行试验。

```shell
sh test.sh
```

代码会读取文件夹下的图片进行问答。

## 参考 📚

感谢以下项目的伟大工作🙌：

- https://github.com/WatchTower-Liu/VLM-learning/tree/main
- https://github.com/QwenLM/Qwen
- https://github.com/haotian-liu/LLaVA

## 联系 ✉

如果你有任何疑问或者想法，十分欢迎随时联系我😊：

hsinyanghuang7@gmail.com

我会在看到邮件的第一时间回复！
