# Fewshoter - 小样本图像分类工具包

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml/badge.svg)](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml)

Fewshoter 是一个基于 CLIP 的小样本图像分类工具包，专注于细粒度识别。它避免完整模型的微调，而是通过小样本支持集进行学习，结合特征提取、原型构建、可选对齐和混合视觉-文本推理。

## 为什么选择 Fewshoter

- 小样本快速适应
- 零训练和轻量级适应模式
- 可插拔分类头：原型、SVM、MLP
- 开放集处理和校准工具
- 实用的 CLI 和 API 工作流，适用于生产环境

## 系统架构概览

```text
            +----------------------+
            |    支持集图像         |
            +----------+-----------+
                       |
                       v
       +----------------+----------------+
       |      多尺度特征提取               |
       +----------------+----------------+
                        |
          +-------------+-------------+
          |                           |
          v                           v
+----------------------+   +----------------------+
|     视觉原型          |   |   文本探针 / NLP      |
+----------+-----------+   +----------+-----------+
           |                          |
           +------------+-------------+
                        v
             +----------+----------+
             |     混合分类器       |
             +----------+----------+
                        |
                        v
             +----------+----------+
             |    预测 + 置信度     |
             +---------------------+
```

## 项目架构

```text
fewshoter/
├── cli/              # 用户接口：train/inference/evaluate/api_server
├── core/             # 配置、episode 合约和共享工具
├── modalities/       # 编码器适配器（CLIP 图像、Perch 音频占位）
├── learners/         # 小样本学习器（原型网络、基础合约）
├── evaluation/       # 评估器（episodic、legacy、compare 模式）
├── data/             # 数据集、采样器和支持集管理
├── features/         # 特征提取/对齐/属性生成/降维
├── peft/             # 参数高效微调（图像编码器 LoRA）
├── perch_integration/# 音频管道合约和 Perch 2.0 占位
├── engine/           # 分类引擎和优化
├── configs/          # 任务特定的 YAML 配置
└── descriptions/     # 类别级描述字典
```

## 快速开始

### 1) 安装

```bash
pip install -e .
```

### 2) 准备支持集

```text
support_set/
├── class_a/
│   ├── 1.jpg
│   └── 2.jpg
└── class_b/
    ├── 1.jpg
    └── 2.jpg
```

### 3) 构建原型

```bash
fewshoter-train \
  --support_dir support_set \
  --output_dir outputs \
  --model ViT-B/32
```

### 4) 运行推理

```bash
fewshoter-inference \
  --prototypes outputs/prototypes.json \
  --image query.jpg
```

### 5) 评估

传统模式（现有分类器）：
```bash
fewshoter-evaluate \
  --prototypes outputs/prototypes.json \
  --test_dir test_set
```

Episodic 小样本基准测试：
```bash
fewshoter-evaluate \
  --mode episodic \
  --episodes 600 \
  --test_dir test_set \
  --output_json results/episodic_5way5shot.json
```

对比传统 vs Episodic：
```bash
fewshoter-evaluate \
  --mode compare \
  --prototypes outputs/prototypes.json \
  --test_dir test_set \
  --output_json results/comparison.json
```

## 架构亮点

### 模块化设计

代码库遵循模块化、基于合约的架构：

- **Modalities（模态）**: 清晰的编码器接口 (`ModalityEncoder`)，支持 CLIP（图像）和 Perch 2.0 占位（音频）
- **Learners（学习器）**: 抽象基类 (`BaseLearner`) 配合原型网络实现；易于扩展 MAML、匹配网络等
- **Evaluation（评估）**: 多种评估模式（legacy、episodic、compare），支持 N-way K-shot 基准测试和置信区间
- **Data（数据）**: 基于 episode 的数据集采样，确定性分割确保可复现的小样本实验
- **PEFT**: LoRA 支持限定在图像编码器分支，文本编码器保持冻结

### 合约和类型安全

关键的可运行时检查合约确保组件互操作性：

```python
from fewshoter.modalities.base import ModalityEncoder
from fewshoter.learners.base import BaseLearner
from fewshoter.evaluation.base import BaseEvaluator

# 所有合约都是 Protocol 类，用于结构类型
# 实现通过 validate_instance() 在运行时验证
```

### 向后兼容性

模块化架构与旧版管道共存：

- 现有脚本无需修改即可继续工作
- 旧版 CLIP 路径保留在图像模态适配器之后
- 配置文件同时支持旧和新配置项
- CLI 保持所有现有标志，同时添加新的 episodic 选项

## API 和 Python 用法

### 基础分类（传统路径）

```python
from fewshoter import Config, create_feature_extractor, SupportSetManager, create_classifier

config = Config()
feature_extractor = create_feature_extractor(config)
support_manager = SupportSetManager.from_prototypes("outputs/prototypes.json", feature_extractor)
classifier = create_classifier(support_manager, config)

label, details = classifier.classify("query.jpg", return_scores=True)
print(label, details["confidence"])
```

### Episodic 评估（小样本基准测试）

```python
from fewshoter.core.config import Config
from fewshoter.evaluation.episodic import EpisodicEvaluator
from fewshoter.learners.prototypical import PrototypicalLearner

config = Config()
config.evaluation.mode = "episodic"
config.evaluation.episodes = 600

learner = PrototypicalLearner(distance="euclidean")
evaluator = EpisodicEvaluator(
    learner=learner,
    encoder=feature_extractor,
    n_way=5,
    k_shot=5,
    n_query=15
)

results = evaluator.evaluate(dataset, split="test")
print(f"准确率: {results['mean_accuracy']:.3f} ± {results['ci_half_width']:.3f}")
```

### 图像编码器 LoRA 微调

```python
from fewshoter.peft.lora import apply_image_encoder_lora

# 对图像编码器的视觉分支应用 LoRA
result = apply_image_encoder_lora(
    encoder=feature_extractor,
    enabled=True,
    rank=8,
    alpha=16.0,
    target_modules=["attn", "mlp"],
    freeze_non_lora=True
)
print(f"在 {result['scope']} 范围的 {result['applied']} 层应用了 LoRA")
```

## 核心能力

- **多模态编码器**: CLIP 用于图像，Perch 2.0 就绪占位用于音频
- **模块化学习器**: 原型网络配合欧氏距离，可扩展基类
- **Episodic 评估**: N-way K-shot 基准测试配合置信区间
- **LoRA 支持**: 图像编码器的参数高效微调（仅限图像分支）
- **多尺度 CLIP 特征提取**: 中间层钩子获取空间特征
- **注意力引导特征对齐**: 判别性 token 挖掘和子中心
- **属性驱动文本探针生成**: 增强类别描述
- **可选 SVM 和 MLP 分类头**: 原型分类的替代方案
- **降维和缓存加速**: PCA 和 mmap/磁盘缓存
- **开放集未知拒绝**: Z-score、margin 和 softmax 策略

## 安全和隐私

- 默认不需要凭证
- 数据路径/配置保持本地环境
- 不要提交私有数据集、模型产物或生成输出
- 部署公共服务前查阅 `SECURITY.md`

## 开发

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## 路线图

### 已完成 ✓

- 模块化编码器/学习器/评估器架构配合合约
- Episodic 评估配合 N-way K-shot 基准测试
- 原型学习器配合欧氏距离
- 图像编码器微调的 LoRA 支持
- Perch 2.0 集成占位（离线模式）
- 向后兼容 CLI 配合新的 episodic 标志

### 进行中

- ONNX/TensorRT 导出路径
- 标准数据集基准套件（miniImageNet、tieredImageNet、CUB）

### 未来

- 实时 Perch 2.0 运行时集成（可用时）
- 额外学习器：MAML、Matching Networks、Relation Networks
- 多模态融合（图像 + 音频联合嵌入）
- 轻量级 Web 演示
- 预训练小样本检查点的模型仓库

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。

## 贡献

提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
