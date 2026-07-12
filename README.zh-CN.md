# signlang-det-training

[English](README.md) | [简体中文](README.zh-CN.md)

一个用于训练通用动态手语特征编码器的开源项目，完整流程封装在可独立运行的 Kaggle Notebook 中。

## 项目简介 🚀

本项目用于训练 `signlang_det`：一个面向原型检索的轻量手部时序编码器。编码器先从 Google Isolated Sign Language Recognition 语料中学习通用时序表示，再使用目标 landmark 数据进行领域适配。

训练期间使用的分类头只提供辅助监督，不会包含在最终导出的编码器中，也不会限制推理阶段可使用的手语词表。

完整流程见 [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb)。

## 核心特点 ✨

- 训练、评估和推理统一使用 `hand168-temporal` 预处理
- 采用腕点相对归一化，并显式处理左右手缺失情况
- 左右手共享 TCN 分支，随后通过带掩码的 Transformer 融合
- 联合使用交叉熵、监督对比损失和 batch-hard triplet 损失
- 使用类别完全隔离的验证集和测试集评估开放词表能力
- 先进行池化余弦检索，再使用受限帧级 DTW 重排
- 支持可恢复的 memory-mapped 缓存和原子 checkpoint
- 导出带 SHA-256 指纹的独立编码器

## 训练流程 🧭

Notebook 从上到下依次完成完整流程：

1. 检查 Kaggle CUDA 环境和所选 GPU 架构。
2. 将 Google ASL parquet 序列转换为可恢复的 `hand168-temporal` 缓存。
3. 使用 P×K 采样训练通用手部时序表示。
4. 在训练时未出现的类别上选择并评估编码器。
5. 审计并预处理目标 object-NPY landmark 数据。
6. 仅加载选中的编码器，重新创建辅助分类头并完成领域适配。
7. 评估池化检索和 Top-K DTW 重排。
8. 导出最终编码器、配置、指标、数据拆分、日志和指纹。

### 特征契约

| 项目 | 约束 |
|---|---|
| 输入 | `float32`，`B × 64 × 168` |
| 每帧特征 | 2 只手 × 21 个关键点 × `(x, y, z, velocity)` |
| Padding 值 | `-100.0` |
| 编码器输出 | `float32`，`B × 64 × 128` |
| 有效帧输出 | L2 归一化 |
| Padding 帧输出 | 全零 |

## 数据要求 📚

运行前需要在 Kaggle Notebook 中挂载以下两个数据集。

### 源语料

Google - Isolated Sign Language Recognition 数据集需要包含：

- `train.csv`
- `sign_to_prediction_index_map.json`
- `train.csv` 的 `path` 列所引用的 parquet 文件

训练只使用 `left_hand` 和 `right_hand` landmark，不使用脸部和身体 landmark。

### 目标 landmark

目标数据集需要提供一个包含 object-NPY 文件的 `landmarks` 目录。每个有效文件必须包含：

- 字符串标签；
- 形状为 `T × 100 × 3` 的 `float32` landmark 数组。

默认布局使用索引 `0:21` 作为右手、`21:42` 作为左手。如果所挂载数据集使用其他已知布局，请修改 `target_right_slice` 和 `target_left_slice`。Notebook 使用受限 pickle 白名单加载对象文件，并将无效文件记录到拒绝清单。

## 在 Kaggle 上运行 ▶️

1. 将 [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) 导入或上传到 Kaggle。
2. 挂载源语料和目标 landmark 数据集。
3. 选择**单张 T4 GPU** 加速器。
4. 检查配置单元；如果自动发现得到多个候选路径，请显式指定数据集路径。
5. 将 `SMOKE_TEST = True` 可执行短流程集成验证；正式训练时保持 `False`。
6. 选择 **Run all**，并完整保留 `/kaggle/working/signlang-det` 输出目录。

Notebook 默认设置 `num_workers = 0` 并关闭 DataParallel，以提高 Notebook 重启稳定性并避免轻量编码器的额外同步开销。兼容的 checkpoint 和未完成的源数据缓存会自动恢复。

## 配置说明 ⚙️

主要设置位于 Notebook 开头的 `Config` 和 `TrainSettings` dataclass 中。

| 配置项 | 默认值 | 作用 |
|---|---:|---|
| `source_root` | 自动发现 | 包含 Google ASL 元数据和 parquet 文件的根目录 |
| `target_landmarks` | 自动发现 | 目标 `landmarks` 目录 |
| `work_dir` | `/kaggle/working/signlang-det` | 缓存、checkpoint、指标和导出目录 |
| `max_frames` | `64` | 最大编码序列长度 |
| `min_frames` | `12` | 与运行时动作分段器一致的最短动作长度 |
| `max_input_frames` | `120` | 拒绝样本前允许的最大动作长度 |
| `embedding_dim` | `128` | 帧级和池化特征维度 |
| `top_k` | `20` | 进入 DTW 重排的池化候选数量 |
| `dtw_window` | `12` | 受限 DTW 窗口 |
| `resume` | `True` | 恢复兼容的训练状态 |
| `use_data_parallel` | `False` | 可选的双 GPU 执行路径 |

正式配置使用 30 epoch × 300 batch 进行通用表示训练，并使用 40 epoch × 200 batch 进行领域适配。这里的 epoch 由固定数量的 P×K batch 组成，不代表完整遍历数据集。

## 输出产物 📦

Notebook 会在 `/kaggle/working/signlang-det` 下写入带版本信息的产物，包括：

- 不可变的逐 epoch checkpoint，以及 `latest.pt` 和 `best.pt`；
- 只包含最终编码器契约和权重的 `signlang_det_encoder.pt`；
- 确定性数据拆分清单和环境信息；
- CSV 与 JSONL 指标、状态文件和持久训练日志；
- 两类数据源的拒绝样本清单；
- 最终检索评估结果；
- 预处理契约标识和编码器 SHA-256 指纹。

每个动态原型都必须同时保存编码器指纹和 `hand168-temporal` 预处理标识。由其他编码器或预处理契约生成的原型不能与当前导出模型混用。

## 限制与校准 ⚠️

- 完整训练依赖 Kaggle 挂载数据集和兼容的 CUDA 环境；仓库本地环境缺少这些条件时无法复现正式训练。
- 当前流程描述的 PyTorch CUDA 构建以 T4 为目标。仅检测到 CUDA 可用并不代表 GPU 架构兼容。
- Notebook 不包含动作分段、服务端 API 或原型管理界面。
- Unknown 手语的距离阈值和 margin 阈值不能从训练 loss 推导，必须使用同时包含已知查询、困难负样本和未知动作的独立数据进行校准。
- 各训练数据集的许可证和使用限制由其所有者负责；本仓库不重新分发任何训练数据。

## 项目文件 🗂️

| 文件 | 说明 |
|---|---|
| [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) | 自包含的 Kaggle 训练与导出流程 |
| [README.md](README.md) | 英文文档 |
| [LICENSE](LICENSE) | 项目许可证 |

## 许可证 📄

本项目基于 [Apache License 2.0](LICENSE) 开源。
