# MedReason E1：Complex-CoT SFT 阶段报告

> 状态：已完成  
> 日期：2026-07-26  
> 代码主干：MedicalGPT  
> Base Model：Qwen2.5-3B-Instruct  
> 训练方式：LoRA SFT

## 1. 阶段目标

E1 的目标不是直接训练一个完整的医疗咨询模型，而是为后续
GRPO、DAPO 和 Hard-Group OPD 建立一个合格的初始策略，使模型具备：

- 基础医疗问答和推理能力；
- 稳定的 `<think>...</think><answer>...</answer>` 输出格式；
- 可由程序严格解析的最终答案；
- 在同一道题的多次采样中产生一定比例的正确与错误轨迹，为
  group-based RL 提供相对奖励信号。

本阶段主要回答两个问题：

1. Complex-CoT SFT 是否改善了可验证医疗推理表现？
2. SFT 模型是否已经具备进入 Vanilla GRPO 的训练条件？

## 2. 数据

### 2.1 Complex-CoT SFT 数据

数据来源：

```text
FreedomIntelligence/medical-o1-reasoning-SFT
subset: en
split: train
```

固定划分：

```text
训练集：4,500
验证集：500
总计：5,000
seed：42
```

数据被转换为 MedicalGPT ShareGPT 格式：

```json
{
  "conversations": [
    {"from": "human", "value": "medical question"},
    {
      "from": "gpt",
      "value": "<think>reasoning process</think><answer>final answer</answer>"
    }
  ],
  "system_prompt": "..."
}
```

Token 长度统计基于 Qwen2.5 tokenizer 和真实 chat template：

| 统计量 | Tokens |
|---|---:|
| 最小值 | 376 |
| P50 | 674 |
| P90 | 877 |
| P95 | 951 |
| P99 | 1,105 |
| 最大值 | 1,442 |
| 平均值 | 693.91 |

数据处理检查包括：

- 必要字段非空；
- `<think>`、`<answer>` 标签合法；
- 标准化问题去重；
- Token 长度过滤；
- 固定训练/验证划分；
- 输出文件 SHA256 记录。

本次扫描的 5,000 条候选全部通过格式和长度检查。需要注意，这一流程
没有验证每条 CoT 的医学事实和中间推理是否正确。

### 2.2 独立 MCQA 评测与后续 RL 数据

另行构建 MedQA 和 MedMCQA 数据，不参与 E1 SFT：

```text
RL train：5,000
  - MedQA：2,500
  - MedMCQA：2,500

RL validation：500
  - MedQA：250
  - MedMCQA：250
```

每条数据保留标准 A/B/C/D 标签，用于准确率计算及后续可验证奖励。

泄漏检查结果：

```text
RL validation ∩ SFT train = 0
RL validation ∩ SFT validation = 0
SFT train ∩ SFT validation = 0
```

以上为标准化题干的精确交集检查。

## 3. 训练配置

| 配置项 | 数值 |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| 精度 | BF16 |
| 微调方法 | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |
| 最大上下文 | 4,096 |
| Micro batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 |
| Epochs | 2 |
| Learning rate | 5e-5 |
| Scheduler | Cosine |
| Gradient checkpointing | Enabled |
| GPU | A100 40GB |

训练共完成 564 steps，耗时约 1 小时 32 分钟。

### 3.1 训练曲线

| 指标 | 结果 |
|---|---:|
| 最终平均 Train Loss | 1.4264 |
| Epoch 1 Eval Loss | 1.404993 |
| Epoch 2 Eval Loss | 1.397388 |
| 最终 Validation Perplexity | 4.0446 |

训练过程中：

- Loss 总体下降；
- Eval Loss 两个 epoch 均有记录，且第二个 epoch 继续小幅下降；
- 未出现 NaN；
- 未出现 CUDA OOM；
- Grad Norm 大部分位于约 0.22～0.43；
- 根目录 Adapter 与 `checkpoint-564` 一致。

从优化目标看，训练过程稳定并正常收敛。

## 4. 评测方法

### 4.1 主要评测：GRPO 对齐的生成式 MCQA

Base 与 E1 使用完全相同的：

- 500 道固定验证题；
- Chat template；
- System prompt；
- `<think>/<answer>` 输出约束；
- 最大生成长度 1,024；
- 严格 A/B/C/D 答案解析器；
- Greedy decoding；
- Seed 42。

该评测直接运行模型生成，和后续 GRPO 的 Prompt、答案解析及 Outcome
Reward 保持一致。

主要指标：

- 生成式选择题准确率；
- 格式合法率；
- 截断率；
- 输出长度；
- Base/E1 配对翻转；
- Exact McNemar 检验。

### 4.2 标准 lm-evaluation-harness

仓库已经加入固定版本的 `lm-evaluation-harness` 脚本、MedQA 任务以及
显式数据路径的 MedMCQA 自定义任务。

标准 harness 的 multiple-choice 模式比较候选选项文本的条件概率，不会
先生成 E1 训练要求的 `<think>`。因此它适合作为标准知识能力补充指标，
但不能替代 GRPO 对齐的生成式评测。

本阶段只完成了 harness smoke test，完整 harness 运行已主动停止，未将
有限样本结果作为正式指标。

## 5. 500 题正式生成式结果

### 5.1 总体结果

| 指标 | Base | E1 SFT | 变化 |
|---|---:|---:|---:|
| Accuracy | 41.8% | 46.0% | **+4.2pp** |
| Format Rate | 93.8% | 100.0% | **+6.2pp** |
| Truncation Rate | 0.0% | 0.0% | 0 |
| Mean Generated Tokens | 192.73 | 412.66 | +219.93 |

95% Wilson 区间：

```text
Base Accuracy：37.56% ～ 46.17%
E1 Accuracy：  41.68% ～ 50.38%
```

E1 相对 Base 呈现正向趋势，但两个区间存在重叠。

### 5.2 分数据源结果

| 数据源 | 样本数 | Base | E1 SFT | 变化 |
|---|---:|---:|---:|---:|
| MedQA | 250 | 43.6% | 45.6% | +2.0pp |
| MedMCQA | 250 | 40.0% | 46.4% | **+6.4pp** |
| Overall | 500 | 41.8% | 46.0% | **+4.2pp** |

提升主要来自 MedMCQA；MedQA 仅有小幅提升。

### 5.3 配对翻转

| 情况 | 数量 |
|---|---:|
| Base 和 E1 均正确 | 132 |
| 仅 Base 正确 | 77 |
| 仅 E1 正确 | 98 |
| Base 和 E1 均错误 | 193 |

E1 净增加：

```text
98 - 77 = 21 道正确题
```

Exact McNemar 检验：

```text
Overall：p = 0.1303
MedQA：  p = 0.6609
MedMCQA：p = 0.1174
```

因此目前合理的结论是：

> E1 在固定 500 题上获得 4.2 个百分点的正向提升，但尚未达到常用的
> `p < 0.05` 统计显著标准，不能把它描述为已经得到稳定、确定的医学能力
> 提升。

## 6. 100 题 × 4 Rollout 审计

### 6.1 配置

```text
Prompts：100
Rollouts per prompt：4
总生成数：400
Temperature：0.8
Top-p：0.95
Max new tokens：1,024
Prompt batch size：16
一次并行生成：最多 64 条 completion
```

该 100 题诊断子集包含：

```text
MedQA：60
MedMCQA：40
```

这是固定验证文件中的前 100 条，用于 GRPO 信号审计，不作为最终
Benchmark 单独报告。

运行效率：

```text
耗时：456.52 秒，约 7 分 37 秒
GPU 利用率：观察值约 92%
显存占用：观察峰值约 16GB / 40GB
```

### 6.2 Group 分布

| Group | 数量 | 比例 |
|---|---:|---:|
| All-correct（4/4） | 16 | 16% |
| Mixed（1～3/4） | 56 | **56%** |
| All-wrong（0/4） | 28 | 28% |

每组正确次数：

| 正确次数 | Prompt 数量 |
|---:|---:|
| 0/4 | 28 |
| 1/4 | 19 |
| 2/4 | 16 |
| 3/4 | 21 |
| 4/4 | 16 |

其他指标：

| 指标 | 结果 |
|---|---:|
| Completion Accuracy | 44.5% |
| Pass@4 / Any-correct Rate | 72.0% |
| Majority-correct Rate | 37.0% |
| Effective Group Ratio | **56.0%** |
| Mean Binary Reward Variance | 0.115 |
| Format Rate | 100.0% |
| Truncation Rate | 0.0% |
| Mean Generated Tokens | 446.00 |
| P95 Generated Tokens | 606.10 |
| Max Generated Tokens | 759 |

### 6.3 分数据源 Group 结果

| 数据源 | Completion Acc | Mixed | All-wrong | Pass@4 |
|---|---:|---:|---:|---:|
| MedQA | 47.5% | **65.0%** | 21.7% | 78.3% |
| MedMCQA | 40.0% | 42.5% | **37.5%** | 62.5% |

MedMCQA 的全错比例更高，后续可能成为 Hard-Group OPD 的主要困难样本
来源。

## 7. 结果解释

### 7.1 已经得到验证的能力

E1 明显学会了：

- 严格遵循 `<think>/<answer>` 格式；
- 稳定输出可解析的单个选项标签；
- 生成更完整、更长的医疗推理；
- 在固定 MCQA 评测上获得小幅正向提升；
- 对同一题产生足够多的正确/错误差异轨迹。

56% 的 mixed group 表明，当前模型已经具备较充分的组内相对奖励信号，
可以进入 Vanilla GRPO。

### 7.2 尚未得到验证的能力

当前结果不能证明：

- 模型已经具备综合医疗咨询能力；
- 所有生成的长 CoT 都在医学上正确；
- E1 的准确率提升已经统计显著；
- 输出更长必然代表推理质量更高；
- 模型在开放式病例和真实临床场景中同样提升。

人工抽查中已经观察到“最终选项正确，但中间医学推理存在概念错误”的
样本。这说明 Outcome Accuracy 不能完整反映过程质量，也构成后续过程
评测或奖励研究的实际动机。

### 7.3 当前主要限制

1. Complex-CoT 数据仅做格式、重复和长度检查，没有逐条医学验证。
2. SFT 数据是开放式医疗问答，而 RL/评测任务是四选一题，存在分布差异。
3. 500 题结果呈正向趋势，但统计证据尚不充分。
4. 100×4 rollout 子集为 60/40 数据源构成，不是严格分层的 50/50 抽样。
5. 暂未建立独立的开放式推理质量、证据一致性和幻觉评测集。

## 8. E1 阶段验收结论

### 8.1 验收状态

```text
训练稳定性：通过
Adapter保存与加载：通过
格式遵循：通过
独立评测无泄漏：通过
准确率非退化：通过
Mixed Group信号：通过
正式GRPO启动条件：通过
```

### 8.2 最终判断

> E1 可以作为后续在线 RL 的初始策略。当前无需继续扩大 SFT 数据或重复
> 训练，应进入 E2 Vanilla GRPO，并将 E1、E2 作为严格对照。

这个结论的依据不是单独的 SFT Loss，而是：

- 独立 500 题生成式 Accuracy 没有退化并提升 4.2pp；
- 格式率达到 100%；
- 100×4 rollout 中 mixed group 达到 56%；
- 无格式崩溃、无截断、无显存或数值异常。

## 9. 下一阶段要求

E2 Vanilla GRPO 应：

1. 从 E1 Adapter 初始化；
2. 使用 RL train 数据，不能使用 validation；
3. 使用 `num_generations=4`；
4. 先使用 Outcome Accuracy + Format Reward；
5. 保存每个训练 prompt 的 group 奖励和正确性；
6. 持续记录 all-correct、mixed、all-wrong 比例；
7. 将训练集 all-wrong prompt 写入 Hard Buffer；
8. 不把当前 100 道验证题写入 OPD 训练数据；
9. 完成后在相同 500 题上与 E1 公平比较。

建议实验顺序：

```text
E0：Base
E1：Complex-CoT SFT
E2：E1 + Vanilla GRPO
E3：E1 + DAPO-style GRPO
E4：E3 + Hard-Group OPD
```

## 10. 结果文件

训练输出：

```text
outputs/medreason-e1-qwen25-3b-lora-r16/
outputs/medreason-e1-qwen25-3b-lora-r16/checkpoint-282/
outputs/medreason-e1-qwen25-3b-lora-r16/checkpoint-564/
```

500 题生成式评测：

```text
outputs/medreason-eval/generative-500/base.jsonl
outputs/medreason-eval/generative-500/base.summary.json
outputs/medreason-eval/generative-500/e1.jsonl
outputs/medreason-eval/generative-500/e1.summary.json
outputs/medreason-eval/generative-500/comparison.json
```

100×4 rollout 审计：

```text
outputs/medreason-eval/rollout-100-g4/e1.jsonl
outputs/medreason-eval/rollout-100-g4/e1.summary.json
outputs/medreason-eval/rollout-100-g4/run.log
```

数据清单：

```text
data/medreason/e1_complex_cot/manifest.json
data/medreason/rl_mcqa/manifest.json
```

