---
theme: seriph
background: https://cover.sli.dev
class: text-center
drawings:
  persist: false
mdc: true

---

# vLLM 开源框架的核心功能与使用方法

---

## vLLM 简介
vLLM 是一款为大型语言模型（LLM）推理和服务而设计的高效易用的开源库 。该项目最初由加州大学伯克利分校的天空计算实验室开发，现已发展成为一个由学术界和工业界共同贡献的社区驱动型项目 。vLLM 的核心目标是实现简单、快速且经济高效的 LLM 服务，使其能够被广泛应用 。该框架在 GitHub 上获得了高度关注，拥有超过 45,000 个星标 ，这表明了社区对其的强烈兴趣和广泛采用。vLLM 能够支持各种硬件平台，包括 NVIDIA、AMD 和 Intel 的 CPU 及 GPU，以及 Gaudi、TPU 和 AWS Inferentia 等专用加速器 。这种广泛的硬件兼容性使得 vLLM 能够适应不同的基础设施环境。
vLLM 的开发旨在解决 LLM 服务中的关键挑战，例如推理速度慢、延迟高以及内存效率低下等问题 。通过采用先进的技术和优化策略，vLLM 显著提升了 LLM 的推理吞吐量和效率，降低了运行成本，使得更多用户能够轻松部署和使用大型语言模型 。

---

## 起源与社区
vLLM 起源于加州大学伯克利分校的天空计算实验室 。该项目并非由单一机构主导，而是迅速发展成为一个由来自学术界和工业界的贡献者共同维护的开源项目 。这种社区驱动的模式确保了 vLLM 能够持续地进行改进、修复 bug，并支持新的模型和功能 。许多组织为 vLLM 的开发和测试提供了计算资源支持 ，进一步促进了项目的成长。

---
zoom: 0.9

---

## 主要优势
vLLM 提供了诸多关键优势，使其成为 LLM 推理和服务的理想选择 。其中包括：
 * 高吞吐量和低延迟：通过先进的优化技术，vLLM 能够处理大量的推理请求，并以极低的延迟返回结果。
 * 高效的内存管理：利用 PagedAttention 等创新技术，vLLM 能够更有效地管理 GPU 内存，从而支持更大的模型和更高的并发请求数。
 * 易于使用和集成：vLLM 可以与 Hugging Face 上流行的预训练模型无缝集成，方便用户部署和使用自己偏好的模型。
 * 灵活的解码算法：支持多种解码算法，包括并行采样和束搜索等，用户可以根据需求调整生成速度和输出质量之间的平衡。
 * 分布式推理能力：支持张量并行和流水线并行，使得在多个 GPU 上部署超大型模型成为可能。
 * 流式输出：能够逐个生成并输出文本标记，提高了交互式应用的响应速度。
 * OpenAI 兼容的 API 服务器：内置的 API 服务器模仿了 OpenAI API 的接口，使得现有基于 OpenAI API 构建的应用可以轻松切换到 vLLM。
 * 广泛的硬件支持：兼容包括 NVIDIA、AMD、Intel 等多种品牌的 CPU 和 GPU，以及特定的 AI 加速器。
 * 前缀缓存和多 LoRA 支持：支持前缀缓存，可以显著加快共享相同前缀的请求的推理速度；同时支持多 LoRA，可以高效地在同一个基础模型上服务多个微调后的模型。
 * 经济高效的 LLM 服务：通过提高吞吐量和降低资源消耗，vLLM 使得 LLM 服务的成本更低。
这些优势共同使得 vLLM 在 LLM 推理和服务领域具有很强的竞争力，能够满足各种应用场景的需求。

---
zoom: 0.47

---

## 核心功能详解

vLLM 的核心功能围绕着提升 LLM 推理的效率、速度和易用性展开。

- 2.1 高吞吐量和低延迟
    vLLM 的设计目标是实现最先进的服务吞吐量 。它通过多种技术手段来达成这一目标，包括：
    * PagedAttention：这是一项关键创新，通过在非连续的内存块（页面）中存储注意力机制的键（key）和值（value），实现了更高效的内存利用 。这种方法避免了传统方法中由于批处理请求中序列长度不同而导致的内存碎片问题，从而显著提高了速度并降低了内存使用。
    * 连续批处理：vLLM 能够动态地将接收到的推理请求进行批处理，即使这些请求到达的时间不同 。这种技术通过并行处理多个请求，最大限度地提高了底层硬件（如 GPU）的利用率，并减少了计算资源的浪费。
    * CUDA/HIP 图优化：vLLM 利用 CUDA 图（适用于 NVIDIA GPU）和 HIP 图（适用于 AMD GPU）来优化神经网络的执行 。这些图允许预编译和高效执行模型的操作，从而减少了开销并提高了推理速度。
    * 优化的 CUDA/HIP 内核：vLLM 集成了高度优化的 CUDA 和 HIP 内核，包括与 FlashAttention 和 FlashInfer 的集成 。这些技术显著加快了注意力计算的速度，而注意力计算是基于 Transformer 的 LLM 的核心组成部分。
    * 推测解码：这种技术允许模型通过首先使用一个更小、更快的模型生成一些“推测性”的标记，然后使用更大、更准确的模型验证它们，从而并行生成多个标记 。这可以显著降低文本生成的整体延迟。
    * 分块预填充：在文本生成的初始“预填充”阶段（模型处理输入提示），vLLM 可以分块处理提示 。这可以提高效率，尤其是在处理长提示时。
- 2.2 支持各种大型语言模型
    vLLM 旨在与各种流行的预训练 LLM 无缝集成 。它支持 Hugging Face Hub 上提供的广泛模型，包括：
    * Transformer 类 LLM：例如 Llama 系列模型 。
    * 混合专家模型（MoE）：例如 Mixtral 和 Deepseek 系列模型 。
    * 嵌入模型：例如 E5-Mistral 。
    * 多模态 LLM：例如 LLaVA 。
    这种广泛的模型支持使得用户可以轻松地使用自己偏好的模型，而无需进行大量的修改或适配工作。
- 2.3 分布式推理的支持
    对于无法在单个 GPU 上容纳的超大型模型，vLLM 提供了强大的分布式推理支持 。它支持以下两种主要的并行化技术：
    * 张量并行：将模型中的单个层分割到多个 GPU 上进行计算。
    * 流水线并行：将模型分成多个阶段，每个阶段在不同的 GPU 上执行。
    通过这些技术，vLLM 能够利用多 GPU 资源来加速大型模型的推理过程，并克服单 GPU 的内存限制 。此外，vLLM 还支持多节点服务 ，进一步扩展了其处理大规模推理任务的能力。
- 2.4 内存优化技术
    vLLM 的核心优势之一是其先进的内存优化技术，特别是 PagedAttention 算法 。传统的 LLM 服务系统在处理不同长度的序列时，往往会遇到内存碎片问题，导致 GPU 内存利用率不高。PagedAttention 通过将注意力机制的键和值存储在离散的内存页中，而不是连续的内存块中，有效地解决了这个问题 。这种方法允许更灵活地分配和管理内存，从而显著提高了内存利用率，并允许在有限的 GPU 内存中服务更大的模型或处理更多的并发请求 。
- 2.5 流式输出
    vLLM 支持流式输出，即在生成文本的过程中，逐个标记地将结果返回给用户 。这种特性对于需要实时反馈的交互式应用（如聊天机器人）至关重要，因为用户无需等待整个响应生成完毕即可开始接收和处理信息，从而提升了用户体验 。
- 2.6 量化支持
    为了进一步降低模型大小、减少内存占用并提高推理速度，vLLM 支持多种量化技术 。这些技术包括：
    * GPTQ
    * AWQ
    * INT4
    * INT8
    * FP8
    量化通过降低模型权重和激活的精度，在通常对模型准确性影响很小的情况下，实现了显著的性能提升和资源节约。
- 2.7 优化的 CUDA/HIP 内核
    vLLM 包含了针对 NVIDIA GPU 的高度优化 CUDA 内核和针对 AMD GPU 的 HIP 内核 。此外，vLLM 还集成了 FlashAttention 和 FlashInfer 等先进技术，这些技术能够极大地加速注意力机制的计算，从而提高整体的推理效率 。FlashAttention 是一种内存高效的注意力算法，能够避免在 GPU 高带宽内存（HBM）中物化大型中间注意力矩阵，从而加快计算速度并减少内存使用 。
- 2.8 推测解码和分块预填充
    vLLM 采用了推测解码和分块预填充等高级优化技术 。推测解码通过使用一个较小的模型来预测生成标记，然后用较大的模型进行验证，从而实现并行生成多个标记，降低延迟。分块预填充则是在处理长输入提示时，将提示分成多个块进行处理，从而提高效率。
- 2.9 OpenAI 兼容的 API 服务器
    vLLM 内置了一个与 OpenAI API 兼容的服务器 。这意味着，已经为使用 OpenAI API 而构建的应用程序可以轻松地将 vLLM 作为替代方案进行集成，通常只需要更改 API 端点 URL 和 API 密钥即可 。vLLM 支持 OpenAI 的 Completions 和 Chat API ，使得用户可以像与 OpenAI 服务交互一样与 vLLM 服务器进行通信。

---
zoom: 0.65

---

## 安装指南

安装 vLLM 通常可以通过 pip 包管理器完成。建议在安装前创建并激活一个虚拟环境，以避免与其他 Python 包产生冲突 。

- 3.1 系统要求
    vLLM 的系统要求取决于您希望使用的硬件。通常来说：
    * 操作系统：推荐使用 Linux 系统 。
    * Python：需要 Python 3.9 或更高版本 。一些较早的文档可能提到 3.8+ 或 3.9-3.12，但建议使用最新稳定版本。
    * GPU：对于 GPU 加速，需要一块支持的 NVIDIA GPU，并且 CUDA 版本需要在 11.8 或更高。推荐的 GPU 计算能力为 7.0 或更高 。对于 AMD GPU，vLLM 也提供 ROCm 支持 。
    * CPU：vLLM 也支持仅使用 CPU 进行推理。

- 3.2 安装方法
    安装 vLLM 的主要方法是使用 pip ：
    * GPU 安装：对于配备 NVIDIA GPU 的系统，可以使用以下命令安装 vLLM 的预编译 wheel 包：
    pip install vllm

    * CPU 安装：如果您的系统没有 NVIDIA GPU 或者您希望仅使用 CPU 进行推理，可以使用以下命令安装 vLLM 的 CPU 版本：
    pip install vllm --no-cuda

    * 其他加速器：vLLM 还支持其他 AI 加速器，如 AMD GPU、Intel CPU、Gaudi 加速器等。针对这些加速器的具体安装说明，请参考 vLLM 的官方文档。
    * 使用 uv：一些文档建议使用 uv 作为更快的 Python 环境管理器来创建和管理环境，然后再使用 pip 安装 vLLM 。
    * 从源代码安装：高级用户或需要进行自定义配置的用户可以选择从 vLLM 的 GitHub 仓库克隆源代码进行安装 。

- 3.3 依赖
    vLLM 的主要依赖包括 ：
    * Python 3.9+
    * 对于 GPU 支持：CUDA 11.8+ (NVIDIA GPU)
    * pip 会自动安装其他必要的 Python 包作为依赖项。
    * 可选的依赖项包括 tensorizer 和 fastsafetensors 等，可以根据需要安装 。
    * 对于 Mixtral 模型，可能需要安装 megablocks 包 。

---

## 基本使用示例

vLLM 的基本使用可以分为离线推理和在线服务两种方式。

---
zoom: 0.85

---

### 4.1 离线推理
离线推理是指在不进行实时交互的情况下，对一批输入提示进行处理并生成文本 。以下是一个基本的 Python 示例：
* 导入必要的类：
```python
from vllm import LLM, SamplingParams
```

* 定义输入提示和采样参数：
```python
prompts = [
    "你好，我的名字是",
    "美国总统是",
    "法国的首都是",
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

* 初始化 LLM 模型：
```python
llm = LLM(model="facebook/opt-125m") # 默认从 Hugging Face 下载模型
```

可以通过设置环境变量 `VLLM_USE_MODELSCOPE=True` 来使用 ModelScope 上的模型 。vLLM 开源框架的核心功能与使用方法

* 处理输出：

```python
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

---
zoom: 0.35

---

### 4.2 在线服务与 OpenAI 兼容的服务器
vLLM 可以作为一个实现了 OpenAI API 协议的服务器来运行，从而方便进行在线服务 。

* 启动 vLLM 服务器：
在终端中运行以下命令，指定要服务的模型：

`vllm serve Qwen/Qwen2.5-1.5B-Instruct`

服务器默认监听 `http://localhost:8000` 。可以使用 `--host` 和 `--port` 参数指定服务器地址。
* 使用 curl 查询服务器：
* 列出模型：
    `curl http://localhost:8000/v1/models`

* 创建文本补全：
    
    ```sh
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "prompt": "旧金山是一个",
            "max_tokens": 7,
            "temperature": 0
        }'
    ```

* 创建聊天补全：

    ```sh
    curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": "2020年谁赢得了世界大赛？"}
        ]
    }'
    ```hat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": "2020年谁赢得了世界大赛？"}
        ]
    }'

* 使用 openai Python 包查询服务器：
首先安装 openai 包：
`pip install openai`

然后可以使用以下 Python 代码与服务器交互：
```python
from openai import OpenAI

openai_api_key = "EMPTY"  # 如果启用了 API 密钥，请替换为您的密钥
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(model="Qwen/Qwen2.5-1.5B-Instruct",
                                    prompt="旧金山是一个")
print("Completion result:", completion)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "讲个笑话。"},
    ]
)
print("Chat response:", chat_response)
```

服务器默认使用 `tokenizer` 中预定义的聊天模板 ，可以通过参数进行覆盖 。可以使用 `--api-key` 参数或 `VLLM_API_KEY` 环境变量启用 API 密钥检查 。

---
zoom: 0.7

---

## 高级用法场景

vLLM 提供了一系列高级功能，以满足更复杂和高性能的 LLM 推理需求。
- 5.1 批处理
    vLLM 从设计上就支持高效的批处理 。在离线推理中，通过向 `llm.generate()` 方法提供一个包含多个提示的列表，可以轻松实现批处理。vLLM 的连续批处理特性能够优化在线服务中多个并发请求的处理。`--max-num-batched-tokens` 引擎参数可以用来控制每个批次处理的最大标记数 。
- 5.2 分布式推理
    对于需要处理非常大的模型或极高并发量的场景，vLLM 提供了强大的分布式推理能力 。通过在启动 vLLM 服务器或初始化 LLM 类时使用 `--tensor-parallel-size` 和 `--pipeline-parallel-size` 参数，可以将模型分布到多个 GPU 上进行推理 。例如，使用 `vllm serve --model <model_name> --tensor-parallel-size 4` 可以在 4 个 GPU 上运行模型。这种方式需要有多块 GPU 的支持。
- 5.3 自定义模型参数
    vLLM 提供了多种方式来定制模型的行为 ：
    * 引擎参数：
    * 对于离线推理，可以通过将参数作为关键字参数传递给 LLM 构造函数来定制模型参数。例如：`llm = LLM(model="...", quantization="awq", dtype="half")`。
    * 对于在线服务，可以通过在 `vllm serve` 命令中使用命令行标志来定制参数。例如：`vllm serve --model "..." --quantization awq --dtype float16`。
    * 常用的定制参数包括 `--model`、`--quantization`、`--dtype`、`--gpu-memory-utilization` 和 `--max-model-len` 等。
    * 完整的引擎参数列表请参考 vLLM 的官方文档。
    * 配置文件：
    * 与生成相关的参数（例如 `temperature`、`top_p` 和 `max_new_tokens`）可以在 `generation_config.json` 文件中进行设置，并使用 `--generation-config` 参数指定该文件。
    * 可以使用 `--override-generation-config` 参数以 JSON 字符串格式覆盖特定的生成参数。
    * 环境变量：
    * 一些配置可以通过环境变量来控制，例如 `VLLM_USE_MODELSCOPE` 和 `VLLM_API_KEY`。
    * 环境变量还可以用于更高级的配置，例如日志记录 。

---

## 常见用例和应用场景

vLLM 的高性能和高效率使其适用于各种应用场景 ：
 * 聊天机器人和对话式 AI：低延迟和高吞吐量对于实时对话代理至关重要 。
 * 问答系统：能够高效准确地处理大量查询。
 * 文本生成：适用于生成各种创意文本格式、代码、脚本等。
 * 代码生成：可以用于建议代码补全和识别潜在错误 。
 * 摘要和翻译：构建高性能的摘要和翻译工具。
 * 内容创作平台：按需生成文本内容。
 * 研究与开发：快速实验和评估不同的 LLM 模型。
 * 企业级 AI 应用：为各种需要快速高效 LLM 推理的内部 AI 应用提供支持。
 * 检索增强生成（RAG）：使用 vLLM 的推理引擎构建完整的 RAG 解决方案，包括嵌入 API 支持 。

---

## 性能评测和与其他推理框架的比较

vLLM 的设计目标是实现最先进的推理吞吐量 。官方博客和 GitHub 仓库中提供了 vLLM 与其他 LLM 服务引擎（如 TensorRT-LLM、SGLang 和 LMDeploy）的性能比较基准 。这些基准的实现位于 nightly-benchmarks 文件夹中，并且可以使用一键运行脚本进行复现 。评测结果表明，vLLM 在推理速度上相比传统 LLM 框架有显著的提升 。

---

## 总结

vLLM 是一款功能强大且易于使用的开源库，专为高效的 LLM 推理和服务而设计。其核心功能包括高吞吐量和低延迟的推理能力、对各种大型语言模型的广泛支持、分布式推理能力、创新的内存优化技术（如 PagedAttention）、流式输出、多种量化技术支持以及与 OpenAI API 的兼容性。通过简单的安装步骤和清晰的使用示例，用户可以快速上手并利用 vLLM 部署和运行自己的 LLM 模型。其广泛的应用场景和卓越的性能表现，使得 vLLM 成为研究人员和开发人员在 LLM 推理和部署方面的理想选择。建议用户查阅官方文档和社区资源，以获取更深入的了解和支持。

---

### 支持的量化技术

| 技术 | 简要描述 | 优势 | 考虑因素 |
|---|---|---|---|
| GPTQ | 一种后训练量化技术，通过近似最优的方式对模型权重进行量化。 | 显著减小模型大小，提高推理速度。 | 可能导致轻微的精度损失。 |
| AWQ | 一种针对 LLM 的准确且高效的权重量化方法。 | 在保持较高精度的前提下，减小模型大小，提高推理速度。 | 需要特定的量化工具和校准数据。 |
| INT4 | 将模型权重和激活量化为 4 位整数。 | 极大地减小模型大小，显著提高推理速度。 | 可能导致较大的精度损失，需要仔细的校准和评估。 |
| INT8 | 将模型权重和激活量化为 8 位整数。 | 在减小模型大小和提高推理速度之间取得较好的平衡。 | 通常精度损失较小。 |
| FP8 | 将模型权重和激活量化为 8 位浮点数。 | 可以在保持较高精度的同时，提高推理速度并减少内存占用。 | 需要较新的硬件支持（如 CUDA 11.8+）。 |

---

### 常用 vllm serve 引擎参数

| 参数 | 描述 | 示例用法 | 适用场景 |
|---|---|---|---|
| --model | 指定要使用的 Hugging Face 模型名称或路径。 | --model "mistralai/Mistral-7B-v0.1" | 在线服务 |
| --quantization | 启用权重量化。 | --quantization "awq" | 在线服务 |
| --dtype | 设置模型权重和激活的数据类型。 | --dtype "float16" | 在线服务 |
| --gpu-memory-utilization | 控制 GPU 内存使用比例。 | --gpu-memory-utilization 0.8 | 在线服务 |
| --max-model-len | 设置模型支持的最大上下文长度。 | --max-model-len 2048 | 在线服务 |
| --tensor-parallel-size | 设置张量并行的副本数。 | --tensor-parallel-size 4 | 在线服务 |
| --host | 指定服务器监听的 IP 地址。 | --host "0.0.0.0" | 在线服务 |
| --port | 指定服务器监听的端口号。 | --port 8000 | 在线服务 |
| --api-key | 启用 API 密钥验证。 | --api-key "your_secret_key" | 在线服务 |
