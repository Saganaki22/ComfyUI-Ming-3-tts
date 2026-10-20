# ComfyUI Ming 3.0 TTS

[English](README.md) | 简体中文

[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) 的原生 ComfyUI 移植，支持 0.5B 稠密模型和 Tiny 9B MoE 模型，以及原始合并权重和 INT8 ConvRot 权重。使用现有 ComfyUI 环境，无需降级 Transformers，也不依赖其他 TTS 自定义节点包或远程 Python 模型代码。

Tiny 9B 对应上游的 Tiny 模型：完整语音模型约有 8.98B 参数，语言骨干约有 7.9B 总参数，每个 token 激活约 1.3B 参数。这里的文件夹名称包含 `tiny-9b`，方便与 0.5B 区分。

在 RTX 5090 上已检查 0.5B 的内置音色、音色设计和中英文克隆，以及 Tiny 的 BF16 / INT8 英文克隆。Tiny INT8 的短英语双人对话也已使用独立参考音色检查，确认对话内容完整且说话人身份切换。其他 GPU、较长对话、低显存配置和不同任务、语言组合的效果仍需验证。结果也受参考录音和文本影响。

Ming 使用 ComfyUI 的模型管理、动态显存加载和预取。其模型块不创建 AIMDO 分配编译区域，以避开 Windows / comfy-aimdo 0.5.6 中出现的内存编译错误；不会修改 ComfyUI 全局编译设置。

## 快速开始

1. 将本目录连同 `assets` 文件夹放入 `ComfyUI/custom_nodes/ComfyUI-Ming-3-tts`。
2. 使用启动 ComfyUI 的 Python 安装依赖。例如，在 ComfyUI 目录中使用 Windows venv：

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r custom_nodes\ComfyUI-Ming-3-tts\requirements.txt
   ```

3. 重启 ComfyUI，添加 **Ming TTS Load Model**、**Ming TTS Generate** 和现有的音频预览或保存节点。
4. 先选择 0.5B 模型，`precision=auto`、`device=auto`。模型尚未安装时，可启用 `download_if_missing`，只下载选中的模型。
5. 将 Load Model 连接到 Generate，选择 `task=ip`、`voice=US_James`，填写要朗读的文本后运行。

生成结果为单声道 44.1 kHz 标准 ComfyUI `AUDIO`。可以直接使用现有的 Load Audio、Preview Audio 和 Save Audio 节点。

## 示例工作流

将 [example_workflows](example_workflows) 中的 JSON 拖入 ComfyUI，或通过 **Workflow → Open** 打开。每个示例都有 Markdown 说明节点，介绍任务、参数、模型下载和目录结构。只需要本节点包和 ComfyUI 内置节点。

| 工作流 | 用途 |
| --- | --- |
| [内置音色 ip](example_workflows/01_builtin_voice_ip.json) | 使用 `US_James` 等命名音色，无需参考录音。 |
| [音色设计](example_workflows/02_voice_design.json) | 用文字描述希望生成的声音。 |
| [手动转录克隆](example_workflows/03_voice_clone_manual.json) | 提供录音及其准确转录。 |
| [Whisper 转录克隆](example_workflows/04_voice_clone_whisper.json) | 自动转录参考录音，再生成新语音。 |
| [自由指令](example_workflows/05_freeform_direction.json) | 用自然语言控制表达，可选参考录音。 |
| [基础属性](example_workflows/06_basic_attributes.json) | 请求语速、音高和音量。 |
| [情感](example_workflows/07_emotion.json) | 使用参考音色并选择情感。 |
| [中文方言](example_workflows/08_chinese_dialect.json) | 选择粤语或川渝话。 |
| [Tiny 9B 多人对话](example_workflows/09_tiny_9b_podcast.json) | 两个独立参考音色，各自使用 Whisper 转录。 |
| [副语言标签](example_workflows/10_vocal_tags.json) | 在文本中使用 `[Sigh]`、`[Laughter]` 等标签。 |
| [跨语言克隆](example_workflows/11_cross_language_clone.json) | 用另一种受支持语言的参考音色生成日语。 |

需要参考录音的示例会将 **Load Audio** 留空，请上传自己的文件。手动克隆还需填写录音的准确转录。下载开关默认关闭；请预先安装模型，或分别启用 Ming / Whisper 的 `download_if_missing`。

语音示例会预览音频，并将 FLAC 保存到 `ComfyUI/output/ming_tts/`。随机种子为 42，生成后控制为 fixed，flow steps 保持 10。大多数示例选择 0.5B INT8，多人对话选择 Tiny 9B INT8；单人工作流也可以切换到另一种模型。

## 上游使用手册

`docs` 中保留了上游 cookbook 的两个原始文件：

- [中文控制参数、完整音色列表与示例](docs/instructions.md)
- [English control reference and voice catalog](docs/instruction_en.md)

文件来自 [inclusionAI/Ming-3.0-tts/cookbooks](https://github.com/inclusionAI/Ming-3.0-tts/tree/main/cookbooks)。其中的 Python 示例使用上游 API；在 ComfyUI 中请对应本 README 的节点设置和工作流使用。

## 安装与可选 FLA

基础依赖在 `requirements.txt` 和 `pyproject.toml` 中声明：`onnxruntime` 用于 CampPlus 说话人编码，`tqdm` 用于终端进度。torch、torchaudio、Transformers、tokenizers、safetensors 和 comfy-kitchen 使用 ComfyUI 环境中的版本。

FLA 是 **Flash Linear Attention**。可选的 `fla-core==0.4.2` 为 Tiny 的 Kimi Delta Attention 提供优化 GPU 内核；0.5B 不使用 KDA，也不需要 FLA。已有兼容 PyTorch 和 Triton 环境时，可执行：

```text
python -m pip install --no-deps fla-core==0.4.2
```

Windows 内核检查使用 PyTorch 2.10.0+cu130 和 triton-windows 3.6.0.post26。基础依赖不会自动安装 FLA，`pyproject.toml` 将其声明为可选 `fla` extra。没有 FLA 或使用非 CUDA 设备时，Tiny 使用较慢的 PyTorch FP32 KDA 回退实现。已安装 FLA 的内核执行错误会直接报告。

每次 Tiny 生成都会在终端显示实际路径：

```text
Ming Tiny KDA: FLA (chunk prefill, fused recurrent decode)
```

或包含原因的回退提示，例如：

```text
Ming Tiny KDA: PyTorch FP32 fallback (FLA not available)
```

### GPU 与 Transformers

| GPU | 状态 |
| --- | --- |
| RTX 30 系列 / Ampere | 预期兼容，未进行实机检查。 |
| RTX 40 系列 / Ada | 预期兼容，未进行实机检查。 |
| RTX 50 系列 / Blackwell | RTX 5090 已检查，其他型号未检查。 |

使用 ComfyUI / PyTorch 的注意力和 Comfy Kitchen 运算，没有要求仅限 50 系列的权重格式。INT8 内核取决于 ComfyUI、comfy-kitchen 和 PyTorch 版本；FLA 还需要适配 GPU 的 Triton 内核。显存容量与 GPU 代际是不同问题。

短时独立检查中，0.5B 约使用 2.5–2.8 GiB 已分配 CUDA 显存，Tiny INT8 约 9.9 GiB，Tiny BF16 约 16.7 GiB。以上不包含其他程序及部分驱动开销，不能当作最低显存保证。长参考音频和长输出可能需要更多显存。CPU 路径较慢；MPS、ROCm、DirectML、XPU 等设备未使用完整权重进行检查。

检查过的 Transformers 版本范围为 **4.57.6–5.20.2**，具体组合如下，并非每个中间版本都已检查：

| Transformers | huggingface_hub | tokenizers |
| --- | --- | --- |
| 4.57.6 | 0.36.2 | 0.22.2 |
| 5.3.0 | 1.3.0 | 0.22.2 |
| 5.5.0 | 1.5.0 | 0.22.2 |
| 5.20.2 | 1.31.0 | 0.23.1 |

5.20.2 的隔离环境还使用了 safetensors 0.8.0。各库必须满足所选 Transformers 版本的依赖要求；如果现有 5.3.0 工作正常，无需为了本节点升级或降级。Ming 的推理为原生实现，Transformers 主要提供 tokenizer API 和 Whisper 模型。

## 模型下载与目录

四个模型在 [drbaph/Ming-3.0-tts-ComfyUI](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/tree/main/checkpoints)：

| 模型 | 直接下载 |
| --- | --- |
| 0.5B 原始合并权重 | [model.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-0.5b/model.safetensors?download=true) |
| 0.5B INT8 ConvRot | [ming-3-tts-0.5b-int8-convrot.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-0.5b-int8-convrot/ming-3-tts-0.5b-int8-convrot.safetensors?download=true) |
| Tiny 9B 原始合并权重 | [ming-3-tts-tiny-9b.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-tiny-9b/ming-3-tts-tiny-9b.safetensors?download=true) |
| Tiny 9B INT8 ConvRot | [ming-3-tts-tiny-9b-int8-convrot.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-tiny-9b-int8-convrot/ming-3-tts-tiny-9b-int8-convrot.safetensors?download=true) |

```text
ComfyUI/models/checkpoints/
├── Ming-3.0-tts-0.5b/model.safetensors
├── Ming-3.0-tts-0.5b-int8-convrot/
│   └── ming-3-tts-0.5b-int8-convrot.safetensors
├── Ming-3.0-tts-tiny-9b/
│   └── ming-3-tts-tiny-9b.safetensors
└── Ming-3.0-tts-tiny-9b-int8-convrot/
    └── ming-3-tts-tiny-9b-int8-convrot.safetensors
```

下载哪个模型，就放入对应的文件夹。每个发布模型文件夹只需要一个权重文件。配置与 tokenizer 位于自定义节点的 `assets/0.5b`、`assets/tiny-9b`，共享说话人编码器为 `assets/campplus.onnx`，不要漏装 `assets`。

每个检查点包含 LLM、flow head、音频 VAE 和桥接层。`precision` 控制计算精度，INT8 则是权重存储格式；选择 BF16 不会将整个 INT8 检查点还原成原始权重。下载只获取选中的文件，不会同时获取另外三个模型。旧工作流使用旧 Tiny 名称时，重启后需重新选择包含 `tiny-9b` 的名称。

### 共享目录、软链接与 junction

现有 `models/ming3_tts` 或 `models/checkpoints/ming3_tts` 模型仍可发现。完整本地模型可以保留自己的配置、tokenizer 和 CampPlus。四个发布名称支持仅权重文件夹；自定义名称需要完整的本地配置与 tokenizer。

可在 `extra_model_paths.yaml` 中添加：

```yaml
shared_ming:
  ming_tts: D:/AI/models/ming3_tts
```

也可以使用已有标准映射：

```yaml
shared_models:
  base_path: D:/AI/models
  checkpoints: checkpoints
```

重启后，节点搜索映射的 checkpoints 目录、其 `ming3_tts` 子目录及相应模型树。支持文件夹软链接和 Windows junction，例如在 ComfyUI 目录执行：

```powershell
New-Item -ItemType Junction -Path '.\models\checkpoints\Ming-3.0-tts-tiny-9b-int8-convrot' -Target 'D:\AI\models\Ming-3.0-tts-tiny-9b-int8-convrot'
```

优先使用已找到的本地模型。新的 Ming 下载始终写入当前安装的 `models/checkpoints`，新的 Whisper 下载写入 `models/whisper`；不会因为额外路径被标记为默认目录而写入共享盘。

### 系统内存

INT8 权重已经保存为 ComfyUI 原生量化格式，无需再次转换。加载时不会把整个模型展开为 BF16。激活、生成缓存和不同计算 dtype 所需的转换仍会使用内存。

加载器使用 ComfyUI 原生文件加载和磁盘速度策略。在快速存储和 AIMDO 路径上，文件映射可避免在显存权重旁长期保留一份完整的 pinned CPU 副本。慢速存储、显存压力或显式 high-RAM 设置仍可能需要 RAM 缓存。

RTX 5090 的一次独立 Tiny INT8 对比中，生成后的进程常驻内存从 18.67 GiB 降到 2.52 GiB，私有常驻内存 USS 从 10.37 降到 2.27 GiB，Windows 私有提交内存从 32.70 降到 14.42 GiB。重复生成及卸载/重新加载结果与基线逐位一致。这不是最低内存保证，也不是无需 RAM 的 SSD 直达显存；系统文件缓存、传输缓冲和卸载仍使用内存。

## 节点与任务

| 节点 | 用途 |
| --- | --- |
| Ming TTS Load Model | 加载原始或 INT8 模型，选择计算精度、设备和下载开关。 |
| Ming TTS Generate | 通过任务下拉菜单切换单人语音生成模式。 |
| Ming TTS Podcast Speaker | 定义一个说话人的名称、参考录音和转录。 |
| Ming TTS Podcast (Tiny 9B) | 原生多说话人对话，使用可增长的说话人输入。 |
| Ming TTS Whisper Transcribe | 将参考录音转成文本，用于克隆或多人对话。 |

| Generate 任务 | 参考输入 | 参数与含义 |
| --- | --- | --- |
| `clone` | 录音及其准确转录 | 选择新语音的目标语言。 |
| `voice_design` | 无 | `style` 描述音色、年龄感、口音等。 |
| `ip` | 无 | `voice` 使用准确的内置音色名称。 |
| `freeform` | 可选录音 | `instruction` 为自然语言表达指令。 |
| `basic` | 录音，无需转录 | `speed` 为 slow/medium/fast；`pitch`、`volume` 为 low/medium/high。 |
| `emotion` | 录音，无需转录 | happy、sad、angry、afraid、disgusted、surprised、neutral。 |
| `dialect` | 录音，无需转录 | Sichuanese（川渝话）或 Cantonese（粤语）。 |

每次只选择一个任务，不会任意混合不同任务的控制。`basic`、`emotion`、`dialect` 使用 CampPlus 说话人嵌入；`clone` 则使用参考录音和转录作为延续提示。各设置都有悬停说明。

### 内置音色 ip

选择 `task=ip`，在 **voice** 中填写准确标识符，无需参考录音或转录：

| 名称 | 声音 | 口音 |
| --- | --- | --- |
| `US_Emma` | 女声 | 美式英语 |
| `US_James` | 男声 | 美式英语 |
| `US_Stella` | 女声 | 美式英语 |
| `UK_George` | 男声 | 英式英语 |
| `UK_Charlotte` | 女声 | 英式英语 |

中文示例包括 `灵小安`、`灵小柔`、`灵楷`。部分音色需要完整的情感变体名称，例如 `灵未央_高兴`、`灵未央_中性`；不要只填基础名、翻译名称或自行拼接未列出的后缀。使用这些名称时仍选择 `ip`，不是单独的 `emotion` 任务。

以上为上游公布的名称，本项目的内置音色语音检查使用了 `US_James`。[完整中文音色表](docs/instructions.md#built-in-premium-voices) 列出了 100 多个中文音色及可用情感变体。

### 语音克隆与 Whisper

```text
Load Audio ─────────────────→ Generate: task.reference_audio
     └→ Whisper Transcribe ─→ Generate: task.reference_text
Load Model ─────────────────→ Generate: model
Generate: AUDIO ────────────→ Preview Audio / Save Audio
```

- `text`：这次要生成的新句子。
- `task.reference_audio`：要克隆音色的录音。
- `task.reference_text`：该录音中实际说出的内容，不是新句子。
- `task.language`：输出语言，可选中文、英语、德语、西班牙语、法语、韩语、日语。

参考录音可与输出语言不同。将 reference_text 控件转换成输入即可连接 Whisper 的 transcript，也可以手动输入准确转录，不需要 Whisper。请检查人名、数字和识别错误。参考越长，编码与 prefill 通常越慢；节点没有额外强加参考时长上限。

Whisper 是独立的本地 ASR。它搜索 `models/whisper`、`models/audio_encoders` 和额外路径中的完整模型文件夹。支持 large-v3-turbo、large-v3、medium、small、base、tiny。`language=auto` 自动识别参考录音语言，也可填写语言名称或代码。

Whisper 使用标准 ComfyUI patcher，在所选设备运行并在转录后卸载。超过 30 秒的录音使用长音频识别路径。Ming 和 Whisper 都有独立的 `download_if_missing` 开关，默认关闭。设备列表包含 `auto`、`cpu` 及 ComfyUI 报告的 `cuda:0` 等设备。

### Tiny 多人对话

每个说话人创建一个 **Ming TTS Podcast Speaker**，提供唯一名称、独立参考录音及该录音的准确转录，再连接到 Podcast 的 `speakers` 输入。可将各自 Whisper 的 transcript 连接到对应 Speaker 的 `text`。

脚本每行一轮，并与 Speaker 的名称完全一致：

```text
Alice: Hello, and welcome to our conversation.
Bob: Thanks, Alice. What are we discussing today?
Alice: How to make a clear reference recording.
```

必须使用 Tiny 模型。说话人顺序决定参考录音、转录和身份嵌入的对应关系。增加说话人时，使用可增长的输入插槽。脚本仍受 `max_steps` 限制；较长对话需要更大的上限。

### 副语言标签

将标签直接放入要生成的 `text`，保留方括号与大小写：

`[Breathing]`、`[Cough]`、`[Sigh]`、`[Laughter]`、`[Surprise-ah]`、`[Surprise-oh]`、`[Dissatisfaction-hnn]`、`[Uhm]`、`[Confirmation-en]`、`[Question-ei]`、`[Question-ah]`、`[Question-en]`、`[Question-oh]`、`[Question-huh]`。

这些标签指导模型生成，并非后期插入音效文件。

## 采样设置与时长

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `seed` | 42 | 随机种子；fixed 保持下次运行种子不变。 |
| `cfg` | 2.0 | 模型原生 guidance 强度，两种模型的公式不同。 |
| `flow_steps` | 10 | 每个音频块的 flow 求解步骤。 |
| `temperature` | 0.0 | 额外随机 flow 噪声，0 时关闭。 |
| `sigma` | 0.25 | temperature 大于 0 时的额外噪声缩放。 |
| `max_steps` | 250 | 音频块数量上限，约 20 秒；模型可提前结束。 |
| `trim` | true | 裁剪结尾静音，不改变语速。 |

一次解码迭代约生成 80 ms 音频，125 个块约为 10 秒。`max_steps` 只设置上限，不能强制精确时长，也不能把已经完成的句子拉长或减速。若要明确控制语速，请使用 `basic` 的 speed。

终端显示保留的 tqdm 进度条、耗时和 it/s，同时保留 ComfyUI 的进度更新。这里的 it/s 计量音频块，不是内部 flow 求解次数。最后一行报告音频时长、生成耗时和 RTF；RTF 小于 1 表示快于实时。生成耗时不包括模型加载，但包括参考处理、prefill 和音频解码。

## INT8 ConvRot

直接下载上面的预转换模型即可，无需自行转换。它们采用 ComfyUI 原生 `int8_tensorwise`、ConvRot、FP32 行尺度和量化元数据，并通过原生混合精度运算加载。

0.5B 的量化范围是 LLM 注意力与 MLP 投影；Tiny 的量化范围是专家、共享专家与稠密 MLP 投影。音频 VAE、flow head、音频桥接、嵌入、路由、说话人和停止预测等保留浮点权重。

| 模型 | 原始权重约占 | INT8 约占 |
| --- | --- | --- |
| 0.5B | 2.76 GiB | 2.42 GiB |
| Tiny 9B | 16.73 GiB | 10.22 GiB |

这些是权重存储大小，不是峰值显存。Tiny 的 1.3B 激活参数不表示只需存储 1.3B 参数。0.5B 转换包含 168 个 INT8 矩阵，其他 1,012 个张量未量化；Tiny 的 30 个原始分片先合并并逐分片校验，再转换 8,904 个矩阵，其他 1,381 个张量保持逐位一致。

一次 RTX 5090 Tiny 英文克隆对比使用相同参考、文本、seed 1234、CFG 2 和 10 flow steps：

| Tiny 权重 | 音频时长 | 生成耗时 | 解码 it/s | 已分配 CUDA 峰值 |
| --- | --- | --- | --- | --- |
| BF16 | 6.2 秒 | 29.79 秒 | 2.84 | 16.74 GiB |
| INT8 ConvRot | 6.1 秒 | 54.08 秒 | 1.61 | 9.89 GiB |

INT8 在该次检查中节省了内存，但速度更慢。它不是已证明的加速选项。转录一致只说明样本内容可辨识，不代表音质、自然度或音色相似度完全相同。

## 常见问题

- **找不到模型：**检查文件夹名称和位置，或启用选中模型的 `download_if_missing`。自定义 Whisper 文件夹需要完整配置、tokenizer、预处理配置和 safetensors 权重。
- **更新后仍使用旧代码或旧输入：**重启 ComfyUI 的 Python 进程；刷新浏览器不会重载自定义节点代码。
- **句子未说完：**若达到 max_steps 上限，可增加上限；若模型自然结束，单纯提高上限不会强制继续，请检查参考、转录和任务。
- **语速过快：**max_steps 不控制语速。`basic` 提供 speed，克隆则受参考条件和模型表达影响。
- **Tiny 太慢：**检查终端的 FLA / fallback 提示、计算精度和卸载压力；INT8 主要用于节省权重内存。
- **FLA 错误：**使用匹配 GPU 的 Triton 和已检查的可选 FLA 版本；未安装 FLA 时使用 PyTorch 路径。已安装内核的执行错误不会静默回退。
- **显存不足：**可尝试 0.5B、INT8、较短参考或输出，以及 CPU。其他已加载模型也会占用显存。
- **克隆音色或内容不理想：**检查参考转录和录音清晰度；将有歧义的数字、缩写写成实际希望朗读的文字。Whisper 识别结果不能证明音色相似度或自然度。
