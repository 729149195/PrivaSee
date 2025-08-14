# PrivaSee

## 启动

- 前端（Vite + React）
  - `npm run dev`
- 后端（FastAPI，conda 环境 `privasee`）
  - `conda activate privasee`
  - `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

---

## 技术原理

目前测试中的两类核心能力：

1) PII 检测与高亮（Presidio + spaCy）
2) 不确定性热力（Integrated Gradients + Monte Carlo Dropout）

### 1. PII 检测与高亮

#### 1.1 组件与总体流程

- 使用 [Microsoft Presidio](https://github.com/microsoft/presidio) 的 `AnalyzerEngine` 完成 PII 实体识别；
- Presidio 的 NLP 引擎选择 `spaCy`，并在后端运行 `en_core_web_sm` 或 `en_core_web_lg`；
- 接口：`POST /analyze` 接收文本，返回每个实体的 `start/end/type/score`；前端按范围生成带实体类型与置信度的彩色胶囊进行高亮。

#### 1.2 识别器原理（规则 + 统计）

Presidio 的检测器通常融合两类能力：

- 统计式 NER：依赖 NLP 模型（此处是 spaCy）对 `PERSON/ORG/LOCATION` 等通用命名实体进行识别；
- 规则/模式检测：对邮箱、URL、IP、银行卡号、手机号等采用正则与校验（如 Luhn 校验）进行精确匹配与打分。

将返回的每个实体的 `score` 当作置信度进行展示。

#### 1.3 语言与预处理

- 目前默认英文模型（更稳定，速度快）。
- 为提高对 “全角数字/连接符” 等的鲁棒性，在后端对文本做了轻量归一化（全角数字→半角、各种破折号→`-`、常见全角标点→半角）。这不会改变字符长度，从而保证返回的实体范围不会错位。

#### 1.4 参考与来源

- Presidio: “Presidio: Data protection and PII anonymization” (GitHub: microsoft/presidio)
- spaCy: “Industrial-strength Natural Language Processing in Python” (Explosion AI)
- Luhn 校验: Hans Peter Luhn, 1954

---

### 2. 不确定性热力（Uncertainty Heatmap）

该功能用于估计“输入的哪些部分更可能影响模型在下一步的输出，并且这些影响存在多大不确定性”。在后端实现了如下组合方法：

#### 2.1 Integrated Gradients（积分梯度）

- 核心思想：把输入从基线（如全零嵌入）沿一条路径逐步移动到真实输入，累积这条路径上关于目标（如下一步 token 的对数几率）的梯度，从而得到更稳定的归因。
- 数学形式（对单个输入维度 \(x_i\)）：
  \[ IG_i(x) = (x_i - x'_i) \int_{\alpha=0}^1 \frac{\partial F(x'+\alpha(x-x'))}{\partial x_i} d\alpha \]
  实现上用有限步数进行离散近似（本项目默认 16 步，可配置）。
- 参考：
  - Sundararajan, Taly, Yan. “Axiomatic Attribution for Deep Networks.” ICML 2017.

#### 2.2 Monte Carlo Dropout（MC Dropout）

- 核心思想：推理时开启 Dropout，进行多次前向（样本数默认 16，可配置），将每次的归因结果进行统计（均值/标准差），以估计不确定性来源于“模型权重的随机性”。
- 参考：
  - Gal, Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML 2016.

#### 2.3 指标与可视化

- 我们针对每个输入 token（精确到字符区间）得到两组统计：
  - 影响力均值：多次前向下 IG 归因的绝对值均值，越大代表对下一步预测影响越大；
  - 不确定性（标准差）：多次前向下 IG 归因的标准差，越大代表模型在该位置的归因不稳定；
- 可视化：热力渲染采用红色透明度映射（默认把均值与标准差的最大值映射到颜色深度）。为避免个别极值导致“满屏深红”，我们采用 95 分位数的鲁棒缩放，最终将值裁剪到 [0,1]。
- 稳定性：为增强可重复性，支持设置 `seed`（默认 42），每次采样会设定随机种子；同时提高采样次数与积分步数减小方差。

#### 2.4 何为“使用 logits/概率的不确定性”

如果可从 LLM API 或本地模型获得下一步 token 的全词表概率（或 logits），可以直接计算分布熵来刻画不确定性：

- 预测熵：\(H(p)=-\sum_i p_i \log p_i\)，归一化后 \(\hat H=H(p)/\log |V|\in[0,1]\)；
- 多次采样（MC）下，可计算：
  - 预测熵 \(H(\bar p)\)
  - 平均熵 \(\overline{H}\)
  - 互信息（BALD）\(MI=H(\bar p)-\overline{H}\)，更能区分“数据噪声”与“模型不确定”。
    当 API 不提供 logits 概率时，我们退而用 IG+MC Dropout 的归因稳定性来近似输入位置的不确定性。

#### 2.5 参考与来源

- Sundararajan, Taly, Yan. “Axiomatic Attribution for Deep Networks.” ICML 2017.
- Gal, Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML 2016.
- Hinton et al. “Distilling the Knowledge in a Neural Network.”（与 logits/温度缩放相关的背景资料）

---

## 接口摘要

- `POST /analyze`：PII 检测。入参：`text`, `model`, `entities`；出参：`[start,end,type,score]`。
- `POST /entities`：返回当前模型支持的实体列表。
- `POST /uncertainty`：不确定性估计。入参：`text`, `model`, `samples`, `ig_steps`, `seed`；出参：`[start,end,token,score_mean,score_std]`。

---

## 前端页面概览与操作说明

页面包含三个组件并排或分组展示：

- TextTest：文本 PII 实体高亮 + 文本不确定性热力
- ImgTest：图片检测（PII/人脸/车牌/OCR）+ 图像不确定性热力 + 地标可定位概率
- AudioTest：音频结构化事件（Onset/Beat/Active segment）+ 不确定性热力 + 原音频播放 + 转写文本的 PII 高亮与不确定性

### 文本（TextTest）

- 左列：PII 命名实体高亮
  - 可选择 NER 模型（`spacy/en_core_web_sm` 或 `en_core_web_lg`）与语言（当前为英文）
  - 可勾选实体类型进行“仅展示/过滤”
  - 高亮胶囊带有实体类型与置信度
- 中列：不确定性热力
  - 基于后端 `/uncertainty`，按字符区间渲染红色透明度（均值与标准差的最大值）
- 右列：输入框（会自适应高度），点击 Analyze 同时触发左中两列的分析

常见用途：合同/邮件等文本的隐私脱敏预审；对模型“自信/不自信”的位置进行针对性复核。

### 图片（ImgTest）

- 面板：上传图片后点击 Analyze
  - 显示 CLIP zero-shot 标签、尺寸、EXIF（含 GPS 解析）、地标可定位概率
- 三列展示：
  - 检测结果（PII/对象/人脸/车牌/OCR）叠加在图片上方，OCR 文本若命中 PII 也会标签显示
  - 不确定性热力：
    - 目标框不确定性（红色，基于置信度的反向/合成估计）
    - OCR 文本不确定性（蓝色，基于 PII 置信度的补充估计）
  - 预览原图

常见用途：照片/截图中的 PII 预检（人脸、证件、车牌、可定位视觉线索），辅助脱敏与可共享性评估。

### 音频（AudioTest）

- 面板：上传音频后点击 Analyze
  - 左列 Waveform & recognized events：
    - Waveform：下采样的波形
    - Onset：声学事件“起点”，瞬时能量/谱差显著上升的位置（常用于切词/鼓点定位）
    - Beat：节拍位置（节奏/速度估计与对齐）
    - Active segment：非静音区段（语音活动检测、剪除静音、限定转写范围）
    - 底部时间刻度 + 黑色播放指示线，支持在画布上拖拽寻址
  - 中列 Uncertainty heatmap：
    - 每个时间片的“不确定性强度”，颜色越深越不稳定
    - 默认用谱熵（librosa）估计；若无 librosa，则回退为能量波动近似
    - 用途：定位“难判定/嘈杂/弱信号”区域，指导人工复核/加大脱敏
  - 右列 Original audio：原音频播放器（与画布联动）
  - 下方两栏 Transcript：
    - Transcript PII highlight：转写文本的 PII 高亮（需可选依赖）
    - Transcript token uncertainty：对转写文本的 token 级不确定性热力

常见用途：会议录音/语音分享的合规检查与脱敏。优先人工听审不确定性较高的区域；在高不确定区段加大遮蔽保护。

---

## 后端接口（图片与音频）

### 图片：`POST /image/analyze`

入参：`file`（multipart/form-data）

返回关键字段：

- `width/height`：图像尺寸
- `exif`：原始 EXIF，附加 `GPSLatitudeDecimal/GPSLongitudeDecimal`
- `clip_top`：CLIP 文本提示的 top-k 标签
- `detections/faces/plates`：YOLOv8 检测框
- `ocr_boxes`：OCR 文本框（附带 `pii_types/pii_scores/uncert`）
- `heat_boxes`：合成的不确定性框（红色叠加）
- `pixel_heat`：CLIP 视觉注意力 roll-out（二维归一化热力）
- `geo_candidates/geo_prob/geo_uncertainty`：地标检索与可定位概率估计

### 音频：`POST /audio/analyze`

入参：`file`（multipart/form-data）

返回关键字段：

- `sample_rate/duration_sec/num_channels`
- `waveform_downsample`：用于前端绘图的定长波形
- `onsets_sec/beats_sec`：起音点与节拍时间（秒）
- `segments`：非静音区段（`start_sec/end_sec/score`）
- `frame_uncertainty`：每帧不确定性（0-1）

### 音频转写（可选）：`POST /audio/transcribe`

依赖：`faster-whisper`（CPU 默认 small/int8），建议同时具备 `ffmpeg`/`libsndfile` 以支持多格式。

入参：`file`（multipart/form-data）

返回：

- `text`：整段转写
- `segments`：分段转写（`start/end/text`）

---

## 安装与依赖（conda 环境）

请在 conda 虚拟环境 `privasee` 中安装依赖，避免装到 base（否则请先卸载再安装到 `privasee`）。

基础依赖（文本 + 图片）：

```bash
conda activate privasee
pip install fastapi uvicorn presidio-analyzer presidio-anonymizer spacy transformers torch pillow ultralytics easyocr
python -m spacy download en_core_web_sm
```

音频/转写相关（推荐）：

```bash
conda install -c conda-forge ffmpeg libsndfile
pip install librosa soundfile faster-whisper
```

如 macOS 上 `brew install libsndfile` 无法访问 ghcr.io，可使用 conda-forge 渠道（如上）。

---

## 常见问题（FAQ）

- 端口与 CORS：
  - 前端默认 `5173`，后端默认 `8000`，CORS 已允许 `http://localhost:5173` 与 `http://127.0.0.1:5173`。
- `/audio/transcribe` 返回 503：
  - 未安装 `faster-whisper` 或音频解码依赖（`ffmpeg`/`libsndfile`）。请按“安装与依赖”章节安装。
- 图片/音频分析过慢：
  - 初次调用会下载模型；可提前联网预热。
- OCR 未返回：
  - `easyocr` 未安装或模型下载失败；或图片语种超出当前 `en/ch_sim` 支持范围。
- 节拍/起音点不明显：
  - 对语音类音频节拍意义较弱；可关注 Active segment 与不确定性热力即可。

---

## 技术原理（详细版）

本节完整梳理文本、图片、音频三类能力所用到的算法与工程细节，并标注主要参考来源与默认超参。

### 文本 Text

#### PII 检测（Presidio + spaCy）
- 管线：`AnalyzerEngine(spacy-nlp)` → 返回实体 `[start, end, type, score]`。
- 识别器：
  - 统计式 NER：`PERSON/ORG/LOCATION` 等通用实体由 spaCy 模型（`en_core_web_sm` 或 `en_core_web_lg`）给出。
  - 规则/模式：`EMAIL_ADDRESS/URL/IP_ADDRESS/CREDIT_CARD/IBAN_CODE/PHONE_NUMBER` 等由正则与校验组合实现（如 Luhn 校验用于银行卡）。
- 预处理：为增强对全角数字/破折号/全角标点的鲁棒性，做字符级归一化（不改变长度，确保区间不偏移）。
- 前端渲染：按实体类型映射颜色与标签，胶囊中显示 `TYPE + score`。

参考：
- Microsoft Presidio（Analyzer/Recognizer 体系、spaCy 接入）
- spaCy（分词/标注/NER）

#### 文本不确定性（IG + MC Dropout）
- 语言模型：默认 `distilgpt2`（可切换 `gpt2`）。
- 归因：Integrated Gradients（积分梯度），从零向量到真实嵌入等分 `ig_steps`（默认 16）积分，取 |Grad×Input| 的和作为每个 token 的影响力。
- 采样：在推理时开启 Dropout，做 `samples` 次前向（默认 16），统计均值与标准差：
  - 均值 ≈ 影响力强弱
  - 标准差 ≈ 不确定性（归因稳定性差 → 不确定性高）
- 归一化：对每条序列使用 95 分位进行鲁棒缩放，裁剪到 [0,1]，映射到红色透明度。

参考：
- Sundararajan et al., 2017（Axiomatic Attribution / Integrated Gradients）
- Gal & Ghahramani, 2016（MC Dropout 近似贝叶斯）

### 图片 Image

#### 目标/人脸/车牌检测（YOLOv8）
- 通用目标：`ultralytics/YOLOv8n`，从 `yolov8n.pt` 自动加载或本地权重。
- 人脸：YOLOv8 社区权重（`yolov8n-face.pt`）。
- 车牌：YOLOv8 社区权重（`yolov8n-license-plate.pt`）。
- 置信度近似不确定性：以 `1 - conf` 作为合成的框级不确定性，用红色半透明遮罩显示。

参考：
- Ultralytics YOLOv8 文档与仓库

#### OCR（EasyOCR）与 OCR 文本的 PII 识别
- OCR：`easyocr.Reader(["en", "ch_sim"])`，返回文本框与置信度。
- 对 OCR 文本再走一轮 Presidio 分析（与文本模块一致），输出 PII 类型与分数；不确定性近似 `1 - max(score)`。

#### CLIP Zero-shot 与 Pixel Attention Rollout
- 文本-图像对齐：`openai/clip-vit-base-patch32`，对一组提示词打分得到 top-k 标签。
- 像素级注意力：提取视觉编码器多层注意力，按层进行 roll-out（加恒等并归一），得到 [H×W] 的相对关注热力。

参考：
- Radford et al., 2021（CLIP）
- Abnar & Zuidema, 2020（Attention Rollout）

#### 地标检索与不确定性（CLIP + FAISS + TTA）
- 索引：遍历 `static/landmarks` 下图片，提取 CLIP 图像特征 → FAISS `IndexFlatIP` 建库。
- 查询：对输入图片提 CLIP 特征并在索引中检索 top-k 候选。
- TTA 不确定性：随机裁剪/水平翻转得到多张 patch，统计 top-1 投票的归一化熵与一致性，结合 margin 与场景先验合成为 `geo_prob`（0~1）。

参考：
- Johnson et al., 2017（FAISS）
- 常见 TTA/投票熵做法（实践经验）

### 音频 Audio

#### 读入与预处理
- 首选 `soundfile`（支持多格式），回退 `librosa.load`，最终回退 Python `wave`（WAV）。
- 通道：内部统一到 `float32`，必要时转单声道（平均）。
- 前端绘图：波形下采样为定长数组（默认 1024 点）。

#### 结构事件
- Onset（起音点）：librosa 的谱差/能量突变检测；无 librosa 时以短窗能量突变近似。
- Beat（节拍）：librosa 节拍追踪（DP + 动态节奏估计）；对语音类音频意义较弱。
- Active segment（非静音）：librosa `effects.split(top_db=30)`；无 librosa 时以短窗能量阈值回退。

参考：
- librosa 文档（onset/beat/effects.split）

#### 音频不确定性（帧级）
- 谱熵：对 STFT 功率谱按帧归一化后计算熵，代表“能量分布均匀度”（越均匀越不确定/嘈杂/弱信号）。
- 回退：若无 librosa，则用短窗能量的归一化波动近似。

参考：
- 经典谱熵指标在语音/音频分析中的应用（教材与综述）

#### 转写（可选）与转写文本不确定性
- 转写：`faster-whisper` 小模型（CPU，int8），支持 `vad_filter`；返回整段与分段时间戳。
- 转写文本 PII：与文本模块一致，送入 `/analyze` 高亮实体。
- 转写文本不确定性：同文本的不确定性方法（IG + MC Dropout）对转写文本渲染热力。

参考：
- Whisper（Radford et al., 2022），faster-whisper 的高效 CTranslate2 实现

### 默认超参与可调项（关键）

- 文本不确定性：`samples=16`，`ig_steps=16`，`max_input_tokens=256`，`seed=42`。
- 图片 CLIP：`openai/clip-vit-base-patch32`；zero-shot prompts 可在后端列表中调整。
- 地标检索：`k=5` 候选；TTA 裁剪数量约 10~12；熵/一致性/场景先验合成时做了归一化与阈值裁剪。
- 音频：谱熵 STFT `n_fft=1024, hop=512`；非静音 `top_db=30`；短窗能量 `win≈20ms, hop≈10ms`。

---

## 架构与数据流

前端（React）通过 HTTP 请求后端（FastAPI）：
- TextTest：`/analyze`（PII），`/uncertainty`（文本不确定性）
- ImgTest：`/image/analyze`
- AudioTest：`/audio/analyze`（波形/事件/帧级不确定性），`/audio/transcribe`（可选转写）→ `/analyze`（转写 PII）→ `/uncertainty`（转写不确定性）

渲染：
- 文本：HTML 片段（高亮胶囊/不确定性 span）
- 图片：Canvas 上绘制框/热力/像素热力
- 音频：Canvas 波形/节拍/起音点/非静音 & 帧级热力，播放指示线与拖拽寻址

---

## 复现与性能建议

- 首次运行会自动下载模型（YOLO/CLIP/Whisper/EasyOCR），建议预热网络或预先缓存。
- CPU 环境：将文本不确定性 `samples/ig_steps` 适当下调；Whisper 可保持 `small-int8`；YOLO 建议 `v8n`。
- GPU 环境：可提高采样与步数以增强稳定性；YOLO/CLIP/Whisper 会明显加速。

---

## 安全、隐私与合规

- 所有分析在本地进行（默认配置下，无需把数据发往远程服务）。
- PII 显示仅用于预审/脱敏，不建议在未授权的环境中存储或传播原始带 PII 的内容。
- 如需导出/分享，请先对高风险区域做遮蔽（文本替换、图像打码/涂抹、音频静音/哔声）。

---

## 局限性与注意事项

- 文本 PII：规则/统计方法各有盲点，特别是在多语种混写或极端格式下会有误检/漏检。
- 图像检测：YOLO 与 OCR 在低光/遮挡/极端角度时精度下降；社区权重的泛化性依赖数据分布。
- 地标检索：依赖本地图库覆盖度；TTA 不确定性仅是启发性指标。
- 音频：节拍/起音点对纯语音意义有限；谱熵不确定性受环境噪声与混响影响。
- 解释性：IG + MC Dropout 给出“稳定性与影响力”的近似，不等同于严格的概率不确定性。

---

## 参考与来源

- Presidio: “Presidio: Data protection and PII anonymization”（Microsoft）
- spaCy: “Industrial-strength Natural Language Processing in Python”（Explosion AI）
- Ultralytics YOLOv8 文档与代码
- EasyOCR: JaidedAI EasyOCR
- CLIP: Radford, Alec, et al., 2021. “Learning Transferable Visual Models From Natural Language Supervision.”
- Attention Rollout: Abnar, Samira; Zuidema, Willem. 2020. “Quantifying Attention Flow in Transformers.”
- FAISS: Johnson, Jeff, et al., 2017. “Billion-scale similarity search with GPUs.”
- Whisper: Radford, Alec, et al., 2022. “Robust Speech Recognition via Large-Scale Weak Supervision.”（使用 faster-whisper 实现）
- Integrated Gradients: Sundararajan, Mukund, et al., 2017. “Axiomatic Attribution for Deep Networks.”
- MC Dropout: Gal, Yarin; Ghahramani, Zoubin. 2016. “Dropout as a Bayesian Approximation.”
- librosa 文档与教程（onset/beat/STFT/谱熵）

