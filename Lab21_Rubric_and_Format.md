# Lab 21 — LoRA Fine-tuning · Rubric & Submission Format

> **Module**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn  
> **Time**: 2 giờ thực hành · 30 phút viết report  
> **Cấu phần**: CP3 · **PLO**: K1, K4

---

## 🎯 Mục tiêu học tập (Learning Objectives)

Sau khi hoàn thành lab này, học viên có thể:

1. **Chuẩn bị dataset** Alpaca format đúng chuẩn (instruction / input / output) với token length analysis (p95)
2. **Configure & train** một LoRA adapter dùng QLoRA 4-bit + Unsloth + TRL SFTTrainer
3. **Phân tích trade-off** giữa các rank khác nhau (r=8, r=16, r=64) qua 4 chiều: training time, VRAM, perplexity, qualitative
4. **Đánh giá** fine-tuned model bằng cả quantitative (perplexity) lẫn qualitative (test prompts) methods
5. **Document & defend** kết quả qua một evaluation report rõ ràng

---

## 📋 Tổng quan Lab

| Bước | Hoạt động | Thời gian | Output |
|------|-----------|-----------|--------|
| 1 | Setup environment + install dependencies | 10 phút | GPU verified, libs installed |
| 2 | Dataset preparation (format, tokenize, split) | 15 phút | `train_ds`, `eval_ds` |
| 3 | Configure LoRA + load model 4-bit | 10 phút | model wrapped với r=16 baseline |
| 4 | Train baseline r=16 với SFTTrainer | 30–40 phút | adapter checkpoint + loss curve |
| 5 | **Rank experiment** — train r=8 và r=64 | 30 phút | 2 adapter checkpoints |
| 6 | Evaluation (perplexity + qualitative) | 15 phút | summary table + before/after examples |
| 7 | Viết evaluation report | 30 phút | `REPORT.md` deliverable |

**Total**: ~2.5 giờ.

---

## ⚙️ Prerequisites

- **GPU**: tối thiểu T4 (16 GB Free Colab) hoặc tốt hơn (L4, A100)
- **Python**: 3.10+ (Colab đã có sẵn)
- **Kiến thức**: Day 16-17 (PyTorch basics), Day 19 (Transformers + HF)
- **Tools**: Google Colab (recommended) hoặc Jupyter local với GPU CUDA

**Notebook để dùng**:
- `Lab21_LoRA_Finetuning_T4.ipynb` — cho Colab Free (T4)
- `Lab21_LoRA_Finetuning_BigGPU.ipynb` — cho A100/L4

---

## 📝 Các bước thực hiện chi tiết

### Step 1 — Dataset Preparation (15 phút)

**Yêu cầu:**
- Chuẩn bị **100–500 examples** Alpaca format từ một domain bạn chọn
- Format: `{"instruction": "...", "input": "...", "output": "..."}`
- Clean: dedup, remove samples có output quá ngắn (<10 tokens), filter templates
- Token length analysis → set `max_seq_length = p95` (round up to power of 2)
- Split: **90% train, 10% eval** (random seed = 42)

**Có thể dùng** (🎨 **Free style — pick whatever excites you**):

#### Sample datasets theo domain (chỉ là gợi ý):

| Domain | Sample Dataset HuggingFace |
|--------|---------------------------|
| Vietnamese general | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` |
| English general | `tatsu-lab/alpaca`, `HuggingFaceH4/ultrachat_200k` |
| Coding (Python) | `iamtarun/python_code_instructions_18k_alpaca` |
| Coding (SQL) | `b-mc2/sql-create-context` |
| Medical | `medalpaca/medical_meadow_medqa` |
| Legal | `joelniklaus/legal_case_document_summarization` |
| Finance | `gbharti/finance-alpaca` |
| Recipe / cooking | `recipe_nlg` |
| Story / fiction | `roneneldan/TinyStories` |
| Vietnamese student feedback | `uitnlp/vietnamese_students_feedback` |

#### Hoặc tự tạo dataset cho domain bạn quan tâm:

- **Tiếng Việt vertical**: tax advisor, medical advice, recipes Việt Nam, lịch sử VN, e-commerce reviews
- **Style/format**: JSON structured output, Markdown formatting, specific tone (Gen-Z / formal), company-specific support agent
- **Niche knowledge**: VinUniversity FAQ bot, course tutor cho 1 môn, technical doc Q&A

> 💡 **Khuyến khích**: dataset tự tạo cho domain bạn passionate → kết quả ấn tượng hơn nhiều. 200 high-quality custom samples > 2000 noisy generic samples.

### Step 2 — Configure LoRA (Baseline r=16)

**Yêu cầu cấu hình**:
```python
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]  # lab spec
lora_dropout = 0
gradient_checkpointing = True
```

**Model**: 🎨 **Free style — chọn 1 trong 16+ options** (xem `Day21_README.md`)

#### Recommended defaults theo GPU:

| GPU bạn có | Recommend | Backup options |
|------------|-----------|----------------|
| T4 (16 GB Free Colab) | `Llama-3.2-3B-Instruct` | `gemma-2-2b-it`, `Phi-3.5-mini`, `Qwen2.5-3B` |
| L4 (24 GB Pro) | `Llama-3.1-8B-Instruct` | `Mistral-7B-v0.3`, `gemma-2-9b-it`, `Qwen2.5-7B` |
| A100 (40 GB) | `Llama-3.1-8B` hoặc `gemma-2-9b` | `Mistral-Nemo-12B` |
| A100 80 GB / H100 | `Llama-3.3-70B-Instruct` | bất kỳ smaller model |

#### Theo use case:
- **Generic instruction-following** → Llama 3.2/3.1, Gemma 2, Mistral
- **Tiếng Việt / multilingual** → Qwen2.5, Gemma 2, Llama 3.2
- **Coding tasks** → DeepSeek-Coder, CodeLlama
- **Reasoning-heavy** → Phi-3.5-mini, Llama 3.1
- **Tiny demo / siêu nhanh** → TinyLlama 1.1B, SmolLM2 1.7B

> ⚠️ Ghi rõ model bạn chọn vào REPORT.md. So sánh r=8 vs r=16 vs r=64 trên **cùng model** mới có ý nghĩa.

### Step 3 — Train Baseline với TRL SFTTrainer

**Hyperparameters**:
- 3 epochs
- Cosine LR schedule, learning_rate = 2e-4
- Warmup ratio = 0.10
- Effective batch size = 8 (qua gradient accumulation)
- Optimizer: `adamw_8bit` (paged AdamW)

**Yêu cầu**:
- Monitor loss curve (training loss và eval loss nếu có)
- Detect overfitting: nếu eval loss tăng trong khi train loss giảm → ghi chú lại

### Step 4 — Rank Experiment (PHẦN QUAN TRỌNG NHẤT)

Train **2 adapters thêm** với rank khác:
- **r = 8**, alpha = 16
- **r = 64**, alpha = 128

Trên **cùng dataset**, **cùng hyperparameters** khác (chỉ thay rank/alpha).

**Đo lường**:
| Metric | Cách đo |
|--------|---------|
| Training time | `time.time()` trước/sau `trainer.train()` |
| Peak VRAM | `torch.cuda.max_memory_allocated()` |
| Eval perplexity | `exp(eval_loss)` từ `safe_evaluate()` |
| Trainable params | sum `p.numel()` cho `p.requires_grad` |
| Qualitative output | generate 5 prompts, đánh giá chủ quan |

### Step 5 — Evaluation

- **Quantitative**: compute perplexity trên eval set cho cả 3 ranks + base model (4 numbers)
- **Qualitative**: generate **ít nhất 5** test prompts (recommend: 20). So sánh side-by-side base vs fine-tuned (r=16)

---

## 📦 Deliverables — 3 lựa chọn nộp bài

Hãy chọn **1 trong 3 options** dưới đây tuỳ vào tốc độ internet + LMS limits của bạn:

### 🥉 Option A — Lightweight ZIP (recommended default · ~5-15 MB)

```
lab21_<MSSV>/
├── REPORT.md                          ← Evaluation report
├── notebook.ipynb                     ← Stripped outputs (Cell > All Output > Clear)
├── adapters/
│   └── r16/                           ← CHỈ submit best rank (skip r=8 và r=64)
│       ├── adapter_model.safetensors
│       └── adapter_config.json        ← (skip tokenizer files — base model có rồi)
└── results/
    ├── rank_experiment_summary.csv    ← Numbers cho CẢ 3 ranks (verify được)
    ├── qualitative_comparison.csv
    └── loss_curve.png
```

> Trade-off: chỉ check 1 adapter weights, nhưng metrics đủ trong CSV để verify experiment.

### 🥈 Option B — GitHub + HuggingFace Hub (most professional · ~1 MB ZIP) ⭐ Bonus +5 pts

```
lab21_<MSSV>/
├── REPORT.md                          ← Có links đến HF Hub adapters
├── notebook.ipynb                     ← Stripped outputs
├── results/                           ← CSVs only
└── LINKS.md                           ← GitHub repo + HuggingFace URLs
```

Push adapter lên HuggingFace Hub (free):
```python
ft_model.push_to_hub("your-username/lab21-<model>-r16")
```

Bonus point vì đây là production-grade workflow + adapter publicly verifiable.

### 🥇 Option C — Code-only (purist · ~500 KB)

```
lab21_<MSSV>/
├── REPORT.md                          ← Numbers + 5 qualitative examples đầy đủ
├── notebook.ipynb                     ← Stripped outputs
└── requirements.txt                   ← Pin versions để reproduce
```

> Instructor đọc REPORT + check notebook chạy được. Phù hợp nếu kết nối internet chậm.

---

## 📝 Yêu cầu cho `REPORT.md` (cả 3 options đều cần)

Bắt buộc phải có 6 sections:

1. **Setup** — base model (ghi rõ key từ MODEL_PICKER), dataset (tên + size), GPU, training cost ước tính
2. **Rank Experiment Results** — bảng so sánh 4 chiều (time, VRAM, perplexity, params)
3. **Loss Curve Analysis** — có overfitting không? Tại sao?
4. **Qualitative Comparison** — 5 examples before/after, ngắn gọn nhận xét
5. **Conclusion về Rank Trade-off** — rank nào tốt nhất? Tại sao? (≥ 100 từ)
6. **What I learned** — 2–3 bullet points reflection cá nhân

---

## 🏆 Scoring Rubric (100 điểm + bonus)

| Tiêu chí | Điểm | Mô tả |
|----------|------|-------|
| **1. Functionality** — code chạy end-to-end | **40** | Cả 3 adapters trained + saved (hoặc verifiable qua CSV nếu Option B/C) |
| **2. Experiment Design & Analysis** | **25** | Rank comparison đầy đủ 4 chiều, có insight về trade-off |
| **3. Evaluation Quality** | **15** | Perplexity computed đúng, ≥5 qualitative examples meaningful |
| **4. Report Quality** | **20** | Rõ ràng, đầy đủ 6 sections, conclusion thể hiện hiểu biết |
| **🎁 Bonus — Option B (HF Hub)** | **+5** | Push adapter lên HuggingFace Hub publicly |
| **🎁 Bonus — Stretch goals** | **+10** | Target ALL layers / DoRA / GGUF merge / W&B (xem dưới) |

**Tổng**: 100 điểm + tối đa 15 bonus = 115 max.

### Chi tiết breakdown

#### 1. Functionality (40 pts)
- (10) Setup environment, dataset prep đúng format Alpaca
- (10) Baseline r=16 train successfully + adapter saved/uploaded
- (10) r=8 và r=64 trained successfully + metrics ghi lại
- (10) Eval perplexity computed cho cả 3 ranks (≥2/3 nếu eval gặp OOM nhưng có recovery)

#### 2. Experiment Design & Analysis (25 pts)
- (5) Bảng so sánh đủ 4 metric (time, VRAM, params, perplexity)
- (10) **Insight về rank selection** — không chỉ liệt kê numbers, phải có phân tích
- (5) Diminishing returns observation — ở đâu tăng rank không cải thiện?
- (5) ROI recommendation — cho dataset này, rank nào best?

#### 3. Evaluation Quality (15 pts)
- (5) Perplexity computed đúng (formula: `exp(eval_loss)`)
- (5) Qualitative examples chọn lọc, không cherry-pick — có cả case win lẫn case loss
- (5) Comparison side-by-side rõ ràng (table format)

#### 4. Report Quality (20 pts)
- (5) Đủ 6 sections theo spec
- (5) Conclusion ≥ 100 từ, thể hiện hiểu biết về LoRA mechanics
- (5) Format markdown sạch, có headings + bảng + code blocks
- (5) "What I learned" — reflection cá nhân, không generic

---

## 🎓 Thang điểm đánh giá tổng

| Mức | Điểm | Mô tả |
|-----|------|-------|
| **Xuất sắc** | 90–100 (+ bonus) | Hoàn thành tất cả + có insight ngoài requirement |
| **Tốt** | 75–89 | Đáp ứng đầy đủ requirements, có analysis tốt |
| **Đạt** | 60–74 | Chạy được code, có report, nhưng analysis còn thiếu sâu |
| **Chưa đạt** | < 60 | Không hoàn thành ≥1 trong 3 trainings, hoặc report thiếu/sơ sài |

---

## 📤 Submission

- **Hạn nộp**: trước 23:59 ngày sau (sau khi lab kết thúc)
- **Format nộp**: zip file trên LMS / Google Drive link / GitHub link (Option B)
- **File name**: `lab21_<MSSV>_<HoTen>.zip` (vd: `lab21_22BI13123_NguyenVanA.zip`)
- **Late policy**: -10% mỗi ngày trễ, sau 3 ngày = 0 điểm
- **Honor code**: được dùng AI assistant (Claude, ChatGPT) để debug và viết report — nhưng phải **tự chạy training trên máy của bạn**, không copy adapter của bạn khác

---

## 💡 Tips & Common Pitfalls

### ✅ Do's
- **Bật gradient checkpointing** — giảm 60% VRAM
- **Set `max_seq_length` = p95** của dataset, không quá lớn
- **Save adapter NGAY sau train** — trước khi eval (eval có thể OOM nhưng adapter đã safe)
- **Strip notebook outputs** trước khi submit (Kernel > Restart & Clear Output) — giảm size đáng kể
- **Dùng `packing=False`** trên T4/L4 — packing buggy với new transformers
- **Option B (HF Hub)** — đẹp trên CV, được bonus

### ❌ Don'ts
- **Đừng** include cả 3 adapter folders trong ZIP nếu submission limit chặt
- **Đừng** quên `del trainer; gc.collect(); torch.cuda.empty_cache()` giữa các rank training
- **Đừng** dùng `eval_strategy="steps"` trên T4 — sẽ OOM giữa training
- **Đừng** copy report của bạn khác — sẽ phát hiện qua perplexity numbers

### 🐛 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `KeyError: 'instruction'` khi map dataset | Auto-detect cột (xem cell 9 notebook) |
| `tokenizer is unexpected kwarg` | TRL ≥0.12 + monkey-patch (đã có trong notebook) |
| `evaluation_strategy is unexpected` | Đổi sang `eval_strategy` (auto-handled) |
| `Dim 3 mismatch (497 vs 552)` | Set `packing=False` |
| `on_train_begin must be called before on_evaluate` | Remove `NotebookProgressCallback` (đã có trong `safe_evaluate`) |
| OOM during eval | `safe_evaluate()` fallback đến manual batch=1 loop |

---

## 🚀 Stretch Goals (Bonus +10 pts)

Cho học viên muốn đi sâu thêm:

1. **Target ALL layers** — train thêm 1 adapter với `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`. So sánh với baseline q+v.
2. **DoRA variant** — dùng `use_dora=True` trong PEFT config. Có cải thiện perplexity không?
3. **Merge + GGUF** — merge adapter với base, convert sang GGUF format, test với llama.cpp
4. **W&B integration** — track loss curves real-time, share W&B run link trong report
5. **Custom domain dataset** — dùng dataset tự tạo từ domain bạn quan tâm. Cần ≥200 examples chất lượng cao.

---

## 📚 References

- **LoRA paper** (Hu et al. 2021) — https://arxiv.org/abs/2106.09685
- **QLoRA paper** (Dettmers et al. 2023) — https://arxiv.org/abs/2305.14314
- **Unsloth docs** — https://github.com/unslothai/unsloth
- **TRL docs** — https://huggingface.co/docs/trl
- **FlashAttention paper** (Dao et al. 2022) — https://arxiv.org/abs/2205.14135

---

## 📊 Sample REPORT.md Template

```markdown
# Lab 21 — Evaluation Report

**Học viên**: <Họ tên> — <MSSV>
**Ngày nộp**: <YYYY-MM-DD>
**Submission option**: A (lightweight) / B (HF Hub) / C (code-only)

## 1. Setup
- **Base model**: <điền model bạn chọn — vd: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`>
- **Dataset**: <tên dataset>, <số samples> (X train + Y eval)
- **max_seq_length**: <số> (p95 = <số>, rounded up)
- **GPU**: <Tesla T4 / L4 / A100>, <X> GB VRAM
- **Training cost**: $<số> (~<phút> @ $<rate>/hr)
- **HF Hub link** (nếu Option B): https://huggingface.co/<username>/<adapter-name>

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | ...             | ... min    | ... GB    | ...       | ...        |
| 16   | ...             | ... min    | ... GB    | ...       | ...        |
| 64   | ...             | ... min    | ... GB    | ...       | ...        |
| Base | -               | -          | -         | ...       | ...        |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png]
- Quan sát: <có / không có overfitting? Lý do?>

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: <improved / same / degraded?>

[... lặp lại 4 examples nữa ...]

## 5. Conclusion về Rank Trade-off

<Tối thiểu 100 từ. Trả lời 3 câu hỏi:>
- Rank nào cho ROI tốt nhất trên dataset này? Tại sao?
- Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?
- Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?

## 6. What I Learned
- <Bullet 1: insight cá nhân>
- <Bullet 2: insight cá nhân>
- <Bullet 3: optional>
```

---

**Chúc các bạn lab vui vẻ! 🚀**

> Câu hỏi → Slack channel `#lab21-help`
