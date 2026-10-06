# Who Wrote It? Attribution of LLM Text with CNNs and LSTMs

CMPSC 448 midterm project (Fall 2026). The task: given only the text an LLM produced, predict which lab built it. Six families, two datasets, four models trained from scratch, four research questions.

Author: Harsh Rathi. Final report in report/report.pdf.

## Results in short

RQ1, identifying the model from output text alone. A Kim style 1D-CNN got 62.96% test accuracy (Macro-F1 0.6272) over 486 held out samples, beating a BiLSTM (56.17%) while running 6.8x faster at inference, 1.46 ms versus 9.98 ms per sample. TF-IDF baselines sit near 53%. Anthropic was the easiest family to spot at 89%; Qwen and DeepSeek confuse each other (33% of Qwen called DeepSeek), which tracks with their shared pretraining corpora.

RQ2, do prompts help? On the controlled benchmark where all six models answered identical prompts, prompt-only classification lands at 6.79%, below the 16.67% chance floor. The prompt carries no model information by construction. On wild LMSYS Arena logs, prompt-only jumps to 27.92% against a 25% baseline. That gap is human query selection bias, people send coding to GPT-4 and venting to Claude. A detector trained on uncurated logs learns the users, not the models.

RQ3 (extra credit), cross domain transfer. Trained on general QA, the CNN holds 48.57% on technical coding and 51.67% on creative writing. Anthropic's fingerprint is the most domain invariant (F1 0.6604); DeepSeek (0.1714) and Qwen (0.2114) depend heavily on code formatting and collapse outside technical text.

RQ4 (extra credit), what is the signal? Stylometric profiling per family plus three ablations. Stripping Markdown costs nothing (55.35%, up 1.03). Truncating every response to 150 words costs 8.02 points (54.32% down to 46.30%), so verbosity is a real macro signal. Punctuation and case carry only 0.62 points.

Model comparison (RQ1 test set):

| Model | Accuracy | Macro-F1 | Train s/epoch | Inference ms |
| --- | --- | --- | --- | --- |
| TF-IDF + LogReg | 53.50% | 0.5330 | 0.8 | 0.08 |
| TF-IDF + SVM | 52.06% | 0.5123 | 1.2 | 0.09 |
| BiLSTM | 56.17% | 0.5396 | 221.8 | 9.98 |
| 1D-CNN | 62.96% | 0.6272 | 51.7 | 1.46 |

## Data

Two sources, on purpose.

Dataset A, controlled. From Stanford's tatsulab/alpaca_eval. Six families (GPT-4 Turbo, Claude 2.1, Gemini Pro, Llama 3 70B, DeepSeek 67B, Qwen2 72B) each answered the same 805 instructions: 4,830 pairs. Splits go by prompt_id, 80/10/10, so no prompt repeats across splits.

Dataset B, in the wild. lmsys/chatbot_arena_conversations, 4,800 balanced conversations (1,200 per family) across OpenAI, Anthropic, Google, and ChatGLM-6B.

## Layout

```
.
├── README.md
├── requirements.txt
├── run_all.py                   # runs everything end to end
├── run_rq1.py                   # baselines, CNN, BiLSTM
├── run_rq2.py                   # input vs output vs both
├── run_rq3.py                   # cross domain transfer
├── run_rq4.py                   # stylometry and ablations
├── generate_figures.py          # 300 DPI figures
├── src/
│   ├── preprocess.py            # ingestion, cleaning, splits
│   ├── dataset.py               # torch datasets and loaders
│   ├── train.py                 # shared training loop
│   ├── evaluate.py              # metrics
│   └── models/
│       ├── baseline.py          # TF-IDF + LR/SVM
│       ├── cnn.py               # Kim 1D-CNN
│       └── rnn.py               # BiLSTM with attention pooling
├── data/                        # raw and processed (not all committed)
├── results/                     # rq1..rq4 json outputs
├── figures/                     # generated pngs
└── report/                      # report.html source + report.pdf
```

## Running it

Python 3.10 or newer. CPU is enough, the models here are small, a GPU just makes the BiLSTM epochs faster.

```bash
pip install -r requirements.txt
python run_all.py
```

Or each research question on its own:

```bash
python run_rq1.py
python run_rq2.py
python run_rq3.py
python run_rq4.py
python generate_figures.py
```

The report PDF is printed from report/report.html with headless Chrome:

```bash
google-chrome --headless --print-to-pdf=report/report.pdf \
  --no-pdf-header-footer "$(pwd)/report/report.html"
```

## References

1. CMPSC 448 lecture slides, chapter 4 (MLP, 1D-CNN, LSTM gating, self attention).
2. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP 2014.
3. Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8).
4. Tian, E., and Cui, A. (2023). GPTZero whitepaper.
5. Mitchell, E., et al. (2023). DetectGPT. ICML 2023.
6. Burrows, J. (2002). Delta: a measure of stylistic difference. Literary and Linguistic Computing.
7. Mosteller, F., and Wallace, D. L. (1964). Inference and Disputed Authorship: The Federalist.
8. Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
