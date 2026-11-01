# Calibration of Arabic Sentiment Models Under Orthographic Noise

**ARTI 402 – Deep Learning · Course Project**
Imam Abdulrahman Bin Faisal University · College of Computer Science & Information Technology
Course Instructor: Dr. Mustafa Youldash

| Name | Student ID | Role |
|---|---|---|
| Maisoon ALabdullah | 2240005251 | Team Leader |
| Jolie Salama | 2240004580 | Team Member |
| Wajd Alkhaldi | 2240005544 | Team Member |
| Manar Alghamdi | 2240005128 | Team Member |

---

## About this project

Sentiment models decide whether a review is positive, negative or mixed. Along with each answer, they also say **how sure they are** (their *confidence*). For a model to be trusted, that confidence must be **honest**: when it says "90% sure", it should be right about 90% of the time. This is called **calibration**.

Real Arabic reviews are full of spelling mistakes: missing hamzas, ة written as ه, confused dots, stretched letters, and keyboard typos. This project asks:

1. Do Arabic sentiment models stay **honest about their confidence** when reviews contain spelling mistakes?
2. Which methods can **fix** their confidence under noise?
3. Can the models safely **answer only when they are confident**, and send the rest to a human?

We fine-tuned three Arabic language models (**AraBERT, MARBERT, CAMeLBERT**), built a generator of realistic Arabic spelling mistakes, and compared five ways of making the models' confidence honest.

## Main findings

1. **All three models are overconfident**, even on clean text, and spelling mistakes make it worse.
2. **Not all mistakes are equal.** Hamza, ة/ه and ى/ي mistakes barely affect the models. **Dot confusion** (ضعيف → ضعيق) and **keyboard typos** do the most damage, and standard Arabic text normalization **cannot fix them**.
3. **Standard temperature scaling** fixes confidence on clean text but **fails under noise**.
4. **Transferable temperature scaling (TTS)** keeps confidence honest at **every** noise level, for all three models.
5. **MC dropout** helps only a little, at 10 times the computing cost.
6. **Noise-augmented training** improves accuracy on messy text by about **4 points**, but still needs TTS for honest confidence.
7. In a realistic **deployment test**, a 10% error target quietly grows to 15–22% on messy reviews without a fix; **TTS cuts this extra error roughly in half**.

### Key numbers

**Calibration error (ECE, 15 bins; lower is better), mean over 3 seeds, test reviews with 50% dot-confusion noise:**

| Model | No fix | MC dropout | TS | **TTS** | Noise-aug. + TTS |
|---|---|---|---|---|---|
| AraBERT | 0.132 | 0.098 | 0.066 | **0.026** | 0.027 |
| MARBERT | 0.095 | 0.057 | 0.064 | **0.041** | 0.046 |
| CAMeLBERT | 0.148 | 0.122 | 0.086 | 0.053 | **0.035** |

**Accuracy, mean over 3 seeds:**

| Model | Clean | 50% mixed noise (standard) | 50% mixed noise (noise-augmented) |
|---|---|---|---|
| AraBERT | 73.0% | 65.4% | **69.3%** |
| MARBERT | 72.3% | 65.3% | **68.7%** |
| CAMeLBERT | 71.4% | 63.5% | **67.9%** |

Full results are in [`results/paper_tables/`](results/paper_tables) and [`figures/`](figures).

---

## Dataset

**Arabic 100K Reviews** (Kaggle): https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews

About 100,000 Arabic reviews of hotels, books, movies, products and airlines, labelled **Positive**, **Negative** or **Mixed** (33,333 each). We use a balanced subset of **36,000 reviews**, split 70 / 15 / 15 into training (25,200), validation (5,400) and test (5,400).

> The dataset is **not** included in this repository, because it belongs to its Kaggle author. Please download it from the link above. The file [`data/splits/split_ids.csv`](data/splits/split_ids.csv) lists which reviews (by row number in the Kaggle file) are in each part, so the exact same split can be rebuilt.

---

## Repository contents

```
├── arabic_sentiment_calibration.ipynb   # The full project: one notebook, 10 sections
├── README.md                            # This file
├── requirements.txt                     # Library versions used
├── .gitignore                           # Keeps private and large files out of the repository
├── data/
│   └── splits/
│       └── split_ids.csv                # Which reviews are in train / validation / test
├── results/
│   ├── versions.json                    # Exact library versions recorded during training
│   └── paper_tables/                    # Tables 1–4 used in the paper (CSV)
│       ├── table1.csv                   # Accuracy: standard vs. noise-augmented training
│       ├── table2.csv                   # Calibration error (ECE) for every model and method
│       ├── table3.csv                   # Deployment test (selective prediction)
│       └── table4.csv                   # Text normalization as a defense
└── figures/                             # All charts used in the paper (PNG)
    ├── summary_ece_all_models.png       # Main figure: calibration error under noise, all methods
    ├── noise_curves_AraBERT_s42.png     # Accuracy and calibration error per noise type
    ├── reliability_AraBERT_s42.png      # Reliability diagrams: clean vs. heavy noise
    ├── ece_methods_mixed.png            # No fix vs. TS vs. TTS, per model
    ├── noiseaug_vs_standard.png         # Standard vs. noise-augmented training
    ├── selective_threshold_transfer.png # Deployment test: error and share accepted under noise
    └── risk_coverage_AraBERT_s42.png    # Error rate vs. share of reviews answered
```

The trained models (about 6 GB) and the dataset are not included. Running the notebook recreates the models; the dataset is downloaded from Kaggle.

**What the notebook contains:**

| Section | Content |
|---|---|
| 1 | Environment setup (GPU, Google Drive, settings) |
| 2 | Data download and exploration |
| 3 | Data cleaning and fixed train / validation / test split |
| 4 | Arabic spelling-noise generator (6 types × 5 levels) |
| 5 | Fine-tuning AraBERT, MARBERT and CAMeLBERT (3 seeds each) |
| 6 | Measuring accuracy and calibration |
| 7 | Calibration methods: TS, TTS, MC dropout, noise-augmented training, normalization |
| 8 | Selective prediction: answering only when confident |
| 9 | Final tables and figures |
| 10 | Summary and conclusions |

Every section starts with a plain-language explanation of what it does and what it found.

---

## How to run

The notebook is designed for **Google Colab** with a free or paid **T4 GPU**.

1. Open `arabic_sentiment_calibration.ipynb` in Google Colab (**File → Upload notebook**).
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Click **Runtime → Run all**.
   - At Section 1.2, allow Colab to access your Google Drive. All data, models and results are saved to `MyDrive/arabic_sa_calibration/`.
   - At Section 2.1, the notebook downloads the dataset automatically (or tells you where to upload it).
4. Wait. Running everything **from scratch takes about 15 hours of T4 GPU time** (18 training runs of about 30 minutes each, plus MC dropout).

**Colab disconnected? Just run all again.** Every long step saves its result to Google Drive and is **skipped** next time, so no finished work is repeated. With all results saved, a full re-run takes about 10–15 minutes.

**Running outside Colab:** the notebook uses Google Drive for storage. To run it elsewhere (e.g. Jupyter with a GPU), change the `PROJECT` folder path in Section 1.2 and install the libraries in `requirements.txt`.

---

## Reproducibility

- **Fixed random seeds:** 42 for data splitting and noise; 42, 43 and 44 for the three training runs of each model.
- **Fixed data split:** created once and saved; `split_ids.csv` rebuilds it from the Kaggle file.
- **Repeatable noise:** each review gets its own fixed random seed, so the noisy test sets are identical on every run.
- **Recorded library versions:** see `requirements.txt` and `results/versions.json`.
- **Results reported as mean ± standard deviation over 3 seeds.**
- Small differences in the last decimal place can appear between runs on different GPUs; the conclusions do not depend on them.

**Training settings:** learning rate 2e-5, batch size 32, 3 epochs (best epoch chosen by validation macro-F1), 10% warm-up, weight decay 0.01, dropout 0.1, maximum length 256 tokens, mixed precision (fp16).

**Models used (Hugging Face):**
- AraBERT: `aubmindlab/bert-base-arabertv02`
- MARBERT: `UBC-NLP/MARBERTv2`
- CAMeLBERT: `CAMeL-Lab/bert-base-arabic-camelbert-mix`

---

## References

- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. (Temperature scaling)
- J. Zhang, W. Yao, X. Chen, L. Feng. *Transferable Post-hoc Calibration on Pretrained Transformers in Noisy Text Classification.* AAAI 2023. (Transferable temperature scaling)
- Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016. (MC dropout)
- A. Al-Laith, R. Kebdani. *Evaluating Calibration of Arabic Pre-trained Language Models on Dialectal Text.* WACL 2025.
- A. Alajmi, A. Salman, I. Ahmad. *Evaluation of Adversarial Robustness in Arabic Language Models.* arXiv 2026.

The full reference list is in the paper.
