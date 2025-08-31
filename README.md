# Entity-Driven Captioning for Chest X-ray Images
**CNN encoders + Transformer decoder with report-aware text embeddings**

An entity-aware captioner that fuses **Chest X-ray features** with **diagnostic report context** to generate short, clinically meaningful captions. CheXNet (DenseNet-121), ResNet, and EfficientNet are compared with a Transformer decoder and evaluated with **BLEU**, **METEOR**, **CIDEr**, and **BioBERT cosine similarity**.

- Main notebook: `notebooks/caption-model.ipynb`
- Write-up: `reports/Data_Analysis_Report.pdf`

## Why this
The decoder attends jointly to **image features** and **report embeddings** (token or **T5-small** fine-tuned on CXR text) so captions mention the entities clinicians care about.

## What’s included
- Entity-aware fusion of X-ray features and report embeddings  
- Transformer decoder (multi-head attention, sinusoidal positions, LayerNorm, Dropout)  
- Comparative study across CNN encoders and text embeddings  
- Metrics: lexical (BLEU/METEOR/CIDEr) and semantic (BioBERT cosine)

## Architecture
![Model architecture](reports/figures/architecture.png)
## Data
IU Chest X-ray (Kaggle), ~7,470 images from 3,955 patients with paired reports.

## Method (brief)
- **Encoders:** CheXNet / ResNet-50 / EfficientNet → pooled features → linear projection to `d_model`  
- **Text:** learned token embeddings or **T5-small** fine-tuned on *findings/impressions*  
- **Decoder:** Transformer block with multi-head attention and sinusoidal positional encoding  
- **Inference:** `<start>` → greedy/beam until `<end>` or max length

## Results
- Best semantic alignment from **EfficientNet + Transformer (+T5)**  
- Lexical scores modest (concise phrasing); **cosine similarity high** indicating key findings are captured

![Training curves (accuracy & loss)](reports/figures/acc_loss_table.png)
![Caption metrics (BLEU/METEOR/CIDEr/Cosine)](reports/figures/metrics_table.png)
![Qualitative examples: generated vs. reference](reports/figures/example_captions.png)

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/caption-model.ipynb
