# TDA-Enhanced Recommendation System


We integrate **Topological Data Analysis (TDA)** with **graph-based recommendation models** to improve robustness and informativeness in multimodal settings using image, text, and behavioral data.

---

## Repository Overview

```
TDA-Multimodal-Recommendation/
│
├── data/                         # Raw and processed datasets
│   ├── Baby/
│   ├── Digital_Music/
|   |--- Musical_Instruments/
│   └── sentence-transformers/
│
├── codes/                        # All model and training code
│   ├── main.py
│   ├── Models.py
│   ├── mean.py
│   └── utility/
│       ├── parser.py
|       ├── metrics.py
|       ├── load_data.py
│       └── batch_test.py
│
├── README.md                     
├── requirements.txt              
├── LICENSE                       
├── .gitignore                   
└── logs/                         
```

---

## Dataset Setup

1. Download the following from the [Amazon Review Dataset](http://jmcauley.ucsd.edu/data/amazon/links.html):
   - `meta-CATEGORY.json.gz`
   - `reviews_CATEGORY_5.json.gz`
   - `ratings_CATEGORY.csv`
   - `image_features_CATEGORY.b`

2. Place them as:
```
data/Baby/meta-data/
├── meta-Baby.json.gz
├── reviews_Baby_5.json.gz
├── ratings_Baby.csv
└── image_features_Baby.b
```

---

## Feature Extraction

### Textual Features (SBERT)

```bash
cd data/sentence-transformers
git lfs install
git clone https://huggingface.co/all-MiniLM-L6-v2
```

### Word2Vec Text Embeddings

```bash
cd data
python text_aux.py
```

---

## TDA Feature Computation
To build the TDA-augmented graph features:
```bash
python data/build_data_TDA.py
```

This computes:
- Betti curves
- Persistence landscapes
- Entropy
- Silhouettes
- Total persistence and lifespan

---

## Running Experiments

Move to the `codes/` directory:

```bash
cd codes
```

### Run baseline LATTICE

```bash
python main.py --model lattice --dataset Baby
```

### Add TDA for text/image

```bash
python main.py --model lattice --dataset Baby --textTDA True --imageTDA True
```

### Use dynamic TDA models

```bash
# TDA computed once (before training)
python main.py --model lattice_tda_first_graph --dataset Baby

# TDA computed each iteration (more dynamic, but slower)
python main.py --model lattice_tda_each_graph --dataset Baby

# TDA + drop least topologically relevant nodes
python main.py --model lattice_tda_drop_nodes --dataset Baby --percentNodesDropped 1
```

---

## Evaluation Metrics

- **Recall@K**
- **Precision@K**
- **NDCG@K**

Evaluations are logged during training for both validation and test sets.

---

## Requirements

Install required libraries:

```bash
pip install -r requirements.txt
```

Includes:
- `torch`, `numpy`, `scipy`
- `giotto-tda`, `gudhi`
- `scikit-learn`
- `sentence-transformers`
- `tqdm`

---

## Models Supported

| Model                      | Description                                                 |
|---------------------------|-------------------------------------------------------------|
| `lattice`                 | Baseline model with modality fusion                         |
| `lattice_tda_first_graph`| TDA computed once, fused with LightGCN embeddings           |
| `lattice_tda_each_graph` | TDA computed at each training step                          |
| `lattice_tda_drop_nodes` | Prunes least topologically-relevant items from the graph    |
| `lightgcn`, `ngcf`, `mf` | Standard CF baselines                                       |

---

## 🧾 Citation

If you use this work, please cite:

```bibtex
@article{bachiri_et_al,
  title={Topological Data Analysis and Graph-Based Learning for Multimodal Recommendation},
  author={Khalil BACHIRI, Ali YAHYAOUY, Maria MALEK, Nicoleta ROGOVSCHI},
  journal={IEEE Access},
  year={2025},
  note={Submitted},
}
```

---
