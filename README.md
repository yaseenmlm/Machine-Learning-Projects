# Machine Learning Projects

Jupyter notebooks exploring NLP and applied machine learning.

## Projects
| Notebook | Topic | Summary |
|---|---|---|
| [Disaster_Tweet_Classifier.ipynb](Disaster_Tweet_Classifier.ipynb) | Text classification | Fine-tunes DistilBERT (67M params) in PyTorch with Hugging Face `Trainer` on 7,613 labelled tweets. fp16 mixed precision, linear warmup, weight decay and early stopping reach **81.5% accuracy / 0.81 weighted F1** on 1,142 held-out tweets. Error analysis shows minority-class recall (0.76) is the main weakness. |
| [Company_Sentiment_BERTopic.ipynb](Company_Sentiment_BERTopic.ipynb) | Topic modelling & sentiment | Sentence-transformer embeddings → UMAP → HDBSCAN via BERTopic over 2,000+ tweets, surfacing **58 topics**, with Matplotlib/Plotly views of topic volume and per-topic sentiment. |
| [Kaggle_Project.ipynb](Kaggle_Project.ipynb) | Kaggle | Kaggle notebook. |

## Tech
Python · PyTorch · Hugging Face Transformers · BERTopic · UMAP · HDBSCAN · scikit-learn · pandas · NumPy · Matplotlib · Plotly

## Run locally
```bash
git clone https://github.com/yaseenmlm/Machine-Learning-Projects.git
cd Machine-Learning-Projects
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install jupyter pandas scikit-learn torch transformers bertopic plotly
jupyter notebook
```
