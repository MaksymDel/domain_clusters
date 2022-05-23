# Automatic Domains for Bergamot

This repository implements clustering pipeline from [Translation Transformers Rediscover Inherent Data Domains](https://aclanthology.org/2021.wmt-1.65.pdf).

## Setup

```bash
conda create -n dc python=3.9
conda activate dc
pip install -r requirements.txt
```

## Run

### Step 1: convert Marian model to HuggingFace

```bash

# in: marian model, decoder config, spm model, vocab
# out: populated hf-model-dir

python convert_marian_bergamot_to_pytorch_.py \
            --npz-model-path artifacts/marian_checkpoint_files/model.npz \
            --yml-decoder-path artifacts/marian_checkpoint_files/model.npz.decoder.yml \
            --spm-model-path artifacts/vocab/vocab.spm \
            --vocab-path artifacts/vocab/vocab.vocab \
            --dest-dir artifacts/domain_clusters/marian_model_hf

```

### Step 2: extract features

```bash

# in: populated hf-model-dir, txt-dataset
# out: serialized sentence embeddings file

python extract_sentence_representations.py \
            --hf-model-dir artifacts/domain_clusters/marian_model_hf \
            --txt-dataset-path artifacts/data/source_dataset.txt \
            --batch-size 1000 \
            --layer-num 4 \
            --out-filename artifacts/domain_clusters/dataset_embedded.npz #--gpu

```

### Step 3: train k-means model and get cluster labels

```bash

# in: serialized sentence embeddings file
# out: k-means model, cluster ids for dataset 

python run_clustering.py \
            --embedded-dataset-path artifacts/domain_clusters/dataset_embedded.npz \
            --out-file-model artifacts/domain_clusters/kmeans_model.dump \
            --out-file-labels artifacts/domain_clusters/cluster_labels.txt \
            --n-clusters 4 \
            --n_init 10 \
            --max-iter 300\
            --batch-size 1024 \
            --max-no-improvement-size 10 \
            --verbose 0 \
            --random-state 42 # --batched

```


## Citation

```bibtex
@inproceedings{del-etal-2021-translation,
    title = "Translation Transformers Rediscover Inherent Data Domains",
    author = "Del, Maksym  and
      Korotkova, Elizaveta  and
      Fishel, Mark",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.65",
    pages = "599--613",
    abstract = "Many works proposed methods to improve the performance of Neural Machine Translation (NMT) models in a domain/multi-domain adaptation scenario. However, an understanding of how NMT baselines represent text domain information internally is still lacking. Here we analyze the sentence representations learned by NMT Transformers and show that these explicitly include the information on text domains, even after only seeing the input sentences without domains labels. Furthermore, we show that this internal information is enough to cluster sentences by their underlying domains without supervision. We show that NMT models produce clusters better aligned to the actual domains compared to pre-trained language models (LMs). Notably, when computed on document-level, NMT cluster-to-domain correspondence nears 100{\%}. We use these findings together with an approach to NMT domain adaptation using automatically extracted domains. Whereas previous work relied on external LMs for text clustering, we propose re-using the NMT model as a source of unsupervised clusters. We perform an extensive experimental study comparing two approaches across two data scenarios, three language pairs, and both sentence-level and document-level clustering, showing equal or significantly superior performance compared to LMs.",
}
```
