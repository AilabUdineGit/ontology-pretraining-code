<h1 align="center"> <p>Ontology Pretraining</p></h1>
<h3 align="center"> Generalizing over Long Tail Concepts for Medical Term Normalization </h3>


This repository contains the source code used for the experimental session of the ["Generalizing over Long Tail Concepts for Medical Term Normalization"](https://aclanthology.org/2022.emnlp-main.588/) paper.

⚠️ This code could produce some errors. The updated version will be released soon.

## Datasets

The datasets used for the experimental session are in the 
`data` folder, expert for `PROP` that cannot be publicly released.

In the `train.csv` and `test.csv` files the relevant columns refer to:

* `ae`: the *ADE* in the original sample text
* `term`: the preferred term *PT*
* `term_llt_or_pt`: the original LLT/PT

We don't have permission to share [MedDRA](https://www.meddra.org)
(or parts of it), so to perform the **ontology pretraining** (OP)
you have to [download it](https://www.meddra.org/subscription/process)
by yourself.

## Models execution

### PubMedBERT

```bash
$model="pubmedbert"
$split=0
$dataset="smm4h"
$output_folder="output"

python train.py --model $model\
    --dataset $dataset \
    --split $split \
    --epochs 10
  
python test.py --model $model \
    --dataset $dataset \
    --split $split \
    --outpath "../$output_folder/pubmedbert_ft"

```

### GPT-2

```bash
$model="gpt2_pre(30)"
$split=0
$dataset="smm4h"
$output_folder="output"

python train.py --model $model\
    --dataset $dataset \
    --split $split \
    --epochs 10
  
python test.py --model $model \
    --dataset $dataset \
    --split $split \
    --outpath "../$output_folder/pubmedbert_ft"
```

### Sci5

#### Execution fine-tuning (+ test)

```bash
$model="razent/SciFive-base-Pubmed"
$output_folder="output"
$model_name="sci5_ft"
$dataset="smm4h"
$dataset_test="smm4h"
$run=0

python3 main.py --dataset=$dataset \
    --dataset_test=$dataset_test \
    --model=$model \
    --train_save_path="../$output_folder/output/$model_name" \
    --test_load_path="../$output_folder/output/$model_name" \
    --results_save_path="../$output_folder/results/$model_name" \
    --split=$output_folder \
    --train \
    --test \
    --ft \
    --run $run \
    --epochs 15 \
    --batch_size 32 \
    --accumulation 2
```

```bash
$model="razent/SciFive-base-Pubmed"
$output_folder="output"
$model_name="sci5_pt"
$dataset="meddra"

python3 main.py --dataset=$dataset \
    --model=$model \
    --train_save_path="../$output_folder/output/$model_name" \
    --split=$output_folder \
    --train \
    --epochs 40 \
    --batch_size 128 \
    --accumulation 2
```


## Cite

```
@inproceedings{portelli-etal-2022-generalizing,
    title = "Generalizing over Long Tail Concepts for Medical Term Normalization",
    author = "Portelli, Beatrice  and
      Scaboro, Simone  and
      Santus, Enrico  and
      Sedghamiz, Hooman  and
      Chersoni, Emmanuele  and
      Serra, Giuseppe",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.588",
    pages = "8580--8591"
}
```
