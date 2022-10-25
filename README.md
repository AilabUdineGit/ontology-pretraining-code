# Ontology Pretraining code

This code could produce some errors. The updated version will be released soon.

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

