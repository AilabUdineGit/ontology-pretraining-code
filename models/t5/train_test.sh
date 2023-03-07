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