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