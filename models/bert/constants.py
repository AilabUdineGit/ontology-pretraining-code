from transformers import BertForSequenceClassification, AutoModelForSequenceClassification

# ---------------------------------------------------

location_related_const = {
    "default": dict(
        BASE_PATH="../",
        MODEL_BASE_PATH="../",
        DS_CONFIG_FILE="ds_config_local.json",
        GRADIENT_ACCUMULATION_STEPS=4,
        BATCH_SIZE=4,
    ),
}

model_related_const = {
    "pubmedbert": dict(
        MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        TOKENIZER_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        MODEL_CLASS=BertForSequenceClassification,
        FROM_PATH=False,
    ),
    "dilbert": dict(
        MODEL_NAME="beatrice-portelli/DiLBERT",
        TOKENIZER_NAME="beatrice-portelli/DiLBERT",
        MODEL_CLASS=BertForSequenceClassification,
        FROM_PATH=False,
    ),
    "bert": dict(
        MODEL_NAME="bert-base-uncased",
        TOKENIZER_NAME="bert-base-uncased",
        MODEL_CLASS=BertForSequenceClassification,
        FROM_PATH=False,
    ),
    "coder": dict(
        MODEL_NAME="GanjinZero/coder_eng",
        TOKENIZER_NAME="GanjinZero/coder_eng",
        MODEL_CLASS=AutoModelForSequenceClassification,
        FROM_PATH=False,
    ),
    # ----------------------------------------------------------------
    "coder_pre(30)": dict(
        MODEL_NAME="epoch_30|llt_to_pt|run_0|coder|bs_16/model",
        TOKENIZER_NAME="GanjinZero/coder_eng",
        MODEL_CLASS=AutoModelForSequenceClassification,
        FROM_PATH=True,
    ),
    "pubmedbert_pre(30)": dict(
        MODEL_NAME="epoch_30|llt_to_pt|run_0|pubmedbert|bs_16/model",
        TOKENIZER_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        MODEL_CLASS=BertForSequenceClassification,
        FROM_PATH=True,
    ),
    "bert_pre(30)": dict(
        MODEL_NAME="epoch_30|llt_to_pt|run_0|bert|bs_16/model",
        TOKENIZER_NAME="bert-base-uncased",
        MODEL_CLASS=BertForSequenceClassification,
        FROM_PATH=True,
    ),
}
