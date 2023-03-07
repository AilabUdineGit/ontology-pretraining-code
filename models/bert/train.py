import argparse

# ---------------------------------------------------
import json
import os
import sys

import pandas as pd
import torch

# ---------------------------------------------------
from cli import setup_parser
from constants import model_related_const
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main(train):
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)

    args, _ = parser.parse_known_args()

    GRADIENT_ACCUMULATION_STEPS = 4
    BATCH_SIZE = 4
    MODEL_BASE_PATH = "./@trained_models/"

    MODEL_NAME, TOKENIZER_NAME, MODEL_CLASS, FROM_PATH = model_related_const[args.model].values()

    if FROM_PATH:
        MODEL_NAME = MODEL_BASE_PATH + MODEL_NAME

    SPLIT = args.split
    WEIGHTS_OUTPUT_DIR = f"{MODEL_BASE_PATH}epoch_{args.epochs}|{args.dataset}|run_{SPLIT}|{args.model}|bs_{BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}"

    if not os.path.exists(WEIGHTS_OUTPUT_DIR):
        os.makedirs(WEIGHTS_OUTPUT_DIR)

    if not os.path.exists("label_to_int.json"):
        MEDDRA_TERMS = pd.read_pickle("../../data/data_meddra/meddra_data.pkl")
        MEDDRA_TERMS = MEDDRA_TERMS.ENG[(MEDDRA_TERMS.level == "PT")].str.lower().tolist()
        label_to_int = {term: idx for idx, term in enumerate(MEDDRA_TERMS)}
        int_to_label = {idx: term for idx, term in enumerate(MEDDRA_TERMS)}

        with open("label_to_int.json", "w") as f:
            json.dump(label_to_int, f, indent=4)
        with open("int_to_label.json", "w") as f:
            json.dump(int_to_label, f, indent=4)

    else:
        with open("label_to_int.json", "r") as f:
            label_to_int = json.load(f)
        with open("int_to_label.json", "r") as f:
            int_to_label = json.load(f)

    # ---------------------------------------------------
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_to_int)
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 512
    model.eval()
    # ---------------------------------------------------

    # ---------------------------------------------------
    def make_tokenize_function(max_len):
        if max_len is None:
            return lambda examples: tokenizer(examples["text"], padding="longest", truncation=True)
        else:
            return lambda examples: tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=max_len
            )

    load_dataset_args = dict(
        path="loader.py",
        data_dir=f"{args.dataset}|{SPLIT}",
    )
    data = load_dataset(
        **load_dataset_args, download_mode="force_redownload", ignore_verifications=True
    )
    data_tmp = data.map(make_tokenize_function(None), batched=True)

    max_seq_len = 0
    for split in ["train", "test"]:
        ls = [len(samp["input_ids"]) for samp in data_tmp[split]]
        ls = max(ls)
        max_seq_len = max(max_seq_len, ls)
    print(max_seq_len)

    data = data.map(make_tokenize_function(max_seq_len), batched=True)

    train_set = data["train"]

    class MyDataset(Dataset):
        def __init__(self, data):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []

            for samp in data:
                if samp["label"] in label_to_int:
                    self.input_ids.append(torch.tensor(samp["input_ids"]))
                    self.attn_masks.append(torch.tensor(samp["attention_mask"]))
                    self.labels.append(torch.tensor(label_to_int[samp["label"]]))

        #                     print(samp["label"])

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

    train_dataset = MyDataset(train_set)

    training_args = TrainingArguments(
        # ------------------------------------------------------- [epochs and batch size]
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE if args.dataset != "llt_to_pt" else 32,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        if args.dataset != "llt_to_pt"
        else 4,
        # ------------------------------------------------------- [hyperparams]
        warmup_steps=100,
        weight_decay=0.01,
        # ------------------------------------------------------- [save and logging]
        output_dir=WEIGHTS_OUTPUT_DIR,
        overwrite_output_dir=True,
        do_eval=False,
        logging_strategy="epoch",  # activate if interested
        save_strategy="no",
        save_total_limit=None,
        # -------------------------------------------------------
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f[0] for f in data]),
            "attention_mask": torch.stack([f[1] for f in data]),
            "labels": torch.stack([f[2] for f in data]),
        },
    )

    trainer.train()
    trainer.save_model(f"{WEIGHTS_OUTPUT_DIR}/model/")
    tokenizer.save_pretrained(f"{WEIGHTS_OUTPUT_DIR}/model/")

    return f"trained {args.model} on {args.dataset}"


if __name__ == "__main__":
    args = sys.argv[1:]
    main(True)
