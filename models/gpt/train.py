import argparse
import os
import sys

import torch

# ---------------------------------------------------
from cli import setup_parser
from constants import model_related_const
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

# ---------------------------------------------------


def main(train):
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)

    args, _ = parser.parse_known_args()

    GRADIENT_ACCUMULATION_STEPS = 4
    BATCH_SIZE = 4
    BEFORE_SAMPLE = "INPUT:"
    BEFORE_LABEL = "\nMEANING:"
    MODEL_BASE_PATH = "./@trained_models/"

    MODEL_NAME, TOKENIZER_NAME, MODEL_CLASS, FROM_PATH = model_related_const[args.model].values()

    if FROM_PATH:
        MODEL_NAME = MODEL_BASE_PATH + MODEL_NAME

    SPLIT = args.split
    WEIGHTS_OUTPUT_DIR = f"{MODEL_BASE_PATH}epoch_{args.epochs}|{args.dataset}|run_{SPLIT}|{args.model}|bs_{BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}"

    if not os.path.exists(WEIGHTS_OUTPUT_DIR):
        os.makedirs(WEIGHTS_OUTPUT_DIR)

    # ---------------------------------------------------
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        TOKENIZER_NAME,
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.cls_token
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    model = MODEL_CLASS.from_pretrained(MODEL_NAME, is_decoder=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 512
    model.eval()
    # ---------------------------------------------------

    # ---------------------------------------------------
    load_dataset_args = dict(
        path="loader.py",
        data_dir=f"{args.dataset}|{SPLIT}",
    )
    data = load_dataset(
        **load_dataset_args, download_mode="force_redownload", ignore_verifications=True
    )
    train_set = data["train"]
    # ---------------------------------------------------

    before_sample = BEFORE_SAMPLE if len(BEFORE_SAMPLE) > 0 else ""
    before_label = BEFORE_LABEL if len(BEFORE_LABEL) > 0 else ""

    max_length = min(
        256,
        max(
            [
                len(
                    tokenizer.encode(
                        f"{tokenizer.bos_token}{before_sample} {samp['text']}{before_label} {samp['label']}{tokenizer.eos_token}".strip()
                    )
                )
                for samp in train_set
            ]
        ),
    )

    print("MAX_LENGTH:", max_length)

    class MyDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []

            before_sample = BEFORE_SAMPLE if len(BEFORE_SAMPLE) > 0 else ""
            before_label = BEFORE_LABEL if len(BEFORE_LABEL) > 0 else ""

            for samp in data:
                gold_label = samp["label"]
                gold_text = f"{tokenizer.bos_token}{before_sample} {samp['text']}{before_label} {gold_label}{tokenizer.eos_token}".strip()
                encodings_dict = tokenizer(
                    gold_text, truncation=True, max_length=max_length, padding="max_length"
                )
                self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
                self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))
                self.labels.append(gold_label)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

    train_dataset = MyDataset(train_set, tokenizer, max_length=max_length)

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
        # logging_steps=100,
        # logging_dir='./logs',
        save_strategy="steps" if args.dataset == "llt_to_pt" else "no",
        save_steps=380 * 10,
        # options for save_strategy: no, epoch, steps
        # choosing "no" and saving manually at the end
        # save_steps=500, # <--- if "steps", choose when to save
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
            "labels": torch.stack([f[0] for f in data]),
        },
    )

    trainer.train()
    trainer.save_model(f"{WEIGHTS_OUTPUT_DIR}/model/")
    tokenizer.save_pretrained(f"{WEIGHTS_OUTPUT_DIR}/model/")

    return f"trained {args.model} on {args.dataset}"


if __name__ == "__main__":
    args = sys.argv[1:]
    main(True)
