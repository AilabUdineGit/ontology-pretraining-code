import argparse
import json
import os

import torch

# ---------------------------------------------------
from cli import setup_parser
from dataset import generate_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def main(args):
    max_length = 64

    torch.manual_seed(42)

    # model_name
    if args.train_load_path is None:
        model = args.model
    else:
        model = os.path.join(args.train_load_path, "model")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)

    train_dataset = generate_dataset(args, "train", tokenizer, max_length)
    if args.dataset == "meddra":
        checkpoint_output_path = os.path.join(
            args.train_save_path, "model"
        )  # f"../../{args.split}/output/{base_model}/"
    else:
        checkpoint_output_path = os.path.join(
            args.train_save_path, args.dataset, f"run_{args.run}", "model"
        )  # f"../../{args.split}/output/{base_model}/{args.dataset}/run_{args.run}/"

    os.makedirs(checkpoint_output_path, exist_ok=True)

    training_args = TrainingArguments(
        # ------------------------------------------------------- [epochs and batch size]
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        # ------------------------------------------------------- [hyperparams]
        warmup_steps=100,
        weight_decay=0.01,
        # ------------------------------------------------------- [save and logging]
        output_dir=checkpoint_output_path,
        overwrite_output_dir=True,
        do_eval=False,
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
    trainer.save_model(checkpoint_output_path)
    tokenizer.save_pretrained(checkpoint_output_path)

    with open(checkpoint_output_path + "/../args.json", "w") as fp:
        json.dump(args.__dict__, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)

    args, _ = parser.parse_known_args()
    main(args)
