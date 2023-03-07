from tqdm import tqdm
import pandas as pd

import os
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

import argparse

# ---------------------------------------------------------


def main(model_path, force):
    meddra = pd.read_pickle("../dataset/llt_to_pt/meddra_data.pkl")
    meddra = meddra[["ENG", "level"]]
    meddra = meddra[(meddra.level == "PT")]  # |(meddra.level=="LLT")]
    meddra.ENG = meddra.ENG.str.lower()
    print("meddra terms for each level")
    print(meddra.level.value_counts())

    dataset = Dataset.from_pandas(meddra)

    def tokenize_function(examples):
        return tokenizer(examples["ENG"], padding="longest")

    def embed_function(examples):
        del_keys = [
            k for k in examples if k not in ["input_ids", "token_type_ids", "attention_mask"]
        ]
        for k in del_keys:
            del examples[k]
        examples = {k: v.to(device) for k, v in examples.items()}
        with torch.no_grad():
            out = model(**examples)[0].cpu()
        return dict(CLS=out[:, 0, :].numpy(), AVG=out.mean(axis=1).numpy())

    original_model_path = model_path

    model_path = "local" + model_path.replace("/", "|")

    output_path = f"{model_path}/meddra_embed_coder.pkl"

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        output_path = f"{model_path}/meddra_embed_coder.pkl"

    if os.path.exists(output_path) and not force:
        print("embeddings already exist at", output_path)
        return

    model = AutoModel.from_pretrained(original_model_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, config=model.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(">> tokenizing")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "token_type_ids", "attention_mask"], output_all_columns=True
    )

    print(">> embedding")
    tokenized_dataset = tokenized_dataset.map(embed_function, batched=True, batch_size=4096)

    meddra["coder_cls"] = None
    meddra["coder_avg"] = None

    print(">> transfering to dataframe")
    for sample in tqdm(tokenized_dataset):
        meddra.at[sample["__index_level_0__"], "coder_cls"] = sample["CLS"]
        meddra.at[sample["__index_level_0__"], "coder_avg"] = sample["AVG"]

    meddra.to_pickle(output_path)
    print("output serialized to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path of the coder model")
    args = parser.parse_args()

    main(args.path, False)
