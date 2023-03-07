from tqdm import tqdm
import pandas as pd

import glob
import torch
import os

import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

import argparse

from get_other_embeddings import main as get_other_embeddings

# ---------------------------------------------------------


def main(model_path, outpath, dataset, split):
    outpath_avg = f"{outpath}/{dataset}/run_{split}"
    outpath_cls = f"{outpath}_cls/{dataset}/run_{split}"
    acc_path = lambda outpath: f"{outpath}/acc.txt"
    pred_path = lambda outpath: f"{outpath}/preds.csv"

    if not os.path.exists(outpath_avg):
        os.makedirs(outpath_avg)
    # if not os.path.exists(outpath_cls):
    #     os.makedirs(outpath_cls)

    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COL], padding="longest")

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

    def compute_accuracy(real_labels, all_preds):
        accs = []
        for k in range(10):
            acc_n = (
                np.mean([1 if r in a[: k + 1] else 0 for (r, a) in zip(real_labels, all_preds)])
                if all_preds is not None
                else -1
            )
            accs.append(round(acc_n, 4))
        return accs

    test_set = f"../dataset/{dataset}/run_{split}/test.csv"

    original_model_path = model_path

    model_path = "local" + model_path.replace("/", "|")

    if not os.path.exists(f"{model_path}/meddra_embed_coder.pkl") or args.force:
        print("embedding meddra vocabulary first")
        get_other_embeddings(original_model_path, args.force)

    if (
        os.path.exists(pred_path(outpath_avg))
        and os.path.exists(pred_path(outpath_cls))
        and (not args.force)
    ):
        print(pred_path(outpath_avg), "and", pred_path(outpath_cls), "already exist")
        return

    model = AutoModel.from_pretrained(original_model_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, config=model.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    meddra = pd.read_pickle(f"{model_path}/meddra_embed_coder.pkl")

    TEXT_COL = "ae"
    df = pd.read_csv(test_set)
    # df = df.drop(columns=["text", "model_generated"])
    dataset = Dataset.from_pandas(df)

    print(">> tokenizing")
    dataset = dataset.map(tokenize_function, batched=True, batch_size=None)
    dataset.set_format(
        "torch", columns=["input_ids", "token_type_ids", "attention_mask"], output_all_columns=True
    )
    print(">> embedding")
    dataset = dataset.map(embed_function, batched=True, batch_size=256)

    sim_function = torch.nn.CosineSimilarity(dim=1)

    all_meddra_cls = torch.stack(meddra.coder_cls.values.tolist())
    all_meddra_avg = torch.stack(meddra.coder_avg.values.tolist())

    df_avg = df.copy()
    df_avg["model_generated"] = None

    df_cls = df.copy()
    df_cls["model_generated"] = None

    for idx, sample in enumerate(tqdm(dataset, desc="calculating similarity")):
        sim = sim_function(sample["CLS"].unsqueeze(0), all_meddra_cls)
        sort_sim, indices = sim.sort(descending=True)
        df_cls.at[idx, "model_generated"] = [meddra.ENG.iloc[i] for i in indices[:10].tolist()]

        sim = sim_function(sample["AVG"].unsqueeze(0), all_meddra_avg)
        sort_sim, indices = sim.sort(descending=True)
        df_avg.at[idx, "model_generated"] = [meddra.ENG.iloc[i] for i in indices[:10].tolist()]

    # for data, op in zip([df_cls, df_avg], [outpath_cls, outpath_avg]):
    for data, op in zip([df_avg], [outpath_avg]):
        data.to_csv(pred_path(op), index=None)
        x = compute_accuracy(data.term, data.model_generated)
        print(x)
        x = pd.DataFrame(x, columns=["acc@k"])
        x.to_csv(acc_path(op), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path of the coder model")
    parser.add_argument(
        "-o", "--outpath", required=True, help="name of the main output dir for this model"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["cadec", "irms", "smm4h"],
        help="dataset to test on",
    )
    parser.add_argument(
        "-s", "--split", required=True, choices=["0", "1", "2", "3"], help="data split to test on"
    )
    parser.add_argument("-f", "--force", action="store_true", help="force to recompute embeddings")
    args = parser.parse_args()

    main(args.path, args.outpath, args.dataset, args.split)
