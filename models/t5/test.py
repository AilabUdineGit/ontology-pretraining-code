import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------
from cli import setup_parser
from dataset import generate_dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration


def main(args):
    max_length = 64

    num_beams = 5

    torch.manual_seed(42)

    checkpoint_input_path = args.model

    if args.ft:
        checkpoint_input_path = os.path.join(
            args.test_load_path, args.dataset, f"run_{args.run}", "model"
        )
    else:
        checkpoint_input_path = os.path.join(args.test_load_path, "model")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_input_path)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_input_path)
    test_ds = generate_dataset(args, "test", tokenizer, max_length)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generated_detok, aes_detok, terms_detok = [], [], []

    with torch.no_grad():
        for sample in tqdm(test_dl, desc="testing"):
            input_ids = sample[0].to(device)
            attention_mask = sample[1].to(device)
            term = sample[2].to(device)

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams,
            )

            generated_detok.append(output)
            aes_detok.append(input_ids)
            terms_detok.append(term)

    generated_detok = [tokenizer.batch_decode(d, skip_special_tokens=True) for d in generated_detok]
    aes_detok = [tokenizer.batch_decode(d, skip_special_tokens=True) for d in aes_detok]
    terms_detok = [tokenizer.batch_decode(d, skip_special_tokens=True) for d in terms_detok]

    report(generated_detok, aes_detok, terms_detok, args)


def report(generated, aes, terms, args):
    aes = [x[0].lower()[len("normalize: ") :] for x in aes]
    terms = [x[0].lower() for x in terms]
    out_df = pd.DataFrame(
        {
            "ae": aes,
            "model_generated": [x[0].lower() for x in generated],
            "gold_label": terms,
        }
    )

    report = classification_report(
        out_df.gold_label, out_df.model_generated, zero_division=0, output_dict=True
    )
    A = report["accuracy"]
    B = report["macro avg"]
    C = report["weighted avg"]

    if args.dataset_test and args.dataset_test != args.dataset:
        save_path = os.path.join(
            args.results_save_path, f"{args.dataset}_to_{args.dataset_test}", f"run_{args.run}/"
        )
    else:
        save_path = os.path.join(args.results_save_path, args.dataset, f"run_{args.run}/")

    os.makedirs(save_path, exist_ok=True)

    with open(save_path + "metrics.txt", "w") as f:
        f.write(
            f"""
    accuracy, macro_avg_p, macro_avg_r, macro_avg_f1, support_1, weighted_avg_p, weighted_avg_r, weighted_avg_f1, support_2
    {round(A,4)}, {', '.join([str(round(x,4)) for x in B.values()])}, {', '.join([str(round(x,4)) for x in C.values()])}
    """
        )

    out_df = pd.DataFrame(
        {
            "ae": aes,
            "model_generated": [[y.lower() for y in x] for x in generated],
            "gold_label": terms,
        }
    )

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

    x = compute_accuracy(out_df.gold_label, out_df.model_generated)

    for i, y in enumerate(x):
        print(i + 1, "|", y)

    x = pd.DataFrame(x, columns=["acc@k"])
    out_df.to_csv(save_path + "preds.csv", index=False)
    x.to_csv(save_path + "acc.txt", index=False)

    with open(save_path + "args.json", "w") as fp:
        json.dump(args.__dict__, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)

    args, _ = parser.parse_known_args()

    main(args)
