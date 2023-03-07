import pandas as pd
from torch.utils.data import Dataset


def generate_dataset(args, train_or_test, tokenizer, max_length):
    train_dataset = args.dataset
    test_dataset = args.dataset_test if args.dataset_test else train_dataset
    if train_or_test == "test":
        return TestDT(tokenizer, max_length, test_dataset, args.run)
    if train_or_test == "train":
        if train_dataset == "meddra":
            return MeddraDT(tokenizer, max_length)
        else:
            return FineTuningDT(tokenizer, max_length, train_dataset, args.run)


class PrototypeDataset(Dataset):
    def encode(self, data, max_length, tokenizer):
        return tokenizer.batch_encode_plus(
            data,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def create_input_and_labels(self, df, max_length, tokenizer):
        input_samples = self.encode(df.ae.tolist(), max_length, tokenizer)
        label_samples = self.encode(df.term.tolist(), max_length, tokenizer)

        self.input_ids = input_samples.input_ids
        self.attention_mask = input_samples.attention_mask
        self.label_ids = label_samples.input_ids
        self.label_attention_mask = label_samples.attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.label_ids[idx],
            self.label_attention_mask[idx],
        )


class MeddraDT(PrototypeDataset):
    def __init__(self, tokenizer, max_length):
        path = "../dataset/llt_to_pt/run_0/train.csv"
        df = pd.read_csv(path, low_memory=False)
        df.loc[:, "prefix"] = "normalize"
        df["ae"] = df["prefix"] + ": " + df["ae"]

        self.create_input_and_labels(df, max_length, tokenizer)


class TestDT(PrototypeDataset):
    def __init__(self, tokenizer, max_length, dataset, run):
        path = f"../dataset/{dataset}/run_{run}/test.csv"
        df = pd.read_csv(path, low_memory=False)
        df.loc[:, "prefix"] = "normalize"
        df["ae"] = df["prefix"] + ": " + df["ae"]
        self.create_input_and_labels(df, max_length, tokenizer)


class FineTuningDT(PrototypeDataset):
    def __init__(self, tokenizer, max_length, dataset, run):
        path = f"../dataset/{dataset}/run_{run}/train.csv"
        df = pd.read_csv(path, low_memory=False)
        df.loc[:, "prefix"] = "normalize"
        df["ae"] = df["prefix"] + ": " + df["ae"]
        self.create_input_and_labels(df, max_length, tokenizer)
