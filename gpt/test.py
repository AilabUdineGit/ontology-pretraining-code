from MeddraPredictors import MeddraPredictor_unconstrained_multiple
# ---------------------------------------------------
import argparse
import os
import pandas as pd
import torch
# ---------------------------------------------------
from cli import setup_parser
from constants import model_related_const
# ---------------------------------------------------
from sklearn.metrics import classification_report
import numpy as np
from loader import GeneralDataset

def main(train):
    
    DEVICE = "cuda"
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)
    parser.add_argument("-t", "--test_dataset", default=None, help="name of the test dataset if different from the training dataset")
    parser.add_argument("-o", "--outpath", required=True, help="name of the main output dir for this model")
    
    args, _ = parser.parse_known_args()

    GRADIENT_ACCUMULATION_STEPS = 4
    BATCH_SIZE = 4
    BEFORE_SAMPLE = "INPUT:"
    BEFORE_LABEL = "\nMEANING:"
    MODEL_BASE_PATH = "./@trained_models/"
    
    MODEL_NAME, TOKENIZER_NAME, MODEL_CLASS, FROM_PATH  = \
    model_related_const[args.model].values()
    

    if FROM_PATH:
        MODEL_NAME = MODEL_BASE_PATH+MODEL_NAME
    
    MODEL = args.model
    SPLIT = args.split
    WEIGHTS_OUTPUT_DIR = f"{MODEL_BASE_PATH}epoch_{args.epochs}|{args.dataset}|run_{SPLIT if args.dataset != 'llt_to_pt' else 0}|{args.model}|bs_{BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}"
    
    if args.test_dataset is not None:
        args.dataset = args.test_dataset
    
    RESULTS_OUTPUT_DIR = f"{args.outpath}/{args.dataset}/run_{SPLIT}"
    
    if not os.path.exists(RESULTS_OUTPUT_DIR):
        os.makedirs(RESULTS_OUTPUT_DIR)
    print(RESULTS_OUTPUT_DIR)
        
    dataset = GeneralDataset()
    generator = dataset._generate_examples(f"../dataset/{args.dataset}/run_{SPLIT}/test.csv", "test")
    test_samples = [dict(
        full_text = samp["full_text"],
        extraction = samp["text"],
        label = samp["label"],
    ) for (_,samp) in generator]
    
    df_test = pd.read_csv(f"../dataset/{args.dataset}/run_{SPLIT}/test.csv")
    
    mymodel = MeddraPredictor_unconstrained_multiple(WEIGHTS_OUTPUT_DIR+"/model", BEFORE_SAMPLE, BEFORE_LABEL).to(DEVICE)
    GENERATED = mymodel(test_samples)
    report(df_test, GENERATED, RESULTS_OUTPUT_DIR)

def report(df_test, GENERATED, RESULTS_OUTPUT_DIR):
        
    df_test['model_generated'] = [[y.lower() for y in x] for x in GENERATED]
    out_df = df_test


    def compute_accuracy(real_labels, all_preds):
        accs = []
        for k in range(10):
            acc_n = np.mean([1 if r in a[:k+1] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
            accs.append(round(acc_n, 4))
        return accs

    x = compute_accuracy(out_df.term, out_df.model_generated)
    x = pd.DataFrame(x, columns=["acc@k"])
    print(x["acc@k"].tolist())
    out_df=out_df.drop(columns=["text", "term_code", "terms_llt_or_pt"], errors='ignore')
    out_df.to_csv(f"{RESULTS_OUTPUT_DIR}/preds.csv", index=False)
    x.to_csv(f"{RESULTS_OUTPUT_DIR}/acc.txt", index=False)

if __name__ == "__main__":
    
    main(True)

