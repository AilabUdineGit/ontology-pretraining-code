from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import json


DEVICE = "cuda"
torch.manual_seed(42)

class MeddraPredictor_multiple(nn.Module):
    
    def __init__(self, model_path="./trained_model/model/", before_sample="INPUT:", before_label="\nMEANING:"):
        super().__init__()
        
        self.MAX_GEN = 20
        self.before_sample = before_sample
        self.before_label = before_label
        
        with open("label_to_int.json", "r") as f:
            self.label_to_int = json.load(f)
        with open("int_to_label.json", "r") as f:
            self.int_to_label = json.load(f)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer = tokenizer
            
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(self.label_to_int))

        self.model = model
        
    def forward(self, test_set):
        
        MAX_GEN = self.MAX_GEN
        
        print(self.model.device)

        
        GENERATED = []
                
        with torch.no_grad():
            
            for samp in tqdm(test_set, desc="predicting"):

                prompt_str = samp["extraction"]
                prompt = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.model.device)
                
                label = torch.tensor(self.label_to_int[samp["label"]])

                                     
                out = self.model(prompt)
                logits = out.logits.squeeze()
            
                
                all_pred_label = logits.argsort(descending=True)[:20].tolist()
                all_pred_label = [self.int_to_label[str(x)] for x in all_pred_label]
                
                
                GENERATED.append(all_pred_label)
                
                
        return GENERATED
