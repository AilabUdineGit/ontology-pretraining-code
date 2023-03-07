from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

DEVICE = "cuda"
torch.manual_seed(42)


def batch(l, size=16):
    i = 0

    while i < len(l):
        i += size
        yield l[i - size : i]


class MeddraPredictor_unconstrained_multiple(nn.Module):
    def __init__(self, model_path="./trained_model/model/", before_sample="", before_label=""):
        super().__init__()

        self.MAX_GEN = 20
        self.before_sample = before_sample
        self.before_label = before_label

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.max_length = 512
        self.model = model
        self.model.eval()

    def forward(self, test_set):
        MAX_GEN = self.MAX_GEN
        BATCH_SIZE = 1

        print(self.model.device)

        before_sample = self.before_sample
        before_label = self.before_label

        GENERATED = []

        with torch.no_grad():
            for samps in tqdm(
                batch(test_set, BATCH_SIZE), desc="predicting", total=len(test_set) // BATCH_SIZE
            ):
                NUM_BEAMS = 10

                prompt_str = [
                    f"{self.tokenizer.bos_token}{before_sample} {samp['extraction']}{before_label}".strip()
                    for samp in samps
                ]

                prompt = self.tokenizer(prompt_str, return_tensors="pt", padding=True).to(
                    self.model.device
                )

                generated = self.model.generate(
                    prompt.input_ids,
                    attention_mask=prompt.attention_mask,
                    num_beams=NUM_BEAMS,
                    num_return_sequences=NUM_BEAMS,
                    max_length=len(prompt[0]) + 20,
                    min_length=0,
                )

                generated = list(batch(generated, NUM_BEAMS))

                generated = [
                    gen[:, prompt.attention_mask[idx].argmin().item() :]
                    if 0 in prompt.attention_mask[idx]
                    else gen[:, prompt.attention_mask[idx].shape[0] :]
                    for idx, gen in enumerate(generated)
                ]

                generated = [
                    [y.strip() for y in self.tokenizer.batch_decode(x, skip_special_tokens=True)]
                    for x in generated
                ]

                GENERATED += generated

        return GENERATED
