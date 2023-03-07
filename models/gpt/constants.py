from transformers import GPT2LMHeadModel, GPTNeoForCausalLM

# ---------------------------------------------------

model_related_const = {
    "gptneo": dict(
        MODEL_NAME="EleutherAI/gpt-neo-1.3B",
        TOKENIZER_NAME="EleutherAI/gpt-neo-1.3B",
        MODEL_CLASS=GPTNeoForCausalLM,
        FROM_PATH=False,
    ),
    "gpt2": dict(
        MODEL_NAME="gpt2",
        TOKENIZER_NAME="gpt2",
        MODEL_CLASS=GPT2LMHeadModel,
        FROM_PATH=False,
    ),
    "gpt2_pre(30)": dict(
        MODEL_NAME="epoch_30|llt_to_pt|run_0|gpt2|bs_16/model",
        TOKENIZER_NAME="gpt2",
        MODEL_CLASS=GPT2LMHeadModel,
        FROM_PATH=True,
    ),
    "gpt2_pre(80)": dict(
        MODEL_NAME="epoch_80|llt_to_pt|run_0|gpt2|bs_16/model",
        TOKENIZER_NAME="gpt2",
        MODEL_CLASS=GPT2LMHeadModel,
        FROM_PATH=True,
    ),
}
