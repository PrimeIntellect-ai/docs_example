import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
)
from datasets import load_dataset


GLOBAL_BATCH_SIZE = 128
MICRO_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE

def main():
    

    ## load model
    model_config = LlamaConfig.from_pretrained("PrimeIntellect/llama-150m-fresh")
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path="PrimeIntellect/llama-150m-fresh", config=model_config)
    model = model.to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    ## prepare data
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    ds = load_dataset("allenai/c4", "en", streaming=True)

    def tokenize_function(data):
        outputs = tokenizer(
            data["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        return outputs

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])["train"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        tokenized_datasets,
        collate_fn=data_collator,
        batch_size=MICRO_BATCH_SIZE,
        num_workers=4,
    )

    for step, batch in enumerate(iterable=train_loader):
        
        is_accumulating = step % GRAD_ACCUMULATION_STEPS == 0

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUMULATION_STEPS
        loss.backward()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"step: {step}")


if __name__ == "__main__":
    main()

