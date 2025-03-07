import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from dataclasses import dataclass, field
from typing import Optional
import transformers
import wandb
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
import bitsandbytes as bnb
from liger_kernel.transformers import AutoLigerKernelForCausalLM

@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="HuggingFaceTB/SmolLM2-360M-Instruct")
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(default=2048)
    logging_steps: int = field(default=100)
    report_to: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine")

class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 2048
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = -100

        self.chunks, self.cum_lengths = self.load_memmap_chunks(data_path, split)
        self.length = self.cum_lengths[-1]

        self.special_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in [
            '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
            '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
            '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
            '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>'
        ]}

    def load_memmap_chunks(self, data_path, split):
        chunks = []
        cum_lengths = [0]
        pattern = os.path.join(data_path, f'{split}_rank*_partial*_input_ids.memmap')
        for memmap_file in sorted(glob.glob(pattern)):
            shape = np.load(memmap_file.replace('.memmap', '_shape.npy'))
            chunk = np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))
            chunks.append(chunk)
            cum_lengths.append(cum_lengths[-1] + shape[0])

        if not chunks:
            memmap_file = os.path.join(data_path, f'{split}_input_ids.memmap')
            shape = np.load(os.path.join(data_path, f'{split}_input_ids_shape.npy'))
            chunks = [np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))]
            cum_lengths = [0, shape[0]]

        return chunks, cum_lengths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk, local_idx = self.get_chunk_and_local_index(idx)
        input_ids = torch.tensor(chunk[local_idx], dtype=torch.long)

        labels = torch.full_like(input_ids, self.ignore_index)

        speech_start = (input_ids == self.special_tokens['<|SPEECH_GENERATION_START|>']).nonzero()[0].item()
        speech_end = (input_ids == self.special_tokens['<|SPEECH_GENERATION_END|>']).nonzero()[0].item()

        prompt = [
            {'role': 'user', 'content': 'Convert the text to speech:<|TEXT_UNDERSTANDING_START|>'},
            {'role': 'assistant', 'content': '<|SPEECH_GENERATION_START|>'}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True)

        prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|TEXT_UNDERSTANDING_START|>'], input_ids[:speech_start])
        prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|SPEECH_GENERATION_START|>'], input_ids[speech_start:speech_end + 1])

        input_ids = self.pad_to_max_length(torch.tensor(prompt_ids, dtype=torch.long), self.pad_token_id)
        labels = self.create_labels(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def replace_token(self, tokens, target, replacement):
        idx = tokens.index(target)
        return tokens[:idx] + replacement.tolist() + tokens[idx+1:]

    def create_labels(self, input_ids):
        labels = torch.full_like(input_ids, self.ignore_index)
        speech_start = (input_ids == self.special_tokens['<|SPEECH_GENERATION_START|>']).nonzero()[0].item()
        labels[speech_start:] = input_ids[speech_start:]
        labels[input_ids == self.pad_token_id] = self.ignore_index
        return labels

    def pad_to_max_length(self, sequence, pad_value):
        return torch.cat([sequence, torch.full((self.max_length - len(sequence),), pad_value, dtype=sequence.dtype)])

    def get_chunk_and_local_index(self, idx):
        for i, cum_len in enumerate(self.cum_lengths[1:], start=1):
            if idx < cum_len:
                return self.chunks[i-1], idx - self.cum_lengths[i-1]
        raise IndexError("Index out of range")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == "wandb":
        wandb.init(project="tts-training", name=training_args.run_name, config=training_args.to_dict())

    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_name_or_path)
    tokenizer.pad_token_id = 128001

    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = TTSDataset(data_args.data_path, 'train', tokenizer)
    # eval_dataset = TTSDataset(data_args.data_path, 'test', tokenizer)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "bias" not in n], "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "bias" in n], "weight_decay": 0.0}
    ]
    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=training_args.learning_rate)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None)
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
