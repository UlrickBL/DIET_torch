from typing import List, Tuple

import torch
import torch.nn as nn

from config import PretrainedEmbeddingConfig


class QwenEncoder(nn.Module):
    def __init__(self, config: PretrainedEmbeddingConfig):
        super().__init__()
        self.config = config

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Qwen tokeniser from {config.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "QwenEncoder requires a fast (Rust-backed) tokeniser "
                "to obtain character offset mappings."
            )

        print(f"Loading Qwen model from {config.model_name} …")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        if config.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            base_model = get_peft_model(base_model, lora_cfg)
            base_model.print_trainable_parameters()

            if config.freeze_base:
                for name, param in base_model.named_parameters():
                    if "lora_" not in name:
                        param.requires_grad_(False)
        else:
            for param in base_model.parameters():
                param.requires_grad_(False)

        self.model = base_model
        self.hidden_size: int = base_model.config.hidden_size

    def _format_input(self, text: str) -> Tuple[str, int]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": text},
        ]
        formatted: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        offset = formatted.rfind(text)
        if offset == -1:
            offset = len(formatted) - len(text)
        return formatted, offset

    def forward(
        self,
        texts: List[str],
        word_spans: List[List[Tuple[int, int]]],
        max_words: int,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        B = len(texts)

        formatted_texts: List[str] = []
        text_offsets: List[int] = []
        for text in texts:
            fmt, off = self._format_input(text)
            formatted_texts.append(fmt)
            text_offsets.append(off)

        enc = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_token_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offset_mapping: torch.Tensor = enc["offset_mapping"]

        ctx = torch.enable_grad() if self.config.use_lora else torch.no_grad()
        with ctx:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden: torch.Tensor = outputs.hidden_states[-1]

        result = torch.zeros(B, max_words, self.hidden_size, device=device)

        for b in range(B):
            text_off = text_offsets[b]
            spans = word_spans[b]
            for w_idx, (w_start, w_end) in enumerate(spans):
                if w_idx >= max_words:
                    break
                adj_start = text_off + w_start
                adj_end = text_off + w_end

                matching_indices: List[int] = []
                for j, (q_s, q_e) in enumerate(offset_mapping[b].tolist()):
                    if q_e > adj_start and q_s < adj_end:
                        if attention_mask[b, j].item():
                            matching_indices.append(j)

                if matching_indices:
                    result[b, w_idx] = hidden[b, matching_indices].mean(dim=0)

        return result

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param
