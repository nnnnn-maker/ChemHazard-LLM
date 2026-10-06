from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100
MODEL_ROOT = Path(
    os.environ.get("CHEMHAZARD_MODEL_ROOT", Path(__file__).resolve().parent / "Model")
)
MODEL_PATHS = {
    "qwen": Path(os.environ["CHEMHAZARD_QWEN_MODEL_PATH"])
    if os.environ.get("CHEMHAZARD_QWEN_MODEL_PATH")
    else MODEL_ROOT / "Qwen2.5-7B-Instruct",
    "mistral": Path(os.environ["CHEMHAZARD_MISTRAL_MODEL_PATH"])
    if os.environ.get("CHEMHAZARD_MISTRAL_MODEL_PATH")
    else MODEL_ROOT / "Mistral-7B-Instruct-v0.1",
    "chatglm": MODEL_ROOT / "chatglm3-6b",
    "gemma": MODEL_ROOT / "gemma-2-9b-it",
    "chemllm": MODEL_ROOT / "ChemLLM-7B-Chat",
}
DEFAULT_TARGET_MODULES = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "chemllm": ["wqkv", "wo", "w1", "w2", "w3"],
    "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class SupervisedChatDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int, force_plain_prompt: bool = False) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.force_plain_prompt = force_plain_prompt

    def __len__(self) -> int:
        return len(self.records)

    def _render_prompt_text(self, messages: list[dict[str, str]]) -> str:
        if not self.force_plain_prompt and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"{message['role'].upper()}: {message['content']}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def _render_full_text(self, messages: list[dict[str, str]]) -> str:
        if not self.force_plain_prompt and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"{message['role'].upper()}: {message['content']}")
        return "\n\n".join(chunks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        messages = record["messages"]
        prompt_messages = messages[:-1]
        prompt_text = self._render_prompt_text(prompt_messages)
        full_text = self._render_full_text(messages)

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = [IGNORE_INDEX] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def data_collator(features: list[dict[str, list[int]]], tokenizer) -> dict[str, torch.Tensor]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = []
    attention_masks = []
    labels = []

    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
        attention_masks.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def should_use_trust_remote_code(args: argparse.Namespace, model_path: Path) -> bool:
    model_name = args.model_name.lower()
    path_text = str(model_path).lower()
    return args.trust_remote_code or model_name == "chatglm" or "chatglm" in path_text or "chemllm" in path_text


def ensure_tokenizer_padding(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "ddp_find_unused_parameters": False if torch.cuda.device_count() > 1 else None,
        "optim": args.optim,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "data_seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = args.eval_strategy
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = args.eval_strategy
    if "lr_scheduler_type" in signature:
        kwargs["lr_scheduler_type"] = args.lr_scheduler_type

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature and value is not None}
    return TrainingArguments(**filtered_kwargs)


def default_output_dir(project_dir: Path, model_name: str) -> Path:
    return project_dir / "checkpoints_lora_sft" / f"{model_name}_coarse_lora"


def resolve_target_modules(args: argparse.Namespace) -> list[str]:
    if args.target_modules:
        return [item.strip() for item in args.target_modules.split(",") if item.strip()]
    return DEFAULT_TARGET_MODULES[args.model_name]


def maybe_build_quantization_config(args: argparse.Namespace):
    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def print_trainable_parameters(model) -> None:
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = 100 * trainable_params / all_params if all_params else 0.0
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {ratio:.4f}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LoRA SFT for PubChem coarse hazard classification.")
    parser.add_argument("--model-name", default="qwen", choices=sorted(MODEL_PATHS.keys()))
    parser.add_argument("--model-path", default="", help="Optional explicit local model directory. Overrides --model-name.")
    parser.add_argument("--train-file", default=str(project_dir / "sft_coarse_jsonl" / "train.jsonl"))
    parser.add_argument("--val-file", default=str(project_dir / "sft_coarse_jsonl" / "val.jsonl"))
    parser.add_argument("--output-dir", default="", help="Defaults to checkpoints_lora_sft/<model-name>_coarse_lora.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="", help="Comma-separated target modules. Empty uses model defaults.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose either --load-in-4bit or --load-in-8bit, not both.")

    set_seed(args.seed)

    if not args.bf16 and not args.fp16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.bf16 = True
        elif torch.cuda.is_available():
            args.fp16 = True

    model_path = Path(args.model_path) if args.model_path else MODEL_PATHS[args.model_name]
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(project_dir, args.model_name)
    trust_remote_code = should_use_trust_remote_code(args, model_path)
    target_modules = resolve_target_modules(args)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    quantization_config = maybe_build_quantization_config(args)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    ensure_tokenizer_padding(tokenizer, model)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    # The validated Mistral experiment tokenizer_config.json had no chat_template.
    # Preserve the original SYSTEM/USER/ASSISTANT fallback even when the pinned
    # official Instruct snapshot supplies a template in a newer tokenizer file.
    force_plain_prompt = args.model_name == "mistral"
    train_dataset = SupervisedChatDataset(train_records, tokenizer, args.max_length, force_plain_prompt)
    val_dataset = SupervisedChatDataset(val_records, tokenizer, args.max_length, force_plain_prompt)

    training_args = build_training_args(args, output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda features: data_collator(features, tokenizer),
    )

    print(f"Using model_name={args.model_name}")
    print(f"Resolved model_path={model_path}")
    print(f"Output directory={output_dir}")
    print(f"trust_remote_code={trust_remote_code}")
    print(f"target_modules={target_modules}")
    print(f"load_in_4bit={args.load_in_4bit}, load_in_8bit={args.load_in_8bit}")

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model saved to: {output_dir}")
    print(f"Eval metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
