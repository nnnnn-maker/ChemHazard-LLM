from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path

LABELS = [
    "flammable",
    "oxidizing",
    "gas_under_pressure",
    "corrosive",
    "acute_toxicity",
    "irritant_harmful",
    "cmr",
    "stot",
    "environmental_hazard",
]
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


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_generation_messages(record: dict) -> list[dict[str, str]]:
    messages = record.get("messages", [])
    generation_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        if role == "assistant":
            continue
        generation_messages.append({"role": role, "content": message.get("content", "")})
    return generation_messages


def fallback_prompt(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"{role}: {message['content']}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def normalize_binary(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "1.0", "true", "yes"} else 0


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError("No JSON object found")


def extract_labels(text: str) -> tuple[dict[str, int], str]:
    try:
        parsed = parse_json_object(text)
        labels = {label: normalize_binary(parsed.get(label, 0)) for label in LABELS}
        return labels, "ok"
    except Exception:
        recovered = {}
        for label in LABELS:
            pattern = rf'"?{re.escape(label)}"?\s*[:=]\s*(true|false|1|0)'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                recovered[label] = normalize_binary(match.group(1))
        if len(recovered) == len(LABELS):
            return {label: recovered[label] for label in LABELS}, "regex_recovered"
        raise


def should_use_trust_remote_code(args: argparse.Namespace, model_path: Path) -> bool:
    model_name = args.model_name.lower()
    path_text = str(model_path).lower()
    return args.trust_remote_code or model_name == "chatglm" or "chatglm" in path_text or "chemllm" in path_text


def resolve_model_path(args: argparse.Namespace) -> Path:
    return Path(args.model_path) if args.model_path else MODEL_PATHS[args.model_name]


def resolve_dtype_kwargs(dtype_value):
    from transformers import AutoModelForCausalLM

    signature = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    if "dtype" in signature:
        return {"dtype": dtype_value}
    if "torch_dtype" in signature:
        return {"torch_dtype": dtype_value}
    return {}


def maybe_build_quantization_config(args: argparse.Namespace):
    import torch
    from transformers import BitsAndBytesConfig

    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def resolve_model_input_device(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            device = embeddings.weight.device
            if str(device) != "meta":
                return device

    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device

    return next(model.parameters()).device


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args)
    base_model_path = Path(args.base_model_path) if args.base_model_path else model_path
    trust_remote_code = should_use_trust_remote_code(args, model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
    )

    dtype_value = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    quantization_config = maybe_build_quantization_config(args)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model_kwargs.update(resolve_dtype_kwargs(dtype_value))

    if args.adapter_path:
        try:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_path, **model_kwargs)
        except Exception:
            from peft import PeftModel

            # Fallback path for PEFT versions/environments where the auto class
            # is unavailable or adapter auto-loading fails. Avoid device_map=auto
            # during adapter attachment because some PEFT/Accelerate versions
            # crash while re-balancing memory for already-dispatched base models.
            base_model_kwargs = dict(model_kwargs)
            base_model_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(base_model_path, **base_model_kwargs)
            model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if not args.device_map:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model.eval()
    return tokenizer, model, model_path, trust_remote_code


def generate_one(record: dict, tokenizer, model, args: argparse.Namespace) -> dict:
    import torch

    messages = extract_generation_messages(record)
    # The historical Mistral tokenizer had no chat_template. Reproduce its
    # fallback prompt explicitly when using the pinned official Instruct weights.
    force_plain_prompt = args.model_name in {"chemllm", "mistral"}
    input_device = resolve_model_input_device(model)

    if (not force_plain_prompt) and hasattr(tokenizer, "apply_chat_template"):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            prompt = fallback_prompt(messages)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
    else:
        prompt = fallback_prompt(messages)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )

    if hasattr(inputs, "items"):
        model_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }
    else:
        model_inputs = {"input_ids": inputs.to(input_device)}

    input_length = model_inputs["input_ids"].shape[-1]
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)

    generated_ids = output_ids[0][input_length:]
    prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    labels = None
    parse_status = "ok"
    try:
        labels, parse_status = extract_labels(prediction_text)
    except Exception as exc:
        parse_status = f"parse_failed: {type(exc).__name__}"

    result = {
        "id": record.get("id", ""),
        "prediction_text": prediction_text,
        "parse_status": parse_status,
    }
    if labels is not None:
        result["labels"] = labels
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference for PubChem coarse hazard SFT/LoRA models.")
    project_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        default=str(project_dir / "sft_coarse_jsonl" / "test.jsonl"),
        help="Input JSONL containing messages and labels.",
    )
    parser.add_argument(
        "--output",
        default=str(project_dir / "predictions" / "coarse_test_predictions.jsonl"),
        help="Output prediction JSONL.",
    )
    parser.add_argument(
        "--model-name",
        default="qwen",
        choices=sorted(MODEL_PATHS.keys()),
        help="Preset local model name under $CHEMHAZARD_MODEL_ROOT or this package's Model directory.",
    )
    parser.add_argument("--model-path", default="", help="Optional explicit model path. Overrides --model-name.")
    parser.add_argument("--base-model-path", default="", help="Base model path when loading an adapter separately.")
    parser.add_argument("--adapter-path", default="", help="Optional PEFT adapter path.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default="auto", help="Use auto for Accelerate placement, or pass an empty string to force single-device loading.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose either --load-in-4bit or --load-in-8bit, not both.")
    if args.base_model_path and not args.adapter_path:
        raise SystemExit("If --base-model-path is provided, please also provide --adapter-path.")

    records = read_jsonl(Path(args.input))
    if args.limit > 0:
        records = records[: args.limit]

    tokenizer, model, model_path, trust_remote_code = load_model_and_tokenizer(args)
    print(f"Using model_name={args.model_name}")
    print(f"Resolved model_path={model_path}")
    print(f"trust_remote_code={trust_remote_code}")
    if args.adapter_path:
        print(f"adapter_path={args.adapter_path}")

    outputs: list[dict] = []
    for index, record in enumerate(records, start=1):
        result = generate_one(record, tokenizer, model, args)
        outputs.append(result)
        print(f"[{index}/{len(records)}] {record.get('id', '')} -> {result['parse_status']}")

    write_jsonl(Path(args.output), outputs)
    print(f"Predictions written to: {args.output}")


if __name__ == "__main__":
    main()
