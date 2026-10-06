# Model requirements

Model weights are not redistributed in this repository. Download each checkpoint from its official repository after reviewing its license, place it at the expected directory, and use the revision recorded in `MODEL_MANIFEST.json`. A model name or a mutable `main`/`latest` reference is not sufficient for exact reproduction.

On the validated server, the model root is `pubchem_work/Model`, one directory above this package. Set `CHEMHAZARD_MODEL_ROOT=../Model` before running the training and inference scripts from `ChemHazard_LLM`.

| CLI | Official repository | Expected directory | Architecture | Access and loading notes |
|---|---|---|---|---|
| `qwen` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `Model/Qwen2.5-7B-Instruct` | `Qwen2ForCausalLM` | Apache-2.0; remote code normally not required |
| `mistral` | [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/tree/ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca) | `Model/Mistral-7B-Instruct-v0.1` | `MistralForCausalLM` | Apache-2.0; remote code not required |
| `gemma` | [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | `Model/gemma-2-9b-it` | `Gemma2ForCausalLM` | Gemma terms; access requires accepting the provider's conditions |
| `chatglm` | [`zai-org/chatglm3-6b`](https://huggingface.co/zai-org/chatglm3-6b) | `Model/chatglm3-6b` | `ChatGLMModel` | Experiment used the historical ID `THUDM/chatglm3-6b`, which now redirects; model license applies; `trust_remote_code=True` |
| `chemllm` | [`AI4Chem/ChemLLM-7B-Chat`](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat) | `Model/ChemLLM-7B-Chat` | `InternLM2ForCausalLM` | Review both repository license metadata and model-card weight terms; `trust_remote_code=True` |

Runtime mapping is fixed separately from model identity: Qwen, Mistral, and Gemma use the `chem2` environment; ChatGLM and ChemLLM use the `chem` environment. The authoritative mapping is `RUNTIME_ENVIRONMENTS.json`. Capture both environments independently with `capture_environment.py --name chem2` and `capture_environment.py --name chem`.

The machine-readable specification is `MODEL_MANIFEST.json`. The recorded revisions must be checked against the experiment-used files; do not substitute a repository's current HEAD. The Mistral weight shards on the validated server match the local official Instruct revision `ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`. The recovered server `tokenizer_config.json` has SHA-256 `ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99`. Its only JSON-content difference from the official Instruct snapshot is that it has no `chat_template`. Therefore, this package explicitly uses the historical plain `SYSTEM`/`USER`/`ASSISTANT` prompt for Mistral training and inference. The downloaded audit file is ignored by Git; the prompt behavior is reproduced in code without redistributing it. The official model directory is not bit-for-bit identical to the historical server directory.

On the validated server, create the auditable local manifest with:

```bash
python capture_model_manifest.py \
  --spec MODEL_MANIFEST.json \
  --root .. \
  --output reproducibility/model_manifest.local.json \
  --public-output reproducibility/model_manifest.public.json \
  --hash-weights
```

The resulting file records model type, architecture, exact revision when resolvable, identity-file checksums, weight-file checksums, and local byte counts. If the models were downloaded as ordinary directories rather than Hugging Face snapshot paths, fill `used_revision` manually from the download record or cache metadata and rerun the command.

If verified experiment-used files are in historical alias directories, add `--model-dir qwen=../Model/Qwen-7B-Chat` and `--model-dir mistral=../Model/Mistral-7B-v0.1` to the capture command. These override only audit paths. For training and inference with the historical directories, set `CHEMHAZARD_QWEN_MODEL_PATH=../Model/Qwen-7B-Chat` and `CHEMHAZARD_MISTRAL_MODEL_PATH=../Model/Mistral-7B-v0.1` after checking their hashes; the published defaults point to correctly named directories.

The historical local directory `Qwen-7B-Chat` contained a Qwen2.5 configuration, but that alias is deliberately not used as the public model name. The manuscript and reproduction code report `Qwen2.5-7B-Instruct`. Likewise, the historical server directory `Mistral-7B-v0.1` is only a misleading alias: its recorded weight hashes match `Mistral-7B-Instruct-v0.1`, not evidence that the base-model repository was used.

The `transformers_version` field inside `config.json` is checkpoint metadata, not proof of the runtime software version. Record runtime packages and GPU details separately for both named environments with `capture_environment.py`.

`experiment_access_date` in `MODEL_MANIFEST.json` is the China-standard-time date of the latest revision-matching Hugging Face local download metadata record retained with each model. The UTC timestamp, metadata filename, and matching-record count are recorded alongside it and copied into the sanitized public manifest. Hugging Face writes this timestamp on download or on successful local-cache validation, so it is evidence of access/validation, not proof of the first download. These dates are distinct from the date on which the model repository webpage is consulted for a manuscript citation.

Before Mistral training or inference, `verify_reproduction_setup.py --verify-mistral` hashes the first safetensors shard and rejects weights that do not match the validated Instruct checkpoint. It also verifies that tokenizer settings match the server after removing the official file's extra `chat_template`; if the recovered audit file is locally present, its raw SHA-256 is checked as well. The main, screening, NITE, and component-ablation shell entry points run this check automatically. The informational warning about the official chat template does not mean the full directory is byte-identical.

## LoRA adapters for exact result reproduction

The five backbone weight sets should remain at their official providers and should not be committed to Git. For exact reproduction of the reported ChemHazard-LLM result, archive the three final Mistral LoRA adapters for training seeds 123, 777, and 2025 in a versioned release such as Zenodo or Hugging Face, subject to the applicable licenses. Each adapter release should include `adapter_config.json`, adapter weights, trainer state or training metadata, SHA-256 checksums, the associated checkpoint revision and tokenizer configuration, and the validation/test prediction checksums.

If the adapters are not released, the repository still supports full retraining, but independent runs may not reproduce the reported metrics bit-for-bit because of hardware and nondeterministic training effects.
