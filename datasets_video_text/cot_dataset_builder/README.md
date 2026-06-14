# Qwen3.6-27B CoT Dataset Builder

This experiment area builds teacher reasoning data for emotion-recognition SFT
and RL experiments from the existing MELD/IEMOCAP video-text manifests.

The default config is:

```bash
datasets_video_text/cot_dataset_builder/configs/qwen36_27b_dataset.yaml
```

It serves two vLLM instances:

```text
GPU 0,1 -> http://127.0.0.1:18000/v1
GPU 2,3 -> http://127.0.0.1:18001/v1
```

The model path is set to:

```bash
/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/qwen3.6_27b
```

The vLLM config follows the Qwen3.6-27B model card style: reasoning is enabled
with `--enable-reasoning --reasoning-parser qwen3`,
`--media-io-kwargs '{"video":{"num_frames":-1}}'`, and request-time
`mm_processor_kwargs` with `fps=2` and `do_sample_frames=true`.

## 0. Start or Stop vLLM

```bash
bash datasets_video_text/cot_dataset_builder/scripts/serve_qwen36_vllm.sh start
bash datasets_video_text/cot_dataset_builder/scripts/serve_qwen36_vllm.sh status
bash datasets_video_text/cot_dataset_builder/scripts/serve_qwen36_vllm.sh stop
```

Logs and pids are written under:

```text
datasets_video_text/cot_dataset_builder/logs/qwen36_27b/
```

If 2-GPU serving OOMs, lower `vllm.max_model_len` in the YAML first. The
current default is `131072`, which is more realistic than the full 262K context
for two A800 80GB cards per instance.

## 1. Visual Reasoning Data

This step sends video + text to Qwen3.6-27B. The teacher output includes
`speaker_visible` and `visual_confidence` directly inside `VISUAL_REASON`.

```bash
STEP=visual DATASET=iemocap SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Output:

```text
datasets_video_text/cot_dataset_builder/results/qwen36_27b/step1_visual_reason/
```

## 2. Dialogue-Only Reasoning Data

This step sends only the dialogue prompt. It is useful for text-only SFT/RL data
and for comparing whether video adds reliable signal.

```bash
STEP=dialogue DATASET=iemocap SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Run both step 1 and step 2:

```bash
STEP=both DATASET=all SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Use `LIMIT=100` for a quick smoke run.

## 3. Build SFT and RL Files

After teacher generation finishes:

```bash
STEP=both DATASET=all SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_build_sft_rl.sh
```

Output:

```text
datasets_video_text/cot_dataset_builder/results/qwen36_27b/step3_sft_rl/{dataset}/{split}/
  visual_sft.jsonl
  visual_rl_rewards.jsonl
  visual_preferences.jsonl
  dialogue_sft.jsonl
  dialogue_rl_rewards.jsonl
  dialogue_preferences.jsonl
```

By default SFT keeps only teacher rows whose parsed final label matches the
manifest label. RL reward files keep all parsed teacher rows and assign reward
`1.0` for exact label match, otherwise `0.0`.
