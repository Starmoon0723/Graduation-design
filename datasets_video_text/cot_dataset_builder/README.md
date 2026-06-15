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

The vLLM config uses `--reasoning-parser qwen3`,
`--media-io-kwargs '{"video":{"num_frames":-1}}'`, and request-time
`mm_processor_kwargs` with `fps=2` and `do_sample_frames=true`. The dataset
generator only reads normal `message.content` as the final answer text.
`reasoning_content` is treated as internal reasoning and is never used as
`VISUAL_REASON` or `DIALOGUE_REASON`; the client records only whether it exists
inside `response_meta` for debugging empty responses.

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

This step sends the current utterance video plus a small target-speaker locator
prompt to Qwen3.6-27B. The text is only for locating the speaker/current
speaking moment; the teacher must not use utterance meaning as emotional
evidence and must not predict a final label. The output field is only
`VISUAL_REASON`, written as one concrete visual-evidence paragraph.

```bash
STEP=visual DATASET=iemocap SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Output:

```text
datasets_video_text/cot_dataset_builder/results/qwen36_27b/step1_visual_reason/
```

## 2. Dialogue-Only Reasoning Data

This step sends only dialogue text. The manifest gold label is provided to the
teacher so it can write label-grounded textual evidence, but the output should
not say that the gold label was provided and must not mention visual/audio cues.
The output field is only `DIALOGUE_REASON`.

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

The generator uses step-specific token limits from the YAML config:

```text
visual:   max_tokens_visual
dialogue: max_tokens_dialogue
fusion:   max_tokens_fusion
predict:  max_tokens_predict
```

If a previous run produced rows with an empty reason, truncated analysis, or
`status=ok_unparsed`, `RESUME=1` will not treat those rows as finished. Rerunning
the same command will retry them and append a later corrected row. Later stages
read only `status=ok` reason rows, so old malformed `Thinking Process` outputs
are ignored.

## 3. Multimodal Fusion Reasoning

This step does not send video. It reads the original manifest row, the
`VISUAL_REASON` from step 1, the `DIALOGUE_REASON` from step 2, the candidate
labels, and the manifest gold label. Qwen3.6-27B then writes the final
gold-aligned multimodal reasoning response:

```text
<think>
...
</think>
<answer>
gold
</answer>
```

Run fusion after visual and dialogue reasoning are available:

```bash
STEP=fusion DATASET=all SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Or run all three reasoning stages in sequence:

```bash
STEP=all_reasoning DATASET=all SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

Output:

```text
datasets_video_text/cot_dataset_builder/results/qwen36_27b/step3_fusion_reason/
```

Optional teacher classification diagnostics can be run separately:

```bash
STEP=predict DATASET=iemocap SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_reasoning_generation.sh
```

The diagnostic output is not used as the sole SFT filter.

## 4. Build SFT, RL, and Preference Files

After steps 1-3 finish:

```bash
DATASET=all SPLIT=train \
bash datasets_video_text/cot_dataset_builder/scripts/run_build_sft_rl.sh
```

Output:

```text
datasets_video_text/cot_dataset_builder/results/qwen36_27b/final_sft/{dataset}/{split}.jsonl
datasets_video_text/cot_dataset_builder/results/qwen36_27b/final_rl/{dataset}/{split}.jsonl
datasets_video_text/cot_dataset_builder/results/qwen36_27b/final_preferences/{dataset}/{split}.jsonl
```

Step 3 reads both `VISUAL_REASON` and `DIALOGUE_REASON`, applies basic quality
checks, and uses the step 3 fusion response directly as the SFT assistant
response. It no longer rule-concatenates visual and dialogue reasons.

SFT no longer depends on teacher prediction correctness. RL files are GRPO-ready
prompts with reference visual/dialogue/fusion reasons and no precomputed 0/1
reward. Preference data is not auto-created without an external rejected
response source, so `final_preferences` is empty by default.
