# IEMOCAP and MELD video-text data preparation

This folder prepares video + text data for evaluating Qwen3-VL-8B style VLMs.
Audio files are not kept in the processed manifests and are removed from MELD
raw extraction when possible.

Server project root used by the scripts:

```bash
/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design
```

## Output layout

By default all data is stored under:

```bash
/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data
```

Important files after preparation:

```text
data/
  meld/
    raw/
    processed/
      train.jsonl
      dev.jsonl
      test.jsonl
      labels.json
      dataset_summary.json
  iemocap/
    raw/
    processed/
      train.jsonl
      dev.jsonl
      test.jsonl
      labels.json
      dataset_summary.json
```

Each JSONL row is one utterance-level sample:

```json
{
  "dataset": "meld",
  "split": "train",
  "sample_id": "meld_train_dia0_utt0",
  "dialogue_id": "0",
  "utterance_id": "0",
  "speaker": "Speaker",
  "text": "Utterance text",
  "emotion": "neutral",
  "sentiment": "neutral",
  "video_path": "/absolute/path/to/video.mp4",
  "context": [
    {"speaker": "A", "text": "previous utterance", "emotion": "joy"}
  ],
  "qwen_prompt": "..."
}
```

## MELD

MELD is publicly downloadable from the official project page. Run:

```bash
bash datasets_video_text/scripts/download_meld.sh
```

The script downloads `MELD.Raw.tar.gz`, extracts it, removes audio-only files,
and creates the processed JSONL files. The official raw archive contains inner
split archives such as `train.tar.gz`, `dev.tar.gz`, and `test.tar.gz`; the
script extracts these automatically. If any of `train_sent_emo.csv`,
`dev_sent_emo.csv`, or `test_sent_emo.csv` is missing from the raw archive, the
script downloads the missing annotation CSV from the official
`declare-lab/MELD` GitHub repository.

Official source checked when this folder was created:
https://affective-meld.github.io/

## IEMOCAP

IEMOCAP requires a license from USC SAIL and should be downloaded through the
official channel. After obtaining the official package, place the downloaded
archive(s) on the server, for example:

```bash
mkdir -p /XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/archives
cp /path/to/IEMOCAP_*.zip /XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/archives/
```

Then run:

```bash
IEMOCAP_ARCHIVE_DIR=/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/archives \
bash datasets_video_text/scripts/download_iemocap.sh
```

The script extracts only useful text/annotation/video material when possible
and creates JSONL manifests. It also supports `IEMOCAP_URLS_FILE` for private,
authorized URLs, but no public mirror is embedded because the dataset is
license-restricted.

The initial IEMOCAP manifest may point to dialogue-level videos such as
`Ses01F_impro01.avi`. For utterance-level VLM evaluation, cut those videos into
sentence clips after `download_iemocap.sh` finishes:

```bash
bash datasets_video_text/scripts/cut_iemocap_sentence_videos.sh
```

This uses `ffmpeg` and writes:

```text
data/iemocap/sentence_videos/
data/iemocap/processed_sentence/train.jsonl
data/iemocap/processed_sentence/dev.jsonl
data/iemocap/processed_sentence/test.jsonl
```

The updated JSONL files set `video_path` to the sentence-level `.mp4` clip and
preserve the original dialogue video as `source_video_path`. Audio is stripped
by default because the target evaluation is video + text. Add `--keep-audio` if
you need audio retained:

```bash
bash datasets_video_text/scripts/cut_iemocap_sentence_videos.sh --keep-audio
```

## Manifest validation

After syncing generated manifests, run:

```bash
python3 datasets_video_text/scripts/validate_manifests.py --check-exists
```

This checks row counts, missing `video_path`, missing prompt/text, major path
patterns, and optionally whether referenced videos exist on the server.

## Qwen3-VL-8B-Instruct evaluation

The evaluation scripts use the generated `qwen_prompt` text and the video clip
from `video_path`. Video inputs are passed to the processor with `fps=2` by
default. The launcher runs one process per GPU, and each process loads one model
copy and handles one shard of the test set.

Default model path:

```bash
/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/qwen3vl_8b
```

Run both test sets on four GPUs:

```bash
bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

By default the launcher sources:

```bash
/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/cache_env_new.sh
```

Override or disable this with:

```bash
ENV_FILE=/path/to/your_env.sh bash datasets_video_text/scripts/run_qwen3vl_eval.sh
ENV_FILE= bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

The launcher traps `Ctrl+C`/`SIGTERM` and stops all worker processes it started.
For an already running old launcher, stop worker PIDs manually with `kill -TERM`.

Useful overrides:

```bash
DATASET=meld bash datasets_video_text/scripts/run_qwen3vl_eval.sh
DATASET=iemocap bash datasets_video_text/scripts/run_qwen3vl_eval.sh
GPUS=0,1,2,3 FPS=2 FLASH_ATTN=1 bash datasets_video_text/scripts/run_qwen3vl_eval.sh
FLASH_ATTN=0 bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

Run text-only ablations with the same model and prompts:

```bash
MODALITY=text_only DATASET=iemocap bash datasets_video_text/scripts/run_qwen3vl_eval.sh
MODALITY=text_only DATASET=meld bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

Build manifests using the prompt style in `data/new_prompt`:

```bash
bash datasets_video_text/scripts/build_new_prompt_manifests.sh
```

This writes:

```text
data/meld/processed_new_prompt/
data/iemocap/processed_sentence_new_prompt/
```

The new prompt manifests use:

```text
prompt field: qwen_prompt_new
gold field: emotion_prompt
speaker names: Speaker_0, Speaker_1, ...
```

IEMOCAP's provided new prompt style uses six labels:

```text
happy, sad, neutral, angry, excited, frustrated
```

Therefore samples outside that label space, such as `fear` and `surprise`, are
dropped by default when building `processed_sentence_new_prompt`.

Run with the new prompt style:

```bash
PROMPT_VERSION=new DATASET=iemocap bash datasets_video_text/scripts/run_qwen3vl_eval.sh
PROMPT_VERSION=new DATASET=meld bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

Run text-only with the new prompt style:

```bash
MODALITY=text_only PROMPT_VERSION=new DATASET=iemocap bash datasets_video_text/scripts/run_qwen3vl_eval.sh
MODALITY=text_only PROMPT_VERSION=new DATASET=meld bash datasets_video_text/scripts/run_qwen3vl_eval.sh
```

Outputs are written to:

```text
datasets_video_text/results/qwen3vl_8b/{MODALITY}_{PROMPT_VERSION}/
  meld_test_shard0.jsonl
  ...
  meld_test_metrics.json
  iemocap_test_metrics.json
  logs/
```

Metrics include accuracy and weighted F1 (`weighted_f1`). Prediction JSONL files
store the raw model output and the parsed label for inspection.

## Train/dev/test split policy

MELD uses its official train/dev/test split.

IEMOCAP has no single universal official split. This preparation uses a common
speaker/session-independent evaluation split:

- `test`: Session5
- `dev`: Session4
- `train`: Sessions1-3

Override this in `prepare_iemocap.py` if your experiment protocol requires
5-fold leave-one-session-out cross validation.
