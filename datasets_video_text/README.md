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

## Train/dev/test split policy

MELD uses its official train/dev/test split.

IEMOCAP has no single universal official split. This preparation uses a common
speaker/session-independent evaluation split:

- `test`: Session5
- `dev`: Session4
- `train`: Sessions1-3

Override this in `prepare_iemocap.py` if your experiment protocol requires
5-fold leave-one-session-out cross validation.
