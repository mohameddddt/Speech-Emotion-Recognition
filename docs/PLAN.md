# Project assessment and roadmap

*Assessment date: 2026-10-23, at commit `ccb001e` plus the notebook move into `notebooks/`.*

## 1. Where the project stands

### What exists

| area | status |
|---|---|
| Data | CREMA-D, 7,442 clips, 91 actors, 6 emotions, committed under `data/AudioWAV/` (~578 MB) |
| Hand-crafted pipeline | 422-dim feature vector (MFCC + deltas, log-mel, chroma, spectral shape, energy, pitch). 5 classifiers compared, soft-voting ensemble saved (`ser/features.py`, `ser/model.py`, `train.py`) |
| Pretrained-embedding pipeline | Frozen WavLM / HuBERT / wav2vec2 encoder. Per-layer mean-pooled embeddings cached and resumable. Layer window and `C` chosen by actor-grouped CV, then a logistic-regression probe is trained (`ser/embeddings.py`, `train_embeddings.py`) |
| Evaluation | Speaker-independent: 19 of 91 actors held out (`GroupShuffleSplit`), model selection with `GroupKFold` on training actors only |
| Inference | `predict.py` CLI. Works with either model and takes file paths only |
| Notebooks | `notebooks/project_notebook.ipynb` (local, with outputs), `notebooks/colab_notebook.ipynb` (Colab copy of the same walkthrough), `notebooks/embeddings_colab.ipynb` (GPU embedding experiment) |
| Demo / website | none |
| Tests / CI / packaging | none |

### Results so far (same held-out actors, 1,552 test clips)

| model | accuracy | macro F1 |
|---|---|---|
| Soft-voting ensemble on hand-crafted features | 0.532 | 0.531 |
| **WavLM-base-plus, layers 5–9 averaged, logistic regression (C=0.01)** | **0.742** | **0.742** |

Per-class recall for the WavLM probe: angry 0.89, neutral 0.88, fear 0.75, happy 0.69, sad 0.66, disgust 0.60.
The largest confusions are disgust→sad (41 clips), happy→fear (37), sad→fear (37), disgust→angry (28) and sad→disgust (26).

Per-layer CV accuracy peaks at layer 7 (0.712). Single layers 5–11 are all above 0.69, and averaging layers 5–9 gives 0.742.

Measured on the development laptop (CPU only): the first WavLM prediction takes ~84 s (importing torch/transformers plus loading weights). After that, one 2.5 s clip takes **~0.4 s**. A free CPU host is fast enough, as long as the model is loaded once at startup.

## 2. Strengths

1. **The evaluation is sound.** Most public CREMA-D projects split randomly by clip, so the same voices appear in train and test, which inflates accuracy. Here the test actors are never seen, hyperparameters are chosen with grouped CV on training actors only, and the test set is used once. The reported numbers mean something.
2. **A large, cheaply obtained gain.** Moving from hand-crafted features to a frozen WavLM encoder raised accuracy from 53% to 74%, using only a linear classifier. The layer scan shows *where* emotion information lives (the middle layers), so the result is explained as well as measured.
3. **Clean code layout.** A small `ser/` package holds the logic, scripts are thin CLIs, and notebooks import the package instead of duplicating code. The design choices are documented in docstrings.
4. **Practical engineering.** Feature and embedding caches, chunked and resumable extraction (it survives a Colab disconnect), parallel CV, and a 56 KB deployable probe that stores the encoder by name rather than its weights.
5. **Real error analysis.** Breakdowns by class, arousal level, intensity and actor, not just a single accuracy number.
6. **Honest README.** It states chance level, the human baseline and the per-class numbers.

## 3. Weaknesses and risks

Ordered by how much they matter for the next goal, a public demo website.

| # | weakness | why it matters |
|---|---|---|
| W1 | **Domain gap.** CREMA-D is acted, studio-recorded, English, 12 fixed sentences, ~2.5 s clips | Website users record with laptop or phone mics, in rooms with noise, often longer or spontaneous speech. Real-world accuracy will be noticeably lower than 74%, and nothing currently measures it. |
| W2 | **No demo and no in-memory inference API** | `predict_file` only accepts paths, the encoder is created lazily on the first call, and there is no audio-length cap. A browser recording of several minutes would be embedded in one pass and could run out of memory. |
| W3 | **One fixed 80/20 split (19 test actors)** | The binomial ±2.2 pp interval is optimistic, because clips from the same actor are correlated. Differences of 1–2 pp between models (all of the hand-crafted ones) are within noise. There is no mean ± std across actor folds. |
| W4 | **Only one encoder actually evaluated** | `hubert-base`, `wav2vec2-base` and `wavlm-large` are wired up but were never run. No fine-tuning, no augmentation, and pooling is mean-only. |
| W5 | **578 MB of WAV files in git** | Every clone, including the Colab notebooks', downloads the full dataset. The same data is on Kaggle (`ejlok1/cremad`, identical `AudioWAV/` layout). |
| W6 | **Uncalibrated probabilities** | The website will show confidence bars. Logistic regression is roughly calibrated in-domain, but out of domain it will look confidently wrong. |
| W7 | **No tests, no CI, loose dependencies** | `requirements.txt` has only lower bounds, and `torch` and `transformers` are missing from it. Nothing protects the feature length (422) or the label order from regressions. |
| W8 | **Notebook duplication and hard-coded numbers** | `colab_notebook.ipynb` is a near-copy of `project_notebook.ipynb`. `BASELINE_ACCURACY = 0.532` is hard-coded, and README numbers are maintained by hand, so they will drift. |
| W9 | **Unused CREMA-D metadata** | Demographics (`VideoDemographics.csv`: sex, age, race) and the crowd-sourced perceptual labels (`tabulatedVotes.csv`) are unused. There is no fairness breakdown, and the label noise behind the ~41% human agreement is ignored. |
| W10 | **Pickle-based model files** | `joblib` pickles are tied to the `ser.embeddings` module path and are unsafe to load from untrusted sources. That is acceptable for our own artifacts, but the probe weights could be stored as plain arrays. |
| W11 | Minor | UTF-8 BOM at the start of `ser/data.py` and `ser/features.py`. `DATA_DIR` and `artifacts/` are hard-coded relative to the repo, which blocks running on Kaggle with `/kaggle/input/...`. A stray 21 MB `artifacts/colab.joblib`. |

## 4. Roadmap

Five phases. Phases 1–2 are the shortest path to a working website. Phases 3–4 make the model behind it better and the numbers more trustworthy. **All training and embedding extraction runs on Kaggle** (see §5). Local runs are limited to one-clip smoke tests.

### Phase 0: housekeeping (done or in progress)

- [x] Move notebooks into `notebooks/`, fix the Colab badge URLs, and add a `sys.path` bootstrap to the local notebook.
- [x] Make paths configurable: `SER_DATA_DIR`, `SER_ARTIFACTS_DIR` and `SER_EMBED_DIR` environment variables, defaulting to today's paths.
- [x] Remove the BOMs. (`artifacts/colab.joblib` is a git-ignored local file; delete it by hand.)
- [ ] Split dependencies (`app/requirements.txt` exists for the Space): `requirements.txt` (core), `requirements-embed.txt` (`torch`, `transformers`), `requirements-app.txt` (`gradio`). Add upper-bound pins on the versions known to work.

### Phase 1: inference layer for the website

> **Status (2026-10-23):** done. `ser/inference.py`, `ser/probe.py`, and temperature scaling in `kaggle/evaluate.py`. The probes ship inside the Space (`app/models/`, ~30 KB each) instead of a separate Hub model repo.

Goal: one object that loads once and classifies a NumPy array in under a second.

1. **`ser/inference.py`** with a `Predictor` class:
   - `Predictor.load(path_or_hub_id)`: loads the probe plus encoder once and warms it up with 1 s of silence.
   - `predict(y: np.ndarray, sr: int) -> {"label", "scores", "duration"}`: resample to 16 kHz mono, then apply the same trim and peak-normalisation as `load_audio`. This means factoring `load_audio` into `load_file` and `preprocess(y, sr)` so training and serving share one code path.
   - `predict_timeline(y, sr, win=3.0, hop=1.5)`: sliding windows over long audio, returning per-window scores and an overall label (mean of window probabilities). Use energy-based VAD (`librosa.effects.split`) to skip silent windows.
   - Hard cap of 60 s of input, so memory stays bounded.
2. **Portable probe format.** Save the scaler mean/scale, the logistic-regression `coef_`/`intercept_`, the classes, the layers and the encoder id as `probe.npz` + `probe.json`, keeping `.joblib` for backwards compatibility. This removes pickle from the serving path and makes a future in-browser version possible.
3. **Temperature scaling.** Fit one temperature on the training actors' CV predictions and store it in `probe.json`.
4. **Publish the model** to a Hugging Face Hub model repo (`mohameddddt/ser-wavlm-probe`) with a model card covering data, evaluation protocol, per-class results, known limits (acted English speech) and licence (CREMA-D is ODbL, so attribution is required).
5. Tests: `predict` on a synthetic 1 s tone returns six scores that sum to 1, and a stereo 44.1 kHz input is handled.

**Done when** `python -c "from ser.inference import Predictor; ..."` classifies a mic-style WAV in under 1 s warm on CPU, and the model downloads from the Hub.

### Phase 2: the website

> **Status (2026-10-23):** built and tested locally as a designed showcase page, replacing the first Gradio prototype: FastAPI `app/server.py` + `app/web/`. It has the live demo (mic/upload/samples, emotion-coloured waveform, timeline), data-driven charts and light/dark/mobile layouts. `app/deploy.py` builds a self-contained Docker Space. Publishing waits on `hf auth login`.

**Recommendation: a Gradio app hosted on a free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM).** It needs no frontend build, gives microphone recording and upload out of the box, and automatically exposes an HTTP/JS API that a custom frontend can call later. `gradio` 6.x is already installed locally. The measured 0.4 s per clip fits the free tier. The cold-start cost is paid once when the Space boots, not on each request.

Layout (`app/app.py`, deployed as the Space root):

```
┌────────────────────────────────────────────────────────────┐
│  Speech Emotion Recognition                                │
│  Record or upload a short sentence. Model: WavLM probe ▾   │
├──────────────────────────────┬─────────────────────────────┤
│  🎙 [ record ]  ⬆ [ upload ]  │   Predicted: ANGRY  (89%)   │
│  waveform preview            │   angry   ████████████ .89  │
│                              │   disgust ██ .06            │
│  Examples (held-out actors): │   fear    ▌ .02  ...        │
│  [angry] [sad] [happy] ...   │                             │
│                              │   Timeline (if > 4 s):      │
│  [ Classify ]                │   ▁▃▇▇▅▂  emotion per 1.5 s │
├──────────────────────────────┴─────────────────────────────┤
│  How it works · Results · Limitations · GitHub · Colab     │
└────────────────────────────────────────────────────────────┘
```

Features, in priority order:

1. Record from the mic, or upload WAV/MP3/FLAC/M4A. Show the top emotion and a probability bar chart for all six classes.
2. 6–12 example clips from **held-out test actors** (licence permits redistribution with attribution), so visitors can see correct and incorrect cases without recording anything.
3. Model selector: WavLM probe (default) or hand-crafted ensemble, to show the 53% → 74% difference live.
4. A timeline view for clips longer than 4 s (`predict_timeline`).
5. "How it works" and "Limitations" tabs: speaker-independent evaluation, the per-class table and confusion matrix, the layer-scan plot, and a clear note that the model was trained on acted English speech and is not a lie detector or diagnostic tool.
6. Privacy: audio is processed in memory and not stored. Say so on the page.
7. Optional feedback button ("was this right? pick the true emotion"), logging only labels and scores, not audio, to a Hub dataset. This builds the real-world test set needed for Phase 4.

Deployment steps:

1. `app/requirements.txt` with CPU-only `torch`, plus `app/README.md` with the Space front-matter (`sdk: gradio`, `app_file: app.py`).
2. The Space installs `ser` from this GitHub repo (`git+https://...`) or vendors the package. It downloads the probe from the Hub at startup and warms up the encoder.
3. A GitHub Action syncs `app/` to the Space on push to `main`.
4. Add the Space link and a screenshot to the top of the README.

**Done when** a stranger can open the Space URL, record "I can't believe you did that" angrily, and get a sensible answer within ~2 s.

*Later, optional:* a custom-designed frontend (e.g. Next.js on Vercel) that calls the Space through `@gradio/client`, or a fully client-side version with the encoder exported to ONNX and run with `transformers.js` / `onnxruntime-web`, using the portable probe from Phase 1.

### Phase 3: more trustworthy evaluation (Kaggle, CPU or GPU)

> **Status (2026-10-23):** 3.1 is implemented in `kaggle/evaluate.py` (nested 5-fold actor CV for every encoder + hand-crafted) and queued on Kaggle together with step 4.1.

1. **Actor-grouped 5-fold CV over all 91 actors**, reported as mean ± std, for every model in the README. Keep the current split as the "legacy" number so old results stay comparable.
2. **Demographic breakdown** of accuracy by sex, age band and race from `VideoDemographics.csv`. Flag any subgroup gap above ~5 pp.
3. **Perceptual labels.** Compare the model with CREMA-D's crowd votes: does it fail on the clips humans also find ambiguous? Try training with soft labels, or dropping clips where fewer than 50% of raters agreed with the intended emotion.
4. **Cross-corpus test.** Without retraining, evaluate on RAVDESS, TESS and SAVEE (all in Kaggle's `dmitrybabko/speech-emotion-recognition-en`), mapping labels to the shared six (drop calm/surprise). This is the best available proxy for website performance.
5. **Real-mic test set.** 50–100 recordings from yourself and friends, plus the website's feedback log. This is a small number but it measures W1 directly.
6. Generate README tables from `artifacts/*.json` with a script instead of editing them by hand.

### Phase 4: better models (Kaggle GPU)

Each step is one Kaggle kernel. Only keep a step if it beats the previous best on **grouped 5-fold CV** *and* does not hurt the cross-corpus score.

| step | idea | expected effect |
|---|---|---|
| 4.1 | Run the existing pipeline for `hubert-base`, `wav2vec2-base`, `wavlm-large` | `wavlm-large` usually gives a few points more. This needs no new code. |
| 4.2 | Add other encoders: Whisper encoder (`openai/whisper-small` / `-medium`), `emotion2vec+` (SER-specific self-supervised model) | Emotion-specific or larger encoders are often stronger in published SER benchmarks. This is the cheapest possible gain. |
| 4.3 | Mean + std pooling and a learned softmax-weighted layer sum. Replace logistic regression with a small MLP head trained in PyTorch on cached embeddings | +1–3 pp, still cheap |
| 4.4 | **Fine-tune** the best base encoder end to end (freeze the CNN, weighted layer sum, attention pooling, 2-layer head), with augmentation: speed perturbation 0.9/1.0/1.1, additive noise (MUSAN / random SNR 5–20 dB), room impulse responses, low-pass/codec simulation of phone mics | Largest expected gain, especially out of domain. Fits a Kaggle T4/P100 with a base model, mixed precision and ~10 epochs. |
| 4.5 | Multi-corpus training (CREMA-D + RAVDESS + TESS + SAVEE, still speaker-independent per corpus) | Better robustness for the website, probably at a small cost on CREMA-D alone |
| 4.6 | Distil or quantise the winner (int8 dynamic quantisation, ONNX export) | Faster CPU inference and cheaper hosting, and enables the in-browser version |

Then refit the winner on all actors (`--full`), publish it to the Hub as a new model version, and point the Space at it.

### Phase 5: engineering quality (in parallel with the phases above)

- `pyproject.toml`, so `pip install -e .` (and `pip install git+https://github.com/...`) works for the Space and for Kaggle.
- `tests/` with pytest, using no dataset (synthetic audio): `parse_filename`, `len(extract_features(...)) == len(feature_names()) == 422`, `speaker_split` is actor-disjoint, `pool_layers` shapes, probe save/load round-trip, `Predictor` on synthetic audio.
- GitHub Actions: `ruff` and `pytest` on push.
- Merge `project_notebook.ipynb` and `colab_notebook.ipynb` into one notebook whose setup cell detects Colab, Kaggle or local. Read baseline numbers from `artifacts/metrics.json` instead of hard-coding them.
- **Remove the audio from git** (W5): use Kaggle `ejlok1/cremad` or the official CREMA-D repo with a `scripts/download_data.py`. Untracking the files going forward is safe. Purging them from history (`git filter-repo`) rewrites history and needs a force-push, so decide that separately.

## 5. Kaggle workflow (the default for all training from now on)

The Kaggle CLI is installed and authenticated as `mohameddddt`.

Target layout:

```
kaggle/
  embeddings/            # Phase 3/4.1–4.3: extract + probe, one encoder per run
    kernel-metadata.json
    run.py
  finetune/              # Phase 4.4
    kernel-metadata.json
    run.py
```

`kernel-metadata.json` (example):

```json
{
  "id": "mohameddddt/ser-embeddings",
  "title": "SER embeddings",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["ejlok1/cremad"]
}
```

`run.py` clones the repo (or `pip install git+...` once `pyproject.toml` exists). It sets `SER_DATA_DIR=/kaggle/input/cremad/AudioWAV` and `SER_ARTIFACTS_DIR=/kaggle/working/artifacts`, then runs `train_embeddings.run(...)` for the selected encoders. Everything in `/kaggle/working` becomes the kernel output.

Commands:

```bash
kaggle kernels push   -p kaggle/embeddings
kaggle kernels status mohameddddt/ser-embeddings
kaggle kernels output mohameddddt/ser-embeddings -p artifacts/kaggle/
```

Notes:

- The embedding cache (`*.npy`, ~150 MB per base encoder) can be published as a Kaggle dataset from the kernel output and attached to later kernels, so the probe and pooling experiments skip extraction.
- The GPU quota is about 30 h per week. Base-encoder extraction takes minutes, and fine-tuning about 1–2 h per run, so plan runs accordingly.
- The Colab notebooks stay as a "try it yourself" path for visitors. Development compute moves to Kaggle.

## 6. Suggested order of work

1. Phase 0 leftovers: configurable paths and split requirements. These are small, and the Kaggle kernels depend on them.
2. First Kaggle kernel: run 4.1 (`hubert-base`, `wav2vec2-base`, `wavlm-large`) and 3.1 (5-fold CV). This is cheap, and it fixes which encoder the website ships.
3. Phase 1: `Predictor`, portable probe, calibration, Hub upload.
4. Phase 2: Gradio Space live with the best probe. **First public milestone.**
5. Phase 3.4 and 3.5: cross-corpus and real-mic evaluation, to learn how far off the website really is.
6. Phase 4.2 → 4.4: stronger encoders, then fine-tuning with augmentation. Update the Space each time a model wins.
7. Phase 5 runs alongside. Purge the audio from git history once nothing depends on the clone containing it.
