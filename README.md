# AlphaRoPE
AlphaRoPE:A Simple Yet Effective Length Extrapolation Method


This repo contains the code for the AlphaRoPE context window extension method.

## Paper

Paper: AlphaRoPE：A Simple Yet Efficient Length Extrapolation Method

## Reproduction

To reproduce, clone the repository and perform a local installation.

```bash
git clone https://github.com/<your-org>/AlphaRoPE.git
cd AlphaRoPE
pip install -e .
pip install -e ntk_yarn/
```

## Training

Prepare tokenized data, then fine-tune with DeepSpeed. Run `accelerate config` first to enable DeepSpeed acceleration.

```bash
python dataset_download.py
python tokenization.py
python truncate.py
accelerate launch finetune.py
```

Key files: `ntk_yarn/` (model implementations), `finetune.py`, `tokenization.py`, `truncate.py`.

## Evaluation

```bash
python eval/pass_key.py
python eval/ppl_sliding_window.py
python eval/ppl.py
```

For **LongBench**, clone [THUDM/LongBench](https://github.com/THUDM/LongBench) from GitHub (not included in this repo). Copy `ntk_yarn/` into `LongBench/LongBench/`, register your checkpoint in `config/model2path.json`, then run:

```bash
python pred.py
python eval.py
```

See the [LongBench README](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md) for details.
