<h1 align="center">
  InkFormer: From Characters to Signatures via Style-Transformation-Aware Online Handwriting Generation
</h1>

This repository provides the training, inference, and evaluation code for **InkFormer-Char**, the character-level handwriting generation model. Additionally, the training code for both the content and style evaluators is available.

## Updates
- [2026/10/23] Code and pretrained weights are released.
- [2026/10/15] InkFormer is accepted by [TMLR 2026](https://openreview.net/forum?id=Nbhw2Y5Nmh) 🎉🎉🎉.

## Repository Structure

```
InkFormer/
├── main.py                     # Training entry
├── inference.py                # Inference entry
├── dataset_char.py / utils.py  # Data loading and shared utilities
├── classifiers/                # Training code for the content/style classifiers (evaluators)
├── evaluation/                 # Evaluation scripts (nDTW / content / style)
├── modules/                    # Model architecture (Transformer_AR and encoders)
├── scripts/                    # train.sh / train_cont.sh / train_sty.sh / infer.sh / evaluate.sh
└── data_ckpt/
    ├── ckpt/                   # Model checkpoints
    └── datas/                  # Datasets
```

## Environment Setup

A Linux machine with NVIDIA GPUs is recommended. Install the dependencies:

```bash
conda create -n inkformer python=3.9 -y
conda activate inkformer
pip install -r requirements.txt
```

Main dependencies: PyTorch, torchvision, accelerate, numpy, Pillow, matplotlib, tqdm, fastdtw, scipy, einops.

## Data & Checkpoints

Download: [Google Drive](https://drive.google.com/drive/folders/11hZj06JCcMAIFrA8MwBeZ1Jg8UU2u7Ac)

After downloading, place the files under `data_ckpt/` as follows:

```
data_ckpt/
├── ckpt/
│   ├── ar_generator_5000.pth    # Generator weights (for inference)
│   ├── cont_evaluator_2000.pth  # Content evaluator weights
│   ├── sty_evaluator_10000.pth  # Style evaluator weights
│   ├── lookup_char2int.npy      # Lookup table for content evaluation
│   └── lookup_idx2ori_on.npy    # Lookup table for style evaluation
└── datas/
    ├── train/                   # Training set (6 pools)
    │   ├── Pot1.0Train_pp1.npy
    │   ├── Pot1.1Train_pp1.npy
    │   ├── Pot1.2Train_pp1.npy
    │   ├── Pot1.0Test_pp1.npy
    │   ├── Pot1.1Test_pp1.npy
    │   └── Pot1.2Test_pp1.npy
    ├── test/                    # Test set
    │   └── competition_set_pp1.npy
    └── standard/                # Standard characters
        └── stand_zi.npy
```

## Training

### Generator

```bash
bash scripts/train.sh
```

- Uses 4 GPUs by default (`CUDA_VISIBLE_DEVICES=0,1,2,3` in `scripts/train.sh`; adjust to your setup)
- Multi-GPU training is launched with accelerate; run `accelerate config` first if you have not configured it yet
- Key hyperparameters (batch-user, sample-per-user, n-reference, lr, epoch-end, etc.) can be edited directly in `scripts/train.sh`
- Checkpoints are saved every 500 epochs (save-epoch) to `data_ckpt/ckpt/ar_generator_{epoch}.pth`
- To resume training, set `--epoch-begin` to an existing epoch; `data_ckpt/ckpt/ar_generator_{epoch_begin}.pth` will be loaded automatically

### Classifiers (Evaluators)

```bash
bash scripts/train_cont.sh   # Content classifier → data_ckpt/ckpt/cont_evaluator_{epoch}.pth
bash scripts/train_sty.sh    # Style classifier  → data_ckpt/ckpt/sty_evaluator_{epoch}.pth
```

If `logs/infer/syn_saver_ref_15.npy` (the generator's inference output) exists, classifier training additionally monitors the accuracy on synthesized characters (`Acc_syn`); otherwise this monitoring is skipped automatically, which does not affect training or checkpoint saving.

## Inference

```bash
bash scripts/infer.sh
```

- Runs on a single GPU (GPU 0)
- Loads `data_ckpt/ckpt/ar_generator_5000.pth` by default; edit the `--epoch` argument in `scripts/infer.sh` to use another checkpoint
- Synthesized results are saved to `logs/infer/syn_saver_ref_15.npy`
- Side-by-side images of standard / synthesized / ground-truth characters are written to `logs/infer/img_infer/`

## Evaluation

```bash
bash scripts/evaluate.sh
```

Runs three evaluations on the inference output `logs/infer/syn_saver_ref_15.npy`:

1. **nDTW** (`evaluation/evaluate_ndtw_1.py`): trajectory similarity
2. **Content score** (`evaluation/evaluate_content_score.py`): character recognizability, computed with the bundled content evaluator
3. **Style score** (`evaluation/evaluate_style_score.py`): style similarity, computed with the bundled style evaluator

## Pipeline Overview

```
scripts/train.sh    →  data_ckpt/ckpt/ar_generator_*.pth (checkpoints)
scripts/infer.sh    →  logs/infer/syn_saver_ref_15.npy (synthesized results)
scripts/evaluate.sh →  nDTW / content score / style score
```

## Acknowledgements
- [CASIA-OLHWDB](https://nlpr.ia.ac.cn/databases/handwriting/home.html), [ICDAR-2013](https://ieeexplore.ieee.org/document/6628856) and [Make Me a Hanzi](https://github.com/skishore/makemeahanzi) are valuable resources for this work.


## Citation
If you find this work useful or use this code in your research, please consider citing the following paper.

```bibtex
@article{
  pan2026inkformer,
  title={InkFormer: From Characters to Signatures via Style-Transformation-Aware Online Handwriting Generation},
  author={Wei Pan and Yilin Shi and Lianwen Jin},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=Nbhw2Y5Nmh},
  note={}
}
```

## License

Our code is released under the MIT License. The pretrained weights, which are trained using part of the CASIA-OLHWDB dataset, follow the [original license](https://nlpr.ia.ac.cn/databases/handwriting/Application_form.html) of the dataset.

