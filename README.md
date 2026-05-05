# DDC-KSE-ViT-GAN: Dual-Domain MRI Reconstruction

> **Accelerated MRI reconstruction combining dual-domain cascades, Fast Fourier Convolution, Swin Transformers, and GAN refinement — optimised for RTX 3090 / multi-coil FastMRI brain.**

---

## Architecture

DDC-KSE-ViT-GAN (Dual-Domain Cascaded K-Space Enhanced Vision Transformer GAN) processes k-space and image domain jointly through N cascades, followed by a GAN refinement stage for perceptual quality.

```
Undersampled k-space  [B, Nc, H, W]
         │
         ▼
┌─────────────────────────────────────────────┐
│         N × Dual-Domain Cascade             │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │   K-space Arm    │  │   Image Arm     │  │
│  │  FFC + UNet      │  │  UNet + ResNet  │  │
│  │  Swin Transformer│  │  Swin Transformer│  │
│  │  Data Consistency│  │  Data Consistency│  │
│  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    GAN Refinement   │
│  ResBlocks + Swin   │
│  PatchGAN (×2 disc) │
└─────────────────────┘
         │
         ▼
Reconstructed image  [B, H, W]
```

### Components

| Component | Details |
|---|---|
| K-space network | FFC + UNet + Swin Transformer (depths 1,1,1 / heads 2,4,8) |
| Image network | UNet + ResNet blocks + Swin Transformer |
| Data consistency | Multi-coil hard DC with sensitivity map estimation |
| GAN refinement | 6 residual blocks + Swin, 2× multi-scale PatchGAN discriminator |
| Sensitivity maps | Estimated from auto-calibration region |
| AMP | bfloat16 (auto-enabled on Ampere+ GPUs) |
| OOM handling | Automatic batch skip + cache clear on CUDA OOM |

### Model Size

| Module | Parameters |
|---|---|
| Generator | ~26.3 M |
| Discriminator | ~5.5 M |
| **Total** | **~31.8 M** |

---

## Results

Evaluated on FastMRI multicoil brain AXFLAIR at **8× acceleration**, per-volume mean:33.5458 dB, per-slice 34.88 dB PSNR (peaking at 38.97 dB and 0.94 SSIM)  and 0.916 SSIM.



### Reconstruction Example (8× acceleration, Brain T1-POST, Epoch 14)

![Reconstruction example](assets/reconstruction_example.png)

> Single slice: PSNR=38.37 dB | SSIM=0.9476 | NMSE=0.0055

---

## Requirements

```
torch >= 2.0
numpy
h5py
matplotlib
tensorboard
```

Install:
```bash
pip install -r requirements.txt
```

For official fastMRI metrics (recommended):
```bash
pip install fastmri
```

---

## Dataset

Download (permission required) [FastMRI multicoil brain] and set paths via environment variables:

```bash
# Windows (PowerShell)
$env:FASTMRI_TRAIN = "D:\train\multicoil_train"
$env:FASTMRI_VAL   = "D:\val\multicoil_val"
$env:FASTMRI_TEST  = "D:\test\multicoil_test_full"

# Linux / macOS
export FASTMRI_TRAIN=/data/multicoil_train
export FASTMRI_VAL=/data/multicoil_val
export FASTMRI_TEST=/data/multicoil_test_full
```

Or pass paths directly via `--train_path`, `--val_path`, `--test_path`.

---

## Training

**Standard training (8× acceleration):**
```bash
python mri_vit_gan_v10.py \
  --accelerations 8 \
  --epochs 30 \
  --eval_average volume
```

**Resume from best checkpoint:**
```bash
python mri_vit_gan_v10.py --resume_best --accelerations 8
```

**Fine-tuning (low LR):**
```bash
python mri_vit_gan_v10.py \
  --resume_best \
  --lr_g 1e-6 --lr_d 1e-6 \
  --epochs 10
```

**Disable GAN (pure reconstruction):**
```bash
python mri_vit_gan_v10.py --no_gan --accelerations 8
```

**Validation only:**
```bash
python mri_vit_gan_v10.py \
  --resume_best \
  --epochs 0 \
  --eval_average volume \
  --accelerations 8
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--accelerations` | `8` | Undersampling factor(s) |
| `--epochs` | `30` | Training epochs |
| `--lr_g` | `1e-6` | Generator learning rate |
| `--lr_d` | `1e-6` | Discriminator learning rate |
| `--eval_average` | `volume` | `volume` (per-patient) or `slice` |
| `--gan_start_epoch` | `15` | Epoch to activate GAN loss |
| `--use_gan` / `--no_gan` | GAN on | Toggle GAN refinement |
| `--amp_dtype` | `bf16` | AMP precision: `bf16` or `fp16` |
| `--num_workers` | `18` | DataLoader workers |
| `--vis_every` | `800` | Save visualisation every N steps |
| `--time_ckpt_minutes` | `25` | Periodic checkpoint interval |
| `--resume` | flag | Resume from last checkpoint |
| `--resume_best` | flag | Resume from best checkpoint |

---

## Outputs

```
outputs_vit_gan_integrated_cascade4/
├── checkpoints/
│   ├── vit_gan_best.pt          ← best validation PSNR
│   └── vit_gan_last.pt          ← most recent epoch
├── visualizations/
│   ├── train/                   ← Recon | ZF | GT | Error | RelError
│   ├── val/
│   └── best_worst/
├── tb/                          ← TensorBoard logs
└── results/
    ├── final_results.json
    └── final_results.csv
```

**TensorBoard:**
```bash
tensorboard --logdir outputs_vit_gan_integrated_cascade4/tb
```

---

## Hardware

Developed and tested on:
- **GPU:** NVIDIA RTX 3090 (24 GB VRAM)
- **OS:** Windows 10
- **CUDA:** 11.8
- **PyTorch:** 2.x with bfloat16 AMP

Memory optimisations included:
- Gradient checkpointing on Swin and GAN blocks
- Automatic OOM batch skip with cache flush
- `cudaMallocAsync` allocator
- `num_workers=0` safe mode for Windows HDF5

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ddckseviтgan2025,
  title  = {DDC-KSE-ViT-GAN: Dual-Domain Cascaded MRI Reconstruction
            with Swin Transformers and GAN Refinement},
  author = {Muhammed},
  year   = {2025},
  url    = {(https://github.com/MuhammedEminU/DDC-KSE-ViT-GAN)}
}
```

**Key references:**
- Yaman et al., "Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks", MRM 2020
- Huang et al., "SwinMR: Learning-Based MRI Reconstruction with Swin Transformer", 2022
- Sriram et al., "End-to-End Variational Networks for Accelerated MRI", MICCAI 2020
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer", ICCV 2021

---

## License

MIT License — see [LICENSE](LICENSE) for details.
