# Deepfake Project Audit

## 🔍 Gaps & Questions

- How is spectral normalization used in the discriminator?
- What data augmentation is applied?
- How do we handle diverse skin tones and genders in training?

## 🔍 Architecture Deep Dive

| Component | Details |
|---------|--------|
| Model Type | StyleGAN2-ADA (assumed) |
| Generator | Input: latent z (512-dim), Output: 1024x1024 RGB |
| Discriminator | Uses R1 regularization, no progressive growing |
| Key Features | Adaptive augmentation, lazy regularization |

## 🖼️ Model Architecture Sketch

![Generator-Discriminator Flow](path-to-diagram.png)
> Simplified view of GAN components in our pipeline

## 💻 Key Code Snippet

```python
loss_G = adv_loss + λ₁ * perceptual_loss + λ₂ * identity_loss
```

## ⚙️ Training Configuration

- Framework: PyTorch
- Batch Size: 32 (synthetic)
- Optimizer: Adam (β₁=0.0, β₂=0.99)
- Learning Rate: 0.002
- Total Steps: ~150k
- Hardware: 1x NVIDIA RTX 5000 Ada

## 📊 Evaluation Metrics

| Metric | Purpose | Target |
|-------|--------|-------|
| FID   | Realism vs. real dataset | < 15 |
| LPIPS | Perceptual diversity | > 0.45 |
| Human Review | Artifact detection | Weekly audits |

## ❓ Open Questions

- How is data leakage prevented between train/val?
- Are synthetic videos temporally coherent?
- What post-processing (e.g., blending, sharpening) is applied?
