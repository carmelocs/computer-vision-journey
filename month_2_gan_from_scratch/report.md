# 📝 Month 2 Report: Building a DCGAN from Scratch  

**Author**: Shawn Cheng  
**Date**: 22 Sep 2025

## 🎯 Goal

Implement and train a DCGAN on CIFAR-10 to understand GAN training dynamics.

## 🖼️ Results

- After 20 epochs, the generator produces structured but blurry images.
- Early mode collapse observed around epoch 5 → fixed by lowering LR slightly.
- FID score: ~65 (CIFAR-10 baseline is ~30–40 for good GANs).

![Epoch 20 Samples](./samples/epoch_20.png)

## 🔍 Key Learnings

1. **Training GANs is unstable** — small changes in hyperparams have big effects.
2. **Label smoothing helps** prevent discriminator from becoming too confident.
3. **Spectral normalization** could improve stability (next step).
4. **W&B made debugging easy** — seeing loss curves live was invaluable.

## ❌ Challenges

- Initial outputs were completely noisy.
- Had to restart training after fixing batch norm placement.
- CUDA memory error → reduced batch size from 256 → 128.

## ✅ Next Steps

- Try StyleGAN2 on faces
- Implement FID metric for automated evaluation
- Explore deepfake-specific losses (perceptual, LPIPS)

## 📚 References

- [DCGAN Paper (Radford et al.)](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
