# Month 2: Build a GAN from Scratch

## Goals

- Understand GAN architecture at code level
- Implement Generator and Discriminator in PyTorch
- Simulate training loop logic
- Prepare for real training on CIFAR-10 / CelebA

## Files

- `gan_concept_demo.ipynb`: Core GAN components and forward pass
- `models/generator.py`, `models/discriminator.py`: Modular classes
- `dcgan_train.py`: Training DCGAN
- `resize_celeba.py`: Resize Celeba images
- `fid_eval.py`: Generate images

## Next Steps

- Download CIFAR-10 / CelebA dataset
- Train DCGAN with logging (W&B)
- Evaluate using FID/LPIPS

## FID Evaluation Setup

To compute FID fairly, we resize real CelebA images to match generator output (64x64):

```bash
python resize_celeba.py
pytorch-fid ./outputs/fid_generated/ ./data/celeba_64x64/ --device cuda
```

## FID After Resizing Real Images to 64x64

- Generated: 5,000 × 64x64
- Real: 202,599 × 64x64 (resized from 178x218)
- FID: **123**

## Interpretation

This is expected for a 20-epoch DCGAN. Next step: train for 50–100 epochs to reduce FID below 60.
