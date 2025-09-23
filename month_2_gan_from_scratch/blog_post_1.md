# How I Built My First GAN from Scratch (and Learned from Its Failures)

> *By Shawn Cheng | Computer Vision Engineer @ IDVerse Australia*

In Month 2 of my computer vision journey, I set out to build a **Deep Convolutional GAN (DCGAN)** from scratch using PyTorch. Not with a pre-trained model. Not by copying code blindly. But **from first principles** — layers, losses, and all.

The goal? To truly understand how generative models work under the hood. Because at IDVerse, where I work on deepfake generation for training data, knowing *how* things break is just as important as knowing how they work.

After 20 epochs of training on CIFAR-10, here’s what I got:

![Generated Images - Epoch 20](./samples/epoch_20.png)  
*64 synthetic images generated after 20 epochs. Still blurry — but showing signs of structure.*

They’re not photorealistic. They don’t look like birds or cars yet. But they’re not noise either.

And that, in itself, is a win.

This isn’t just about code. It’s about **learning through failure**, debugging in real time, and building intuition — one epoch at a time.

---

## 🛠️ What I Built

I implemented a **DCGAN** based on the [original paper by Radford et al. (2015)](https://arxiv.org/abs/1511.06434), trained on the **CIFAR-10 dataset** (60k 32x32 color images across 10 classes).

### Architecture Overview

- **Generator**:  
  Input: 100-dim latent vector → Transposed convolutions + BatchNorm → Output: 3x64x64 image (Tanh)
- **Discriminator**:  
  Input: Image → Strided convolutions + LeakyReLU → Output: Real/Fake probability (Sigmoid)
- **Loss**: Binary cross-entropy (BCE)
- **Optimizer**: Adam (`lr=0.0002`, `β₁=0.5`)
- **Batch size**: 128
- **Hardware**: NVIDIA RTX 2000 (CUDA 12.0)

All code is open-sourced in my learning repo:  
👉 [github.com/carmelocs/computer-vision-journey](https://github.com/carmelocs/computer-vision-journey)

---

## 🧪 The Training Process

Training a GAN is less like driving a car and more like balancing two wrestlers.

Here’s how it unfolded:

| Epoch | Observations |
|-------|-------------|
| 1–5   | Pure noise. Discriminator dominated. Generator made no progress. |
| 6–10  | First hints of structure. Color blobs emerged. Mode collapse started. |
| 11–15 | Repeated patterns — generator “cheated” by producing similar outputs. |
| 16–20 | Slight improvement in diversity. Still blurry, but coherent. |

The key insight? **GANs don’t learn smoothly**. They oscillate between chaos and order — and your job is to keep them dancing.

---

## 🔍 Key Lessons from the Trenches

### 1. **Stability Is Everything**

GANs are notoriously unstable. I had to:

- Lower the learning rate after epoch 5
- Reduce batch size from 256 → 128 to avoid CUDA OOM
- Use **Weights & Biases (W&B)** to track loss curves in real time

💡 Pro tip: Start with **label smoothing** and **gradient penalty** in future projects.

### 2. **You Can’t Just “Train” a GAN**

It’s a game of balance:

- If the **discriminator gets too strong**, the generator gets vanishing gradients.
- If the **generator wins early**, mode collapse sets in.

I found success by alternating updates 1:1 and monitoring both losses carefully.

### 3. **Blurriness Is Normal**

At epoch 20, expecting sharp images is unrealistic. Realistic results take hundreds of epochs — especially without tricks like spectral normalization or progressive growing.

But blurriness ≠ failure. It means learning is happening.

---

## 📊 Evaluation: Beyond Visual Inspection

I used:

- **W&B logs** to monitor `loss_gen` and `loss_disc`
- **Visual inspection** of generated samples every 5 epochs
- **FID estimate**: ~75 (baseline for good GANs on CIFAR-10 is ~30–40)

While performance isn’t stellar yet, the trajectory is positive — and that’s what matters.

---

## ✅ What’s Next?

Now that I’ve built a working DCGAN, I’m moving to:

- **StyleGAN2-ADA**: For higher-resolution, more realistic face generation
- **Latent space editing**: Manipulating attributes like age, expression, lighting
- **Deepfake detection**: Using synthetic data from my own models to train robust detectors

Because at IDVerse, we don’t just generate fakes — we also need to detect them.

---

## 🧩 Final Thoughts

Building a GAN from scratch taught me more than any tutorial ever could. It wasn’t about perfect results — it was about understanding the **dance between generator and discriminator**, the **importance of hyperparameters**, and the **value of patience**.

And that’s the real value of learning in public:  
You’re not just coding — you’re growing.

If you’re starting your own CV journey, remember:
> **Fail fast. Learn faster. Iterate constantly.**

Because mastery isn’t about getting it right the first time.  
It’s about getting it right *eventually* — and knowing how you got there.

---

## 📚 Resources

- [DCGAN Paper (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Weights & Biases (W&B)](https://wandb.ai/site)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🔗 Connect With Me

- GitHub: [Shawn Cheng](https://github.com/carmelocs)
- LinkedIn: [Shawn Cheng](https://www.linkedin.com/in/shawn-cheng-a41647105/)

Let me know if you try this yourself — I’d love to see your generated images!

`# machinelearning #computervision #generativemodels #deeplearning`
