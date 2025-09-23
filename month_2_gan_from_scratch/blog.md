# How I Built My First GAN from Scratch (and Learned from Its Failures)

> *By Shawn Cheng | Computer Vision Engineer @ IDVerse Australia*

In Month 2 of my computer vision journey, I set out to build a **Deep Convolutional GAN (DCGAN)** from scratch using PyTorch. The goal? To understand how generative models work under the hood — not just use them.

After 20 epochs of training, here’s what I got:

![Generated Images - Epoch 20](https://github.com/carmelocs/computer-vision-journey/blob/main/month_2_gan_from_scratch/samples/epoch_20.png?raw=true)

Blurry, abstract, but full of promise.

This isn’t just about code. It’s about **learning through failure**, and why that’s essential in AI.

---

## 🛠️ What I Built

I implemented a **DCGAN** on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes (cars, birds, cats, etc.).

The architecture:

- **Generator**: Transposed convolutions, BatchNorm, ReLU → Tanh output
- **Discriminator**: Conv layers, LeakyReLU, Sigmoid output
- **Loss**: Binary cross-entropy (original GAN loss)
- **Optimizer**: Adam (β₁=0.5)
- **GPU**: NVIDIA RTX 2000 (CUDA 12.0)

All code is open-sourced at:  
👉 [https://github.com/carmelocs/computer-vision-journey/tree/main/month_2_gan_from_scratch](https://github.com/carmelocs/computer-vision-journey/tree/main/month_2_gan_from_scratch)

---

## 🧪 The Training Process

I trained for 20 epochs with a batch size of 128. Here’s what happened:

| Epoch | Observations |
|-------|-------------|
| 1–5   | Outputs were pure noise. Discriminator dominated. |
| 6–10  | First hints of structure appeared. Color patches emerged. |
| 11–15 | Mode collapse: generator repeated similar patterns. |
| 16–20 | Improved diversity, but still blurry. |

The key insight? **GANs don’t learn linearly.** They oscillate between chaos and order.

---

## 🔍 Key Lessons

### 1. **Stability is Everything**

GANs are notoriously unstable. I had to:

- Lower the learning rate after epoch 5
- Reduce batch size to avoid GPU OOM
- Use W&B to track loss curves in real-time

💡 Pro tip: Start with **label smoothing** and **gradient clipping**.

### 2. **You Can’t Just “Train” a GAN**

It’s a game of balance:

- Generator tries to fool the discriminator
- Discriminator tries to get better at detecting fakes

If one gets too strong, training fails.

### 3. **Blurriness Is Normal**

At epoch 20, my images are blurry — but that’s expected. Realistic results come after hundreds of epochs.

---

## 📊 Evaluation

I used **visual inspection** and **W&B logs** to monitor progress:

- Loss curves showed convergence
- Generated samples improved over time
- No major artifacts (e.g., checkerboard patterns)

FID score estimate: ~70–80 (baseline for CIFAR-10 is ~30–40). But again — this is early.

---

## ✅ What’s Next?

Now that I’ve built a working GAN, I’m moving to:

- **StyleGAN2**: For higher-resolution, more realistic outputs
- **Latent space editing**: Manipulating facial attributes (age, expression)
- **Deepfake detection**: Using my model to generate synthetic data for training detectors

---

## 🧩 Final Thoughts

Building a GAN from scratch taught me more than any tutorial ever could. It wasn’t about perfect results — it was about **understanding the process**.

And that’s the real value of learning in public: you’re not just coding — you’re growing.

If you’re starting your own CV journey, remember:
> **Fail fast, learn fast, iterate fast.**

---

## 📚 Resources

- [DCGAN Paper (Radford et al.)](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Weights & Biases for Experiment Tracking](https://wandb.ai/site)

---

## 🔗 Connect With Me

- GitHub: [Shawn Cheng](https://github.com/carmelocs)
- LinkedIn: [Shawn Cheng](https://www.linkedin.com/in/shawn-cheng-a41647105/)

Let me know if you try this yourself — I’d love to see your results!

`# machinelearning #computervision #generativemodels #deeplearning`
