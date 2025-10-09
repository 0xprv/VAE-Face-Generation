# 🧠 VAE-Face-Generation (CelebA)

This is my implementation of a **Variational Autoencoder (VAE)** for **face generation** using the **CelebA dataset**.  
I built this project to explore probabilistic deep learning and to understand how latent space representations can be used to generate realistic human faces.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
  - [Encoder](#encoder)
  - [Latent Space](#latent-space)
  - [Decoder](#decoder)
  - [Loss Function](#loss-function)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [Latent Space Interpolation](#latent-space-interpolation)
- [References](#references)

---

## 🧩 Overview

In this project, I trained a **Variational Autoencoder (VAE)** on the **CelebA** dataset to generate human face images.  
A VAE is a generative model that learns to encode input data into a latent space and then decode it back to reconstruct or generate new data.

By training on the CelebA dataset, I aimed to teach the model to learn high-level facial representations — gender, expressions, hair color, pose, etc.  
After training, I was able to sample random latent vectors and generate new faces that never existed in the dataset.

---

## 🏗️ Architecture

The architecture I implemented is composed of an **Encoder**, a **Latent Space**, and a **Decoder**.

### **1. Encoder**
The encoder compresses the input image into two latent vectors — the **mean (μ)** and **log-variance (log σ²)** — representing a probability distribution.

I used a series of convolutional layers to extract features, followed by linear layers that output these two vectors.

**Example structure:**
