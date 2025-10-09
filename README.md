# 🧠 VAE-Face-Generation (CelebA)

This project is my implementation of a **Variational Autoencoder (VAE)** for **face generation** trained on the **CelebA dataset**.  
I built and trained this model to explore how deep generative models can learn the underlying structure of human faces and reproduce realistic-looking samples from a latent distribution.

---

## 📚 Overview

A **Variational Autoencoder (VAE)** learns a probabilistic representation of data by encoding inputs into a lower-dimensional latent space and then decoding them back into the original form.  
In my implementation, I used the **CelebA face dataset**, where the model learns to compress a 128×128 face image into a 200-dimensional latent vector and then reconstruct it from that compact representation.

The model architecture was carefully designed to balance **feature extraction**, **compression**, and **reconstruction accuracy**, while keeping training stable using batch normalization and dropout.

---

## 🏗️ Architecture

The model has three major components:
1. **Encoder**
2. **Latent Space**
3. **Decoder**

### **1. Encoder**

The encoder’s job is to extract high-level features from input images and compress them into a dense latent representation.  
I used **four convolutional blocks** with downsampling, followed by a flattening operation.

Each block performs the following:
- A 2D convolution with stride 2 (to reduce spatial dimensions by half)
- Batch normalization for stability
- LeakyReLU activation with a very small negative slope (0.001)
- Dropout for regularization (20% rate)

The progression of feature map sizes during encoding is:

Input: 3 x 128 x 128
→ 32 x 64 x 64
→ 64 x 32 x 32
→ 64 x 16 x 16
→ 64 x 8 x 8
→ Flattened to 4096 features


From the final 4096-dimensional representation, I generated two 200-dimensional vectors:
- The **mean (μ)** of the latent distribution  
- The **log variance (log σ²)** of the latent distribution

These two vectors define a Gaussian distribution in latent space for each input image.

---

### **2. Latent Space**

Instead of directly encoding the image into a fixed latent vector, the model encodes it into a **distribution**.  
During training, I sample a latent vector `z` using the **reparameterization trick**, which ensures gradients can flow through the sampling process:

\[
z = μ + σ ⊙ ε, \quad \text{where } ε ∼ 𝒩(0, I)
\]

This stochastic approach forces the model to learn smooth, continuous representations — allowing me to generate new faces by sampling random points in latent space.

The **latent dimension** in my model is **200**, meaning each face is represented as a 200-element vector that captures its key visual traits (gender, expression, pose, lighting, etc.).

---

### **3. Decoder**

The decoder reverses the encoding process. It takes a 200-dimensional latent vector and reconstructs an image of size 3×128×128.

The decoding process starts with a fully connected layer that expands the 200-dimensional latent vector back to a 4096-dimensional feature map (corresponding to 64×8×8).  
This is followed by **four transpose-convolutional blocks** that progressively upsample the feature maps back to the original spatial size.

Each decoder block mirrors the encoder:
- Transposed convolution for upsampling
- Batch normalization
- LeakyReLU activation
- Dropout for stability

The final output layer uses a **transposed convolution** to map back to 3 channels (RGB).  
Because of convolutional padding effects, the output size slightly exceeds 128×128, so I added a small trimming layer to crop it back precisely to (3×128×128).

The decoding path progression looks like this:

Input latent vector (200)
→ Expand to 4096 → Reshape to 64 x 8 x 8
→ 64 x 16 x 16
→ 64 x 32 x 32
→ 32 x 64 x 64
→ 3 x 129 x 129 → Cropped to 3 x 128 x 128



This structure successfully reconstructs faces with fine details while maintaining overall smoothness and diversity in generated samples.

---

## ⚖️ Loss Function

I trained the model using the standard **VAE loss**, which combines two parts:

1. **Reconstruction Loss** — measures how accurately the output image matches the input (using either BCE or L1 loss).  
2. **KL Divergence Loss** — regularizes the latent space so that the learned distributions stay close to a unit Gaussian \( 𝒩(0, I) \).

The total loss function is:
\[
\mathcal{L} = \text{Reconstruction Loss} + \beta \times \text{KL Divergence}
\]

Balancing these two ensures that the model not only reconstructs faces accurately but also learns a well-behaved, smooth latent space for generation.

---

## 🧠 Dataset

I trained this model on the **CelebA** dataset, which contains over **200,000 aligned celebrity face images**.  
Each image was cropped and resized to **128×128**.  

This dataset is ideal for generative modeling because it covers a wide range of facial attributes — age, gender, pose, emotion, hairstyle, lighting, etc.

---

## 🚀 Training Process

I trained the model using **PyTorch** with:
- Batch size: 32  
- Latent dimension: 200  
- Optimizer: Adam (learning rate 1e-3)  
- Dropout: 0.2 in all convolutional layers  
- Epochs: around 50 for convergence  

During training, I monitored both reconstruction quality and KL loss.  
After sufficient epochs, the model started producing clear and diverse face reconstructions.

---

## 🧪 Results

After training, the model could:
- Reconstruct input faces with high fidelity.
- Generate new faces by sampling random latent vectors.
- Perform smooth **latent space interpolation**, morphing one face into another naturally.

Here is regenration of same image from latent space of dimension 200 features

 ![](https://raw.githubusercontent.com/0xprv/VAE-Face-Generation/refs/heads/main/result.png) 
---

## 🔄 Latent Space Interpolation

One of the most fascinating aspects of this model is latent space interpolation.  
By linearly interpolating between two latent vectors, I was able to smoothly transition from one face to another, watching gradual morphs in facial features, pose, and expressions.

This demonstrates that the VAE learned a **continuous and meaningful latent manifold** of human faces.

---

## 📖 References

- Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes*  
- Liu et al. (2015). *Deep Learning Face Attributes in the Wild (CelebA)*  
- PyTorch Documentation — [https://pytorch.org](https://pytorch.org)

---

## 👤 Author

**Reliable Information**  
I developed this VAE model to explore deep generative learning and understand the mathematical and architectural principles behind probabilistic image generation.  
This project was built from scratch to stay true to the fundamentals — no shortcuts, no pre-trained models, just honest experimentation and results.

---

## 🪪 License
MIT License — freely available for research and educational use.
