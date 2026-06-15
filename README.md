**NameForge** is a character-level language generation project inspired by Andrej Karpathy's famous *Makemore* series. Built from scratch in **PyTorch**, the project explores the evolution of neural language models, progressing from simple statistical approaches to deep neural network architectures capable of generating realistic names character by character.

The primary goal of this project is to develop a deep understanding of how modern language models work under the hood by implementing every component from first principles.

---

## Features

- Character-level text generation
- Bigram Language Models
- Multi-Layer Perceptrons (MLPs)
- Embedding Layers
- Batch Normalization
- Manual Backpropagation
- WaveNet-inspired Architecture
- Training and sampling pipelines implemented from scratch

---

## Project Structure

```text
NameForge/
│
├── part1_bigram/
│   └── bigram.py
│
├── part3_batchnorm/
│   └── batchnorm.py
│
├── part4_backprop/
│   └── backprop.py
│
├── part5_wavenet/
│   └── wavenet.py
│
├── names.txt
└── README.md
```

---

## Learning Journey

### Part 1 — Bigram Language Model
- Built a statistical character-level language model
- Learned probability distributions and sampling
- Generated names using character transition frequencies

### Part 2 — Neural Network Language Model
- Implemented embeddings
- Built a Multi-Layer Perceptron (MLP)
- Trained using gradient descent

### Part 3 — Batch Normalization
- Implemented BatchNorm from scratch
- Improved training stability and convergence

### Part 4 — Backpropagation
- Derived gradients manually
- Verified gradients against PyTorch Autograd
- Developed a deeper understanding of optimization

### Part 5 — WaveNet Architecture
- Implemented hierarchical context aggregation
- Reduced parameter count while increasing model capacity
- Generated higher-quality names

---

## Results

### Model Performance

| Dataset | Loss |
|----------|----------|
| Train | 1.769 |
| Validation | 1.994 |

### Generated Names

```text
arlij
chetta
heago
rocklei
hendrix
jamylie
broxin
denish
anslibt
marianah
astavia
annayve
aniah
jayce
nodiel
remita
niyelle
jaylene
aiyan
aubreana
```

The model learns character-level patterns from a dataset of names and generates novel names that resemble realistic human names while remaining unique. As the architecture progresses from a Bigram model to a WaveNet-inspired network, the generated outputs become increasingly coherent and natural.



*Generated names vary depending on model architecture and training.*

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

---

## Key Concepts Explored

- Neural Networks
- Language Modeling
- Character Embeddings
- Gradient Descent
- Backpropagation
- Batch Normalization
- Probability & Statistics
- Deep Learning Fundamentals

---

## Motivation

Rather than treating deep learning frameworks as black boxes, this project focuses on understanding the mathematical foundations behind language models. Every major component is implemented and analyzed from first principles to build intuition for modern AI systems.

---

## Acknowledgements

This project is heavily inspired by Andrej Karpathy's excellent educational series:

- Makemore
- Neural Networks: Zero to Hero

Special thanks to Andrej Karpathy for making deep learning concepts accessible through hands-on implementation.

---



## Author

Krishan Pratap Sharma

If you found this project interesting, feel free to star the repository and connect with me on LinkedIn.
