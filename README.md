<div align="center">

# ML → Deep Learning → Neural Networks → Transformers/LLMs

### The Complete Roadmap from Math to Modern AI

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/LLMs-412991?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Learning-2ea44f?style=for-the-badge)
![License](https://img.shields.io/badge/Level-Beginner_to_Advanced-blueviolet?style=for-the-badge)

*Learn the story, not just the algorithms — every stage below explains **what it is** and **why it exists**.*

</div>

<br>

<div align="center">

### 📑 Table of Contents

![01](https://img.shields.io/badge/01-Foundation-4C51BF?style=for-the-badge)&nbsp;
![02](https://img.shields.io/badge/02-Classical_ML-C05621?style=for-the-badge)&nbsp;
![03](https://img.shields.io/badge/03-Neural_Nets-805AD5?style=for-the-badge)&nbsp;
![04](https://img.shields.io/badge/04-Backprop-DD6B20?style=for-the-badge)
<br><br>
![05](https://img.shields.io/badge/05-From_Scratch-E53E3E?style=for-the-badge)&nbsp;
![06](https://img.shields.io/badge/06-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)&nbsp;
![07](https://img.shields.io/badge/07-CNN_Vision-3182CE?style=for-the-badge)&nbsp;
![08](https://img.shields.io/badge/08-RNN_LSTM-2C7A7B?style=for-the-badge)
<br><br>
![09](https://img.shields.io/badge/09-Transformers-6B46C1?style=for-the-badge)&nbsp;
![10](https://img.shields.io/badge/10-BERT_GPT-10A37F?style=for-the-badge)&nbsp;
![11](https://img.shields.io/badge/11-LLM_Eng-553C9A?style=for-the-badge)&nbsp;
![12](https://img.shields.io/badge/12-RAG-2B6CB0?style=for-the-badge)
<br><br>
![13](https://img.shields.io/badge/13-AI_Agents-B7791F?style=for-the-badge)&nbsp;
![Ladder](https://img.shields.io/badge/★-Project_Ladder-1A202C?style=for-the-badge)

</div>

| Jump to | Focus |
|---|---|
| [🟣 01 · Foundation](#-01--foundation--python--math) | Python + Math |
| [🟠 02 · Classical ML](#-02--classical-machine-learning) | Regression / Classification |
| [🟪 03 · Neural Networks](#-03--neural-networks--the-most-important-stage) | The Neuron |
| [🟧 04 · Backprop + Optimization](#-04--backpropagation--optimization) | Gradients |
| [🟥 05 · NN From Scratch](#-05--build-neural-networks-from-scratch) | NumPy Only |
| [🔥 06 · PyTorch](#-06--pytorch) | Real Framework |
| [🔵 07 · CNN](#-07--cnn--computer-vision) | Computer Vision |
| [🟦 08 · RNN → LSTM → GRU](#-08--rnn--lstm--gru) | Sequences |
| [🟣 09 · Transformers](#-09--transformers--the-big-one) | Attention |
| [🟢 10 · BERT / GPT](#-10--bert--gpt) | Encoder / Decoder |
| [🟪 11 · LLM Engineering](#-11--llm-engineering) | Fine-tuning |
| [🔷 12 · RAG](#-12--rag) | Retrieval |
| [🟡 13 · AI Agents](#-13--ai-agents) | Tool Use |
| [⚫ Project Ladder](#-project-ladder) | Build in Order |

---

## 🧭 The Big Picture

```
                    AI · ARTIFICIAL INTELLIGENCE
                              │
                              ▼
                    MACHINE LEARNING
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
      Supervised         Unsupervised       Reinforcement
          │                   │
          ▼                   ▼
   Regression /            Clustering
   Classification          Dimensionality
          │
          ▼
                    DEEP LEARNING
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
       ANN            CNN              RNN
        │                              │
        │                              ▼
        │                            LSTM
        │                              │
        └──────────────┬───────────────┘
                       ▼
                  TRANSFORMERS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Attention      BERT          GPT
          │                         │
          ▼                         ▼
       Encoder                  Decoder
          │                         │
          └────────────┬────────────┘
                       ▼
                       LLMs
                       │
                       ▼
                 Generative AI
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         RAG          Agents       Fine-tuning
```

**Read it top to bottom:** AI is the umbrella field. Machine Learning is the subset where systems learn patterns from data instead of hardcoded rules. Deep Learning is the subset of ML that uses layered neural networks. Transformers are the specific deep-learning architecture that made today's LLMs possible. Everything below walks that exact path.

---

## 🟣 01 · Foundation — Python + Math

![Priority](https://img.shields.io/badge/PRIORITY-Don't_Rush_This_Stage-E53E3E?style=for-the-badge)

**Why this stage exists:** every ML algorithm is math wearing a Python costume. If you skip the math, you'll be able to copy-paste code but not debug it, tune it, or read a paper and understand what's actually happening. This stage is the price of admission — pay it once, properly.

### 🐍 Python Toolkit

![Variables](https://img.shields.io/badge/-Variables-4A5568?style=flat-square)
![Functions](https://img.shields.io/badge/-Functions-4A5568?style=flat-square)
![Classes](https://img.shields.io/badge/-Classes-4A5568?style=flat-square)
![Lists/Dicts](https://img.shields.io/badge/-Lists%2FDicts-4A5568?style=flat-square)

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/-Git%2FGitHub-181717?style=flat-square&logo=github&logoColor=white)

*NumPy/Pandas matter more than the syntax itself: almost every ML library represents data as arrays and dataframes, so fluency here is fluency everywhere downstream.*

### 📐 Mathematics
*You don't need to become a mathematician — focus on the math ML actually uses.*

<table>
<tr>
<td width="20%" align="center"><img src="https://img.shields.io/badge/-Linear_Algebra-4C51BF?style=for-the-badge" /></td>
<td>
<img src="https://img.shields.io/badge/-Vectors-434190?style=flat-square" />
<img src="https://img.shields.io/badge/-Matrices-434190?style=flat-square" />
<img src="https://img.shields.io/badge/-Dot_Product-434190?style=flat-square" />
<img src="https://img.shields.io/badge/-Matrix_Multiplication-434190?style=flat-square" />
<img src="https://img.shields.io/badge/-Transpose-434190?style=flat-square" />
<img src="https://img.shields.io/badge/-Eigenvalues%2FEigenvectors-434190?style=flat-square" />
<br><br>
<i>Why: every layer of a neural network is literally a matrix multiplication. Data, weights, and even attention are all vectors/matrices being transformed.</i>
</td>
</tr>
<tr>
<td align="center"><img src="https://img.shields.io/badge/-Calculus-2B6CB0?style=for-the-badge" /></td>
<td>
<img src="https://img.shields.io/badge/-Derivatives-2C5282?style=flat-square" />
<img src="https://img.shields.io/badge/-Partial_Derivatives-2C5282?style=flat-square" />
<img src="https://img.shields.io/badge/-Gradients-2C5282?style=flat-square" />
<img src="https://img.shields.io/badge/-Chain_Rule-2C5282?style=flat-square" />
<br><br>
<i>Why: a model learns by knowing which direction to nudge each weight. That "direction" is a derivative, and the chain rule is literally how backpropagation works.</i>
</td>
</tr>
<tr>
<td align="center"><img src="https://img.shields.io/badge/-Probability-2C7A7B?style=for-the-badge" /></td>
<td>
<img src="https://img.shields.io/badge/-Probability-285E61?style=flat-square" />
<img src="https://img.shields.io/badge/-Conditional_Probability-285E61?style=flat-square" />
<img src="https://img.shields.io/badge/-Bayes_Theorem-285E61?style=flat-square" />
<img src="https://img.shields.io/badge/-Distributions-285E61?style=flat-square" />
<img src="https://img.shields.io/badge/-Expectation-285E61?style=flat-square" />
<img src="https://img.shields.io/badge/-Variance-285E61?style=flat-square" />
<br><br>
<i>Why: model outputs are often probabilities (softmax), and "confidence," sampling, and uncertainty all come from probability theory.</i>
</td>
</tr>
<tr>
<td align="center"><img src="https://img.shields.io/badge/-Statistics-276749?style=for-the-badge" /></td>
<td>
<img src="https://img.shields.io/badge/-Mean%2FMedian-22543D?style=flat-square" />
<img src="https://img.shields.io/badge/-Std_Deviation-22543D?style=flat-square" />
<img src="https://img.shields.io/badge/-Correlation-22543D?style=flat-square" />
<img src="https://img.shields.io/badge/-Sampling-22543D?style=flat-square" />
<img src="https://img.shields.io/badge/-Hypothesis_Testing-22543D?style=flat-square" />
<br><br>
<i>Why: you use statistics to judge whether your data is clean, whether a result is real or noise, and whether your model actually improved.</i>
</td>
</tr>
</table>

> 🎯 **Goal of this stage:** be able to read the equations in an ML paper or codebase and know what each symbol *means*, not just what to type.

---

## 🟠 02 · Classical Machine Learning

> Learn the **idea** before the library.

```
Input X → Model → Prediction ŷ → Compare with y → Loss → Update model
```

**Why this stage exists:** classical ML teaches the full learning loop — data in, prediction out, measure the error, adjust — in its simplest form, without the complexity of layers and backprop. Every deep learning model later is just a bigger, more flexible version of this exact loop.

![Regression](https://img.shields.io/badge/REGRESSION-2B6CB0?style=for-the-badge)
![Linear](https://img.shields.io/badge/-Linear-2C5282?style=flat-square)
![Polynomial](https://img.shields.io/badge/-Polynomial-2C5282?style=flat-square)
![Ridge](https://img.shields.io/badge/-Ridge-2C5282?style=flat-square)
![Lasso](https://img.shields.io/badge/-Lasso-2C5282?style=flat-square)

*Regression predicts a continuous number (a price, a score). Ridge and Lasso add a penalty term so the model doesn't overfit by relying too heavily on any one feature.*

![Classification](https://img.shields.io/badge/CLASSIFICATION-C05621?style=for-the-badge)
![Logistic](https://img.shields.io/badge/-Logistic_Regression-9C4221?style=flat-square)
![KNN](https://img.shields.io/badge/-KNN-9C4221?style=flat-square)
![Naive Bayes](https://img.shields.io/badge/-Naive_Bayes-9C4221?style=flat-square)
![Trees](https://img.shields.io/badge/-Decision_Trees-9C4221?style=flat-square)
![Random Forest](https://img.shields.io/badge/-Random_Forest-9C4221?style=flat-square)
![Gradient Boosting](https://img.shields.io/badge/-Gradient_Boosting-9C4221?style=flat-square)
![XGBoost](https://img.shields.io/badge/-XGBoost-9C4221?style=flat-square)

*Classification predicts a category (spam vs not-spam). Trees split data on yes/no questions; Random Forest and XGBoost combine many trees so their combined errors cancel out — this is why they dominate tabular-data competitions.*

### 🔑 You MUST Understand

![Training](https://img.shields.io/badge/-Train%2FVal%2FTest_Data-718096?style=flat-square)
![Features](https://img.shields.io/badge/-Features_%26_Labels-718096?style=flat-square)
![Loss](https://img.shields.io/badge/-Loss_Function-718096?style=flat-square)
![Optimization](https://img.shields.io/badge/-Optimization-718096?style=flat-square)
![Overfitting](https://img.shields.io/badge/-Overfitting%2FUnderfitting-718096?style=flat-square)
![Bias-Variance](https://img.shields.io/badge/-Bias_vs_Variance-718096?style=flat-square)
![Regularization](https://img.shields.io/badge/-Regularization-718096?style=flat-square)
![Cross-validation](https://img.shields.io/badge/-Cross--validation-718096?style=flat-square)
![Feature Eng](https://img.shields.io/badge/-Feature_Engineering-718096?style=flat-square)

*Train/val/test exists so you never grade your model on the same data it studied from. Overfitting is memorizing the training set instead of learning the pattern — regularization and cross-validation are the main defenses against it.*

### 🛠️ Projects

![Project](https://img.shields.io/badge/BUILD-House_Price_Predictor-2ea44f?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Disease_Risk_Classifier-2ea44f?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Fraud_Detection-2ea44f?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Spam_Classifier-2ea44f?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Student_Performance_Predictor-2ea44f?style=flat-square&logo=target&logoColor=white)

---

## 🟪 03 · Neural Networks — The Most Important Stage

![Core Stage](https://img.shields.io/badge/⭐-Core_Stage-805AD5?style=for-the-badge)

**Why this stage exists:** classical ML models can only draw fairly simple decision boundaries. Stacking neurons into layers lets a model approximate *any* function, no matter how complex — that flexibility is the entire reason deep learning works.

### The Single Neuron
```
x₁ ──w₁──┐
x₂ ──w₂──┤──► Σ + b ──► Activation ──► Output
x₃ ──w₃──┘

z = w₁x₁ + w₂x₂ + w₃x₃ + b
a = activation(z)
```

*A neuron is just a weighted sum plus a bias, squeezed through a nonlinear function. The weights decide how much each input matters; the bias shifts the decision threshold; the activation is what stops the whole network from collapsing into one big linear equation.*

### ⚡ Activation Functions

![Sigmoid](https://img.shields.io/badge/-Sigmoid-805AD5?style=for-the-badge)
![Tanh](https://img.shields.io/badge/-Tanh-6B46C1?style=for-the-badge)
![ReLU](https://img.shields.io/badge/-ReLU-D53F8C?style=for-the-badge)
![Leaky ReLU](https://img.shields.io/badge/-Leaky_ReLU-B83280?style=for-the-badge)
![Softmax](https://img.shields.io/badge/-Softmax-97266D?style=for-the-badge)

*Sigmoid/Tanh squash values into a fixed range (good for probabilities, bad for deep nets — they cause vanishing gradients). ReLU fixed that by being simple and non-saturating, which is why it's the modern default. Softmax turns a row of numbers into a probability distribution — it's what sits at the output of a classifier.*

### 🏗️ The Architecture
```
Input Layer → Hidden Layer → Hidden Layer → Output Layer
```
*This is an **Artificial Neural Network (ANN)**. Each hidden layer learns increasingly abstract features from the layer before it — this is exactly what "deep" in deep learning refers to.*

---

## 🟧 04 · Backpropagation + Optimization

> This is where you really start understanding deep learning.

```
Forward Pass → Prediction → Loss → Backpropagation
   → Gradient → Optimizer → Update Weights → Forward Pass Again
```

**Why this stage exists:** a network with random weights is useless. Backpropagation is the algorithm that tells every single weight in the network exactly how much it contributed to the error, so the optimizer knows how to fix it. This loop, repeated thousands of times, *is* training.

**Loss Functions**

![MSE](https://img.shields.io/badge/-MSE-DD6B20?style=for-the-badge)
![BCE](https://img.shields.io/badge/-Binary_Cross_Entropy-DD6B20?style=for-the-badge)
![CCE](https://img.shields.io/badge/-Categorical_Cross_Entropy-DD6B20?style=for-the-badge)

*MSE punishes the squared distance between prediction and truth — good for regression. Cross-entropy punishes confident-but-wrong probabilities — good for classification. The loss function is the single number the entire training process is trying to minimize.*

**Optimization Path**

![GD](https://img.shields.io/badge/1-Gradient_Descent-FEEBC8?style=flat-square&labelColor=DD6B20&color=DD6B20)
➜
![MiniGD](https://img.shields.io/badge/2-Mini--batch_GD-FBD38D?style=flat-square&labelColor=DD6B20&color=DD6B20)
➜
![SGD](https://img.shields.io/badge/3-SGD-F6AD55?style=flat-square&labelColor=DD6B20&color=DD6B20)
➜
![Momentum](https://img.shields.io/badge/4-Momentum-ED8936?style=flat-square&labelColor=DD6B20&color=DD6B20)
➜
![Adam](https://img.shields.io/badge/5-Adam-DD6B20?style=flat-square&labelColor=DD6B20&color=DD6B20)
➜
![AdamW](https://img.shields.io/badge/6-AdamW-C05621?style=flat-square&labelColor=DD6B20&color=C05621)

*Plain Gradient Descent computes on the whole dataset (slow). SGD/mini-batch trade some accuracy for huge speed. Momentum smooths out noisy updates. Adam adapts the learning rate per-parameter — it's the default in almost every modern training script, and AdamW fixes how it handles regularization.*

---

## 🟥 05 · Build Neural Networks From Scratch

![Rule](https://img.shields.io/badge/RULE-Don't_Only_Use_PyTorch-E53E3E?style=for-the-badge)

**Why this stage exists:** frameworks hide backpropagation behind `.backward()`. If you've never computed a gradient by hand, that call is magic, not understanding. Building it once in raw NumPy converts "I've heard of backprop" into "I could reimplement it."

Build a tiny neural network using:

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

Implement yourself:

![Neuron](https://img.shields.io/badge/1-Neuron-FEB2B2?style=flat-square&labelColor=C53030&color=C53030)
➜
![Forward](https://img.shields.io/badge/2-Forward_Propagation-FC8181?style=flat-square&labelColor=C53030&color=C53030)
➜
![Loss](https://img.shields.io/badge/3-Loss-F56565?style=flat-square&labelColor=C53030&color=C53030)
➜
![Gradient](https://img.shields.io/badge/4-Gradient_Calc-E53E3E?style=flat-square&labelColor=C53030&color=E53E3E)
➜
![Backprop](https://img.shields.io/badge/5-Backpropagation-C53030?style=flat-square&labelColor=C53030&color=C53030)
➜
![GD](https://img.shields.io/badge/6-Gradient_Descent-9B2C2C?style=flat-square&labelColor=C53030&color=9B2C2C)
➜
![Training](https://img.shields.io/badge/7-Training-742A2A?style=flat-square&labelColor=C53030&color=742A2A)

Then reproduce the exact same network in PyTorch and compare. You'll understand far more deeply than someone who only ever writes:
```python
model = nn.Sequential(...)
```

---

## 🔥 06 · PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Why this stage exists:** now that you understand the mechanics, PyTorch lets you build real, GPU-accelerated networks without hand-deriving every gradient — it automates exactly the process you just did manually in stage 5.

**Core Concepts**

![Tensor](https://img.shields.io/badge/-Tensor-742A2A?style=flat-square)
![Dataset](https://img.shields.io/badge/-Dataset-742A2A?style=flat-square)
![DataLoader](https://img.shields.io/badge/-DataLoader-742A2A?style=flat-square)
![nn.Module](https://img.shields.io/badge/-nn.Module-742A2A?style=flat-square)
![forward()](https://img.shields.io/badge/-forward()-742A2A?style=flat-square)
![loss](https://img.shields.io/badge/-loss-742A2A?style=flat-square)
![optimizer](https://img.shields.io/badge/-optimizer-742A2A?style=flat-square)
![backward()](https://img.shields.io/badge/-backward()-742A2A?style=flat-square)

```
Dataset → DataLoader → Model → Forward → Loss → Backward → Optimizer → Repeat
```

*A Tensor is a NumPy array that can track gradients and live on a GPU. `Dataset`/`DataLoader` handle feeding data in batches. `.backward()` is autodiff doing, automatically, the exact chain-rule math you wrote by hand in stage 5.*

**Also Learn**

![CUDA](https://img.shields.io/badge/-GPU%2FCUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Training loops](https://img.shields.io/badge/-Training_Loops-C05621?style=flat-square)
![Validation loops](https://img.shields.io/badge/-Validation_Loops-C05621?style=flat-square)
![Checkpoints](https://img.shields.io/badge/-Checkpoints-C05621?style=flat-square)
![LR Scheduling](https://img.shields.io/badge/-LR_Scheduling-C05621?style=flat-square)
![Early stopping](https://img.shields.io/badge/-Early_Stopping-C05621?style=flat-square)
![Transfer learning](https://img.shields.io/badge/-Transfer_Learning-C05621?style=flat-square)

*Transfer learning — starting from a pretrained model instead of random weights — is how almost every real-world project actually gets built; training from scratch is rare outside research.*

---

## 🔵 07 · CNN — Computer Vision

```
Image → Convolution → Activation → Pooling → Convolution → Flatten → Fully Connected → Prediction
```

**Why this stage exists:** feeding raw pixels into a plain ANN would need millions of weights and ignore the fact that nearby pixels are related. Convolution slides a small filter across the image, so the network learns local patterns (edges, then shapes, then objects) with far fewer parameters.

**Concepts**

![Kernels](https://img.shields.io/badge/-Kernels%2FFilters-2C5282?style=flat-square)
![Feature maps](https://img.shields.io/badge/-Feature_Maps-2C5282?style=flat-square)
![Stride](https://img.shields.io/badge/-Stride-2C5282?style=flat-square)
![Padding](https://img.shields.io/badge/-Padding-2C5282?style=flat-square)
![Pooling](https://img.shields.io/badge/-Pooling-2C5282?style=flat-square)
![Channels](https://img.shields.io/badge/-Channels-2C5282?style=flat-square)
![Receptive field](https://img.shields.io/badge/-Receptive_Field-2C5282?style=flat-square)

*Pooling shrinks the feature map, keeping the strongest signals and making the model tolerant to small shifts in the image. Stacking convolutions grows the receptive field, so deeper layers "see" larger regions of the original image.*

**Model Evolution**

![LeNet](https://img.shields.io/badge/-LeNet-90CDF4?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![AlexNet](https://img.shields.io/badge/-AlexNet-63B3ED?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![VGG](https://img.shields.io/badge/-VGG-4299E1?style=flat-square&labelColor=2B6CB0&color=4299E1)
➜
![ResNet](https://img.shields.io/badge/-ResNet-3182CE?style=flat-square&labelColor=2B6CB0&color=3182CE)
➜
![EfficientNet](https://img.shields.io/badge/-EfficientNet-2B6CB0?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![ViT](https://img.shields.io/badge/-Vision_Transformers-2C5282?style=flat-square&labelColor=2B6CB0&color=2C5282)

*ResNet's key idea — "residual/skip connections" — solved the problem of very deep networks failing to train, and that same skip-connection trick is reused inside every Transformer block later.*

### 🛠️ Projects

![Project](https://img.shields.io/badge/BUILD-Image_Classifier-3182CE?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Cat_vs_Dog-3182CE?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Plant_Disease_Detection-3182CE?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Traffic_Sign_Recognition-3182CE?style=flat-square&logo=target&logoColor=white)

---

## 🟦 08 · RNN → LSTM → GRU

```
x₁ → RNN → h₁
x₂ → RNN → h₂  (using previous hidden state)
x₃ → RNN → h₃
x₄ → RNN → output
```

**Why this stage exists:** images are fixed-size, but language, audio, and time series are sequences with order and memory. RNNs process one step at a time, carrying a "hidden state" forward — a running summary of everything seen so far.

![Sequence](https://img.shields.io/badge/-Sequence-2C7A7B?style=flat-square)
![Hidden state](https://img.shields.io/badge/-Hidden_State-2C7A7B?style=flat-square)
![RNN](https://img.shields.io/badge/-RNN-2C7A7B?style=flat-square)
![Vanishing](https://img.shields.io/badge/-Vanishing%2FExploding_Gradient-2C7A7B?style=flat-square)
![LSTM](https://img.shields.io/badge/-LSTM-285E61?style=flat-square)
![GRU](https://img.shields.io/badge/-GRU-285E61?style=flat-square)

*Plain RNNs forget long-range context because gradients shrink to zero across many steps ("vanishing gradient"). LSTM and GRU add gates that decide what to keep, forget, or output — a controllable memory instead of a leaky one.*

> Even LSTM/GRU still process one token at a time, which is slow and still struggles with very long context — that's exactly the gap **Transformers** were built to close.

---

## 🟣 09 · Transformers — The Big One

![Goal](https://img.shields.io/badge/🎯_GOAL-Major_Milestone-6B46C1?style=for-the-badge)

**Why this stage exists:** instead of processing a sequence one token at a time, a Transformer looks at *all* tokens simultaneously and lets each one directly "attend to" every other one. That parallelism is why Transformers train so much faster than RNNs, and why they scale to the huge models behind every modern LLM.

```
Tokens → Token Embeddings → Positional Information → Self-Attention
   → Multi-Head Attention → Feed Forward Network → Add & Norm
   → Transformer Block → Output
```

### Deeply Understand

![Tokenization](https://img.shields.io/badge/-Tokenization-553C9A?style=flat-square)
![Embeddings](https://img.shields.io/badge/-Embeddings-553C9A?style=flat-square)
![Positional](https://img.shields.io/badge/-Positional_Encoding-553C9A?style=flat-square)
![QKV](https://img.shields.io/badge/-Query%2FKey%2FValue-6B46C1?style=flat-square)
![Attention](https://img.shields.io/badge/-Attention-6B46C1?style=flat-square)
![Self-attention](https://img.shields.io/badge/-Self--Attention-6B46C1?style=flat-square)
![Multi-head](https://img.shields.io/badge/-Multi--Head_Attention-6B46C1?style=flat-square)
![FFN](https://img.shields.io/badge/-Feed--Forward_Network-805AD5?style=flat-square)
![Residual](https://img.shields.io/badge/-Residual_Connections-805AD5?style=flat-square)
![LayerNorm](https://img.shields.io/badge/-Layer_Normalization-805AD5?style=flat-square)

*Tokenization breaks text into pieces; embeddings turn those pieces into vectors; positional encoding injects word order back in (since attention alone has no sense of sequence). Query/Key/Value is a lookup mechanism — Query asks "what am I looking for," Key answers "what do I contain," Value is "what I'll give you if you attend to me." Multi-head attention just runs several of these lookups in parallel, each learning a different kind of relationship (grammar, meaning, coreference, etc).*

### The Core Formula

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

*In words: score how relevant every token is to every other token (`QKᵀ`), scale so the numbers stay stable (`/√dₖ`), turn scores into weights that sum to 1 (`softmax`), then blend the Values using those weights. That blended result is what each token "pays attention to."*

> 💡 **Don't memorize the equation. Understand why every symbol is there.**

---

## 🟢 10 · BERT / GPT

| Architecture | Component | Use Case |
|---|---|---|
| ![BERT](https://img.shields.io/badge/BERT-4285F4?style=for-the-badge&logo=google&logoColor=white) | Encoder | Understanding representations |
| ![GPT](https://img.shields.io/badge/GPT-10A37F?style=for-the-badge&logo=openai&logoColor=white) | Decoder | Autoregressive generation |
| ![Transformer](https://img.shields.io/badge/Original_Transformer-1A202C?style=for-the-badge) | Encoder + Decoder | Translation / Seq2Seq |

*BERT's encoder sees the whole sentence at once (both directions), which makes it great at understanding tasks like search or classification but bad at generating new text. GPT's decoder can only see what came before the current token, generating one word at a time — that restriction is exactly what makes it good at writing fluent, coherent text.*

---

## 🟪 11 · LLM Engineering

**Why this stage exists:** an LLM is "just" a giant GPT-style decoder trained on enormous amounts of text. This stage is about how to actually *use* one — how it was trained, how it generates text, and how to specialize a general model for your own task.

**Fundamentals**

![Pretraining](https://img.shields.io/badge/-Pretraining-553C9A?style=flat-square)
![Next-token](https://img.shields.io/badge/-Next--Token_Prediction-553C9A?style=flat-square)
![Context window](https://img.shields.io/badge/-Context_Window-553C9A?style=flat-square)
![Embeddings](https://img.shields.io/badge/-Embeddings-553C9A?style=flat-square)
![Attention](https://img.shields.io/badge/-Attention-553C9A?style=flat-square)
![Inference](https://img.shields.io/badge/-Inference-6B46C1?style=flat-square)
![Sampling](https://img.shields.io/badge/-Sampling-6B46C1?style=flat-square)
![Temperature](https://img.shields.io/badge/-Temperature-6B46C1?style=flat-square)
![Top-k](https://img.shields.io/badge/-Top--k-6B46C1?style=flat-square)
![Top-p](https://img.shields.io/badge/-Top--p-6B46C1?style=flat-square)

*Pretraining = predicting the next token over huge unlabeled text corpora, which is how a model absorbs grammar, facts, and reasoning patterns without needing labels. Temperature/Top-k/Top-p all control how "random" vs "safe" the next word choice is at generation time.*

```
Pretrained Model → Fine-tuning → Specialized Model
```

**Fine-tuning Toolkit**

![LoRA](https://img.shields.io/badge/-LoRA-9F7AEA?style=for-the-badge)
![QLoRA](https://img.shields.io/badge/-QLoRA-805AD5?style=for-the-badge)
![PEFT](https://img.shields.io/badge/-PEFT-6B46C1?style=for-the-badge)
![Instruction Tuning](https://img.shields.io/badge/-Instruction_Tuning-553C9A?style=for-the-badge)

*Full fine-tuning updates every weight in a multi-billion-parameter model — expensive. LoRA/QLoRA/PEFT freeze the original weights and train small "adapter" layers instead, getting 90% of the benefit for a fraction of the compute and memory.*

---

## 🔷 12 · RAG

```
Documents → Chunking → Embeddings → Vector Database
   → Similarity Search → Relevant Context → LLM → Answer
```

**Why this stage exists:** an LLM only knows what was in its training data, and it can't cite your private documents. RAG fixes this by retrieving the most relevant chunks of *your* data at answer time and handing them to the model as context — no retraining required.

### 🛠️ Project: Chat with PDF → Leveled Up

![PDF](https://img.shields.io/badge/1-PDF-BEE3F8?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![Chunk](https://img.shields.io/badge/2-Chunking-90CDF4?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![Embed](https://img.shields.io/badge/3-Embedding-63B3ED?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![VectorDB](https://img.shields.io/badge/4-Vector_DB-4299E1?style=flat-square&labelColor=2B6CB0&color=4299E1)
➜
![Retriever](https://img.shields.io/badge/5-Retriever-3182CE?style=flat-square&labelColor=2B6CB0&color=3182CE)
➜
![Reranker](https://img.shields.io/badge/6-Reranker-2B6CB0?style=flat-square&labelColor=2B6CB0&color=2B6CB0)
➜
![LLM](https://img.shields.io/badge/7-LLM-2C5282?style=flat-square&labelColor=2B6CB0&color=2C5282)
➜
![Answer](https://img.shields.io/badge/8-Answer_%2B_Sources-1A365D?style=flat-square&labelColor=2B6CB0&color=1A365D)

*Chunking splits documents into small passages (embeddings work better on focused text). The vector DB stores each chunk's embedding so a similarity search can find the closest matches to a question. The reranker is a second, more precise pass that re-sorts the retrieved chunks before they're handed to the LLM — this second pass is what separates a toy demo from a production-quality RAG system.*

---

## 🟡 13 · AI Agents

```
LLM → Reasoning → Tool selection → Tool → Observation → LLM → Final answer
```

**Why this stage exists:** an LLM alone can only talk. An agent lets it *act* — search the web, run code, call an API — by looping between "think," "act," and "observe the result" until the task is done.

![Tool calling](https://img.shields.io/badge/-Tool_Calling-B7791F?style=flat-square)
![Function calling](https://img.shields.io/badge/-Function_Calling-B7791F?style=flat-square)
![Agent loops](https://img.shields.io/badge/-Agent_Loops-B7791F?style=flat-square)
![Memory](https://img.shields.io/badge/-Memory-975A16?style=flat-square)
![Planning](https://img.shields.io/badge/-Planning-975A16?style=flat-square)
![Multi-agent](https://img.shields.io/badge/-Multi--Agent_Systems-975A16?style=flat-square)
![Evaluation](https://img.shields.io/badge/-Evaluation-744210?style=flat-square)
![Guardrails](https://img.shields.io/badge/-Guardrails-744210?style=flat-square)

*Guardrails and evaluation matter more here than anywhere else in the roadmap: once a model can take real actions, an unchecked mistake can have real consequences — so testing and limits aren't optional polish, they're part of the system.*

### 🛠️ Projects

![Project](https://img.shields.io/badge/BUILD-Research_Agent-DD6B20?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Coding_Assistant-DD6B20?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Gov_Service_Assistant-DD6B20?style=flat-square&logo=target&logoColor=white)
![Project](https://img.shields.io/badge/BUILD-Data_Analysis_Agent-DD6B20?style=flat-square&logo=target&logoColor=white)

---

## ⚫ Project Ladder

<div align="center">

| Level | Project |
|:---:|---|
| ![beginner](https://img.shields.io/badge/BEGINNER-2ea44f?style=for-the-badge) | House Price Prediction |
| ![beginner](https://img.shields.io/badge/BEGINNER-2ea44f?style=for-the-badge) | Student Score Prediction |
| ![easy](https://img.shields.io/badge/EASY-D4AC0D?style=for-the-badge) | Spam Classifier |
| ![easy](https://img.shields.io/badge/EASY-D4AC0D?style=for-the-badge) | Customer Churn Prediction |
| ![easy](https://img.shields.io/badge/EASY-D4AC0D?style=for-the-badge) | Fraud Detection |
| ![medium](https://img.shields.io/badge/MEDIUM-DD6B20?style=for-the-badge) | Neural Network from Scratch |
| ![medium](https://img.shields.io/badge/MEDIUM-DD6B20?style=for-the-badge) | Image Classifier with CNN |
| ![medium](https://img.shields.io/badge/MEDIUM-DD6B20?style=for-the-badge) | Sentiment Analysis |
| ![hard](https://img.shields.io/badge/HARD-C53030?style=for-the-badge) | LSTM Text Generator |
| ![hard](https://img.shields.io/badge/HARD-C53030?style=for-the-badge) | Transformer from Scratch |
| ![hard](https://img.shields.io/badge/HARD-C53030?style=for-the-badge) | Mini GPT |
| ![hard](https://img.shields.io/badge/HARD-C53030?style=for-the-badge) | RAG Chatbot |
| ![advanced](https://img.shields.io/badge/ADVANCED-702459?style=for-the-badge) | Fine-tuned LLM |
| ![advanced](https://img.shields.io/badge/ADVANCED-702459?style=for-the-badge) | AI Agent |
| ![capstone](https://img.shields.io/badge/★_CAPSTONE-1A202C?style=for-the-badge) | Production AI System |

</div>

*Each rung forces you to reuse everything below it — the CNN project needs stage 1–4's math, the RAG chatbot needs stage 9's attention intuition to even choose an embedding model sensibly. Skipping rungs shows up later as confusion, not efficiency.*

---

## 🧭 Recommended Order (Given Your Current Stack)

Since you're already touching PyTorch, embeddings, softmax, feed-forward networks, optimization, and Transformers:

```
Python → NumPy → Linear Algebra → Calculus → Statistics
   → Machine Learning → Neural Networks → Activation Functions
   → Loss Functions → Forward Propagation → Backpropagation
   → Gradient Descent → SGD + Momentum → Adam / AdamW
   → PyTorch → CNN → RNN → LSTM/GRU → Attention
   → Transformers → BERT/GPT → LLMs → RAG → Fine-tuning
   → AI Agents → Production AI
```

*You can likely move faster through 1–6 as review, but treat stage 9 (Transformers) as the one place worth slowing back down — it's the architecture underneath everything after it.*

---

## ⭐ The One Rule

<div align="center">

![Rule](https://img.shields.io/badge/THE_ONE_RULE-Learn_the_Story,_Not_the_List-1A202C?style=for-the-badge)

</div>

```
Why do we need a neural network?
        ↓
   Why weights?  →  Why bias?  →  Why activation?
        ↓
    Why loss?  →  Why gradients?  →  Why backpropagation?
        ↓
  Why optimization?  →  Why attention?  →  Why Transformers?
        ↓
                  Why LLMs?
```

*Every "why" in that chain has a one-sentence answer, and you now have all of them scattered through this document. If you can recite the chain without looking anything up, you don't just know ML — you understand it.*

<div align="center">

### If you understand that chain, Deep Learning stops being mysterious.

---

![Made](https://img.shields.io/badge/Made_for-The_ML_→_DL_→_LLM_Journey-2ea44f?style=for-the-badge)
![Author](https://img.shields.io/badge/Roadmap_by-Gemeda_Tamiru_Golo-4C51BF?style=for-the-badge)

</div>
