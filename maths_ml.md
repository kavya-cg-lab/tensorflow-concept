# Math for AI/ML Interviews: Your Guide

A quick map of the math you need, **what it is, where it's used, and what interviewers ask**. Not a textbook. Use it to know *what* to know and *why*.

**Priority legend:** 🔴 must know cold · 🟡 should know · 🟢 good to know

---

## 0. The Big Picture

| Math Area | Role in ML (one line) |
|---|---|
| Linear Algebra | How data is **represented** and transformed (vectors, matrices, tensors) |
| Calculus | How models **learn** (gradients, backprop) |
| Probability | How we model **uncertainty** and generate data |
| Statistics | How we **estimate, test and evaluate** from data |
| Optimization | How we **find the best parameters** |
| Information Theory | How we **measure information / loss** (entropy, KL, cross-entropy) |

**Mental model:** Data = vectors → Model = function (matrix ops + nonlinearities) → Loss = measure of error (probability/information theory) → Learning = gradient-based optimization (calculus).

---

## 1. Linear Algebra 🔴

### 1.1 Vectors & Matrices
- **Vector**: a list of numbers = a point/direction in space. A data sample (features) is a vector.
- **Matrix**: a table of numbers = a linear transformation, or a dataset (rows = samples, columns = features).
- **Tensor**: n-dimensional array (images = 3D, batches of images = 4D).
- **Used in:** every model input/weight. Neural net layer: `y = Wx + b`.

### 1.2 Key Operations
| Operation | Meaning | Where used |
|---|---|---|
| Dot product `a·b` | Similarity (large if aligned) | Attention scores, cosine similarity, neuron computation |
| Matrix multiplication | Compose transformations | Forward pass of every NN layer |
| Transpose | Flip rows/columns | Gradients, covariance `XᵀX` |
| Inverse | Undo a transformation | Linear regression closed form `(XᵀX)⁻¹Xᵀy` |
| Element-wise (Hadamard) product | Multiply matching entries | Gates in LSTM/GRU |

### 1.3 Norms 🔴
- **L1 norm** = sum of |xᵢ| → promotes **sparsity** (Lasso).
- **L2 norm** = √(sum of xᵢ²) → penalizes large weights (Ridge, weight decay).
- **Used in:** regularization, distance metrics (Euclidean = L2), gradient clipping.

### 1.4 Eigenvalues & Eigenvectors 🔴
- `Av = λv`: direction `v` that a matrix only **stretches** (by λ), never rotates.
- **Used in:** **PCA** (eigenvectors of covariance matrix = principal directions; eigenvalues = variance explained), spectral clustering, PageRank, stability analysis of RNNs.

### 1.5 SVD (Singular Value Decomposition) 🔴
- `A = UΣVᵀ`: any matrix factors into rotation → scaling → rotation.
- **Used in:** dimensionality reduction, **LoRA / low-rank approximation**, recommender systems (matrix factorization), noise removal, computing pseudo-inverse, LSA in NLP.

### 1.6 Other Important Ideas
- **Rank**: number of independent dimensions. Low-rank = compressible (LoRA).
- **Orthogonality**: perpendicular vectors, zero dot product (independent features).
- **Positive (semi)definite matrices**: covariance matrices are PSD; guarantees convexity (Hessian).
- **Projection**: shadow of a vector onto another; linear regression = projecting `y` onto the column space of `X`.
- **Determinant**: scaling factor of volume; zero → matrix not invertible. 🟢
- **Cosine similarity** = `a·b / (‖a‖‖b‖)`: used in embeddings, semantic search, RAG. 🔴

**Common Interview Questions**
- Why does PCA use eigenvectors of the covariance matrix?
- What's the difference between eigen-decomposition and SVD?
- What happens if `XᵀX` is not invertible? (multicollinearity → use regularization / pseudo-inverse)
- Why is matrix multiplication central to GPUs/deep learning?

---

## 2. Calculus 🔴

### 2.1 Derivatives & Gradients
- **Derivative**: rate of change of a function.
- **Gradient ∇f**: vector of partial derivatives; points in direction of **steepest increase**. We move *opposite* to it to minimize loss.
- **Used in:** gradient descent, every training loop.

### 2.2 Chain Rule 🔴 (the single most important idea)
- `d/dx f(g(x)) = f′(g(x))·g′(x)`
- **Backpropagation = chain rule applied layer by layer** from output to input.

### 2.3 Jacobian & Hessian 🟡
- **Jacobian**: matrix of first derivatives of a vector function (used in backprop through layers, normalizing flows).
- **Hessian**: matrix of second derivatives = curvature. Used in Newton's method, to analyze minima/saddle points, and in second-order optimizers.

### 2.4 Useful Derivatives to Memorize
| Function | Derivative | Why it matters |
|---|---|---|
| Sigmoid σ(x) | σ(x)(1−σ(x)) | Vanishing gradients |
| tanh | 1 − tanh²(x) | RNNs |
| ReLU | 1 if x>0 else 0 | Default activation; "dying ReLU" |
| Softmax + cross-entropy | `ŷ − y` | Elegant gradient for classification |
| MSE `(y−ŷ)²` | `−2(y−ŷ)` | Regression |

### 2.5 Related Concepts
- **Vanishing / exploding gradients**: repeated multiplication of small/large derivatives across layers → fixes: ReLU, residual connections, normalization, gradient clipping, LSTM.
- **Automatic differentiation**: how PyTorch/TensorFlow compute gradients (computational graph). 🟡
- **Taylor series**: local approximation of functions; basis for optimization methods. 🟢
- **Integration**: needed for probability (area under pdf), expectations. 🟡

**Common Interview Questions**
- Derive the gradient for linear regression / logistic regression.
- Explain backprop in simple terms.
- Why do sigmoid/tanh cause vanishing gradients?
- What is a saddle point and why is it common in high dimensions?

---

## 3. Probability 🔴

### 3.1 Fundamentals
- **Random variable**: numeric outcome of a random process (discrete or continuous).
- **PMF / PDF / CDF**: probability of value / density / cumulative probability.
- **Expectation E[X]**: average value. **Variance Var(X)** = E[(X−μ)²]: spread.
- **Covariance / Correlation**: how two variables move together.

### 3.2 Conditional Probability & Bayes' Theorem 🔴
- `P(A|B) = P(B|A)·P(A) / P(B)`
- Posterior ∝ Likelihood × Prior.
- **Used in:** Naive Bayes, Bayesian inference, spam filters, medical-test style questions, Bayesian optimization, probabilistic reasoning.
- **Independence**: `P(A,B) = P(A)P(B)`. Naive Bayes *assumes* feature independence given the class.

### 3.3 Key Distributions 🔴
| Distribution | Type | Where it appears |
|---|---|---|
| **Bernoulli** | Binary outcome | Binary classification (logistic regression output) |
| **Binomial** | # successes in n trials | A/B test conversions |
| **Categorical / Multinomial** | One of K classes | Softmax outputs, next-token prediction |
| **Gaussian (Normal)** | Bell curve | Noise assumption, weight init, VAEs, Gaussian processes, MSE loss ↔ Gaussian likelihood |
| **Poisson** | Counts over time | Event counts, rare events |
| **Uniform** | Equal likelihood | Random init, sampling |
| **Exponential** | Time between events | Survival/waiting time 🟢 |
| **Beta / Dirichlet** | Distribution over probabilities | Bayesian priors, topic models (LDA) 🟡 |

### 3.4 Important Theorems
- **Law of Large Numbers**: sample mean → true mean as n grows (why more data helps).
- **Central Limit Theorem (CLT)** 🔴: averages of many samples ≈ Gaussian, regardless of original distribution. Foundation of confidence intervals & hypothesis tests.

### 3.5 Estimation: MLE & MAP 🔴
- **MLE (Maximum Likelihood)**: pick parameters that make the observed data most probable. 
  - MSE loss = MLE under Gaussian noise.
  - Cross-entropy loss = MLE under Bernoulli/Categorical.
- **MAP (Maximum A Posteriori)**: MLE + a prior. 
  - Gaussian prior → **L2 regularization**; Laplace prior → **L1 regularization**.

### 3.6 Probabilistic Models & Sampling 🟡
- Markov chains, HMMs, Bayesian networks, Gaussian Mixture Models (EM algorithm), MCMC, Monte Carlo sampling, reparameterization trick (VAEs), temperature/top-k/top-p sampling in LLMs.

**Common Interview Questions**
- Explain Bayes' theorem with an example (e.g., disease test).
- Why is MSE tied to Gaussian noise and cross-entropy to Bernoulli?
- What is the difference between MLE and MAP? How do priors relate to regularization?
- What does CLT say and why does it matter?

---

## 4. Statistics 🔴

### 4.1 Descriptive
- Mean, median, mode, variance, standard deviation, percentiles, skewness, outliers.
- Use **median** when outliers exist; **standardize** features using mean & std.

### 4.2 Inference
- **Sampling & bias**: sampling bias, selection bias, survivorship bias.
- **Confidence interval**: range likely to contain the true parameter.
- **Hypothesis testing** 🔴: null vs alternative; **p-value** = probability of seeing data this extreme if null is true (NOT the probability the null is true). Type I error (false positive, α), Type II error (false negative, β), **power** = 1−β.
- **Common tests:** t-test, chi-square, ANOVA, z-test. **Used in:** A/B testing, feature selection, model comparison.

### 4.3 Bias–Variance Tradeoff 🔴
- **Bias**: error from wrong/simple assumptions → underfitting.
- **Variance**: sensitivity to training data → overfitting.
- Total error = Bias² + Variance + Irreducible noise.
- Fixes: more data, regularization, ensembling (bagging reduces variance, boosting reduces bias), simpler/more complex model, cross-validation.

### 4.4 Regression & Model Evaluation 🔴
- **Linear/Logistic regression assumptions**: linearity, independence, homoscedasticity, no multicollinearity.
- **Metrics (regression):** MSE, RMSE, MAE, R².
- **Metrics (classification):** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, log-loss.
  - Imbalanced data → use precision/recall/F1/PR-AUC, not accuracy.
- **Cross-validation**: k-fold, stratified, time-series split; prevents overfitting to one split.
- **Data leakage**: information from test/future leaks into training; classic pitfall.

### 4.5 Other Useful Topics 🟡
- Correlation ≠ causation; Simpson's paradox; multicollinearity (VIF); bootstrap resampling; curse of dimensionality; normalization vs standardization.

**Common Interview Questions**
- Explain p-value to a non-technical person.
- How would you run and interpret an A/B test?
- Precision vs recall tradeoff, and when to favor each?
- How do you detect/handle overfitting?

---

## 5. Optimization 🔴

### 5.1 Core Idea
Find parameters θ that minimize a **loss function** L(θ).

### 5.2 Gradient Descent Family 🔴
`θ ← θ − η·∇L(θ)` (η = learning rate)

| Variant | Idea | Notes |
|---|---|---|
| **Batch GD** | Use all data per step | Stable, slow |
| **Stochastic GD (SGD)** | One sample per step | Noisy, fast, can escape local minima |
| **Mini-batch GD** | Small batches | Standard in practice |
| **Momentum** | Accumulate past gradients | Smooths oscillations, speeds up |
| **RMSProp** | Scale by running avg of squared gradients | Adaptive learning rates |
| **Adam / AdamW** 🔴 | Momentum + RMSProp (+ decoupled weight decay) | Default for deep learning & transformers |

### 5.3 Learning Rate 🔴
- Too high → diverges/oscillates; too low → slow/stuck.
- **Schedules**: step decay, cosine annealing, warmup (important for transformers).

### 5.4 Convexity 🔴
- **Convex** function: any local minimum is global (linear/logistic regression, SVM). 
- **Non-convex** (neural nets): many local minima & saddle points; in practice, good minima are found by SGD-style methods.

### 5.5 Regularization (as optimization/constraints) 🔴
- **L2 (Ridge / weight decay)**: shrinks weights smoothly.
- **L1 (Lasso)**: sparse weights, feature selection.
- **Dropout, early stopping, data augmentation, batch/layer normalization**.

### 5.6 Other Concepts 🟡
- **Lagrange multipliers / constrained optimization**: used in SVM derivation.
- **Newton's method / second-order**: uses Hessian, faster convergence but expensive.
- **Convergence, local vs global minima, plateaus, saddle points**.
- **Gradient clipping**, **weight initialization** (Xavier/He) to keep gradient scale stable.

**Common Interview Questions**
- Why Adam over SGD (and when SGD generalizes better)?
- What happens if the learning rate is too large?
- L1 vs L2: differences and when to use each?
- Why is non-convex optimization still workable for deep nets?

---

## 6. Information Theory 🟡→🔴

| Concept | Formula (idea) | Where used |
|---|---|---|
| **Entropy H(p)** | `−Σ p log p` (uncertainty) | Decision trees (information gain), measuring uncertainty |
| **Cross-entropy H(p,q)** | `−Σ p log q` | **Loss for classification & language models** |
| **KL divergence** | `Σ p log(p/q)` (how q differs from p; asymmetric, ≥ 0) | VAEs, knowledge distillation, RLHF/PPO constraints, t-SNE |
| **Mutual information** | Shared information between variables | Feature selection |
| **Perplexity** | `exp(cross-entropy)` | Evaluating language models |
| **Gini impurity** | `1 − Σ p²` | Decision tree splits (alternative to entropy) |

**Key link:** Minimizing cross-entropy = minimizing KL divergence to the true distribution = maximizing likelihood.

**Common Interview Questions**
- Why cross-entropy instead of MSE for classification?
- Is KL divergence a distance metric? (No: not symmetric, no triangle inequality.)
- How are entropy and information gain used in decision trees?

---

## 7. Where Each Math Shows Up: Algorithm Map 🔴

| Algorithm / Topic | Math you need |
|---|---|
| Linear Regression | Linear algebra (normal equation), calculus (gradient), MLE (Gaussian) |
| Logistic Regression | Sigmoid, log-likelihood, cross-entropy, gradient descent |
| SVM | Geometry/dot products, Lagrange multipliers, kernels |
| Decision Trees / Random Forest | Entropy, Gini, information gain, bias–variance (bagging) |
| Gradient Boosting (XGBoost) | Gradients/Hessians, additive models, regularization |
| k-NN / k-Means | Distance metrics (L1/L2/cosine), centroids, convergence |
| PCA | Covariance, eigenvectors/eigenvalues, SVD |
| Naive Bayes | Bayes' theorem, conditional independence |
| Gaussian Mixture / EM | Gaussians, MLE, latent variables |
| Neural Networks | Matrix mult, chain rule (backprop), activation derivatives, optimization |
| CNN | Convolution (sliding dot product), pooling, linear algebra |
| RNN / LSTM | Chain rule through time, vanishing gradients, gates |
| **Transformers / LLMs** | Dot-product attention, softmax, matrix mult, layer norm, cross-entropy, positional encoding (sin/cos) |
| VAE / GANs / Diffusion | KL divergence, Gaussians, sampling, reparameterization, min-max game |
| Reinforcement Learning | Probability, Markov Decision Processes, expectation, Bellman equation, policy gradients |
| Recommender Systems | Matrix factorization (SVD), cosine similarity |
| Embeddings / RAG | Vectors, cosine similarity, dimensionality |

### Transformer Attention in One Line 🔴
`Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- `QKᵀ` = dot-product similarity; `√d_k` keeps values stable (prevents softmax saturation); softmax → probabilities; multiply by `V` → weighted sum.

---

## 8. Other Useful Topics 🟢
- **Discrete math / combinatorics**: permutations, combinations (counting, probability questions).
- **Graph theory**: GNNs, knowledge graphs, PageRank.
- **Numerical stability**: log-sum-exp trick, avoiding overflow/underflow, floating-point precision (FP16/BF16).
- **Distance metrics**: Euclidean, Manhattan, cosine, Mahalanobis, Hamming.
- **Curse of dimensionality**: in high dimensions distances become less meaningful; data gets sparse.
- **Kernels**: implicit mapping to high-dimensional space (SVM, Gaussian processes).

---

## 9. Quick Formula Cheat Sheet

```
Dot product:        a·b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ
L2 norm:            ‖x‖₂ = √(Σ xᵢ²)
Gradient descent:   θ ← θ − η ∇L(θ)
Sigmoid:            σ(z) = 1 / (1 + e^(−z))
Softmax:            softmax(zᵢ) = e^(zᵢ) / Σ e^(zⱼ)
MSE:                (1/n) Σ (yᵢ − ŷᵢ)²
Cross-entropy:      −Σ yᵢ log(ŷᵢ)
Bayes:              P(A|B) = P(B|A)P(A) / P(B)
Variance:           E[X²] − (E[X])²
Entropy:            −Σ p log p
KL divergence:      Σ p log(p/q)
Precision:          TP / (TP + FP)
Recall:             TP / (TP + FN)
F1:                 2·P·R / (P + R)
Bias–Variance:      Error = Bias² + Variance + Noise
PCA:                eigenvectors of covariance matrix
Linear reg (closed): w = (XᵀX)⁻¹ Xᵀ y
```

---

## 10. How to Prepare (Study Roadmap)

1. **Week 1: Linear Algebra:** vectors, matrices, dot product, norms, eigen, SVD, PCA. *(Resource: 3Blue1Brown "Essence of Linear Algebra")*
2. **Week 2: Calculus + Optimization:** derivatives, chain rule, gradient descent, Adam; **derive backprop by hand for a 2-layer net.**
3. **Week 3: Probability + Statistics:** Bayes, distributions, CLT, MLE/MAP, hypothesis testing, bias–variance, metrics.
4. **Week 4: Information theory + connect everything:** cross-entropy, KL, then re-derive logistic regression end to end and walk through attention.
5. **Always:** practice explaining each concept in **plain English in 30–60 seconds**, then give an example of where it's used.

### Interview Tips
- Interviewers care about **intuition + application** more than heavy proofs.
- For every concept, be ready to answer: **What is it? Why does it matter in ML? Where have I seen it used?**
- Be able to **derive by hand**: linear regression gradient, logistic regression gradient, softmax+cross-entropy gradient, PCA objective.
- Know the **"why" behind defaults**: why ReLU, why Adam, why cross-entropy, why normalize inputs, why `√d_k` in attention.
- Link theory to **practical issues**: overfitting, vanishing gradients, class imbalance, data leakage.

---

## 11. Self-Check Questions

- [ ] Explain what a gradient is and why we subtract it.
- [ ] Why does backprop work? (chain rule)
- [ ] What do eigenvalues tell us in PCA?
- [ ] Difference between L1 and L2 regularization, and the Bayesian view of each?
- [ ] Why cross-entropy for classification? How is it related to MLE and KL?
- [ ] Explain bias–variance with an example.
- [ ] What is p-value, Type I/II error, power?
- [ ] Precision vs recall vs F1, and when to use which?
- [ ] Why does softmax + cross-entropy have a clean gradient (ŷ − y)?
- [ ] What is the CLT and why is it useful?
- [ ] Why does attention divide by √d_k?
- [ ] How does SVD relate to LoRA?

Good luck! 🚀
