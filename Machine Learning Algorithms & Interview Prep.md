# Machine Learning Algorithms & Interview Prep
### For a 2-Years-Experienced Engineer — "Which Algorithm, Why, and How It Works"

At 2 years of experience, interviewers stop asking "define linear regression" and start asking **"you built X — why did you choose algorithm Y over Z?"** This guide is built around that shift: algorithm mechanics + when/why to use them + how real interviews at this level are framed.

---

## PART 1: HOW INTERVIEWS CHANGE AT 2 YEARS OF EXPERIENCE

Fresher interviews test **definitions**. 2-YOE interviews test:
1. **Trade-off reasoning** — "why this algorithm and not another, for this specific data/business constraint?"
2. **Debugging judgment** — "your model's accuracy is 95% in training, 60% in production — what do you check?"
3. **Practical decisions** — "how did you handle class imbalance / missing data / model drift in your actual project?"
4. **System-level thinking** — "how would you scale this to 10M users?" or "how would you retrain this model periodically?"

Keep a mental (or written) list of 2-3 real projects you can discuss in this "why did I choose X" format — interviewers weight this far more than algorithm trivia.

---

## PART 2: CORE ALGORITHMS — HOW THEY WORK (Deep Dive)

### 2.1 Linear Regression
**How it works:** Fits a line/hyperplane `y = w1x1 + w2x2 + ... + b` by minimizing the sum of squared errors between predictions and actual values (typically via gradient descent or the closed-form Normal Equation).
**Use when:** Target is continuous, relationship with features is roughly linear, and you need an interpretable model (e.g., "each additional bedroom adds $20K to price").
**Company-style question:** *"We need to predict delivery time and explain the prediction to operations managers — which algorithm?"*
→ Linear Regression (or a slightly more complex but still interpretable model) — because interpretability is a hard requirement here, not just accuracy.

---

### 2.2 Logistic Regression
**How it works:** Computes a weighted sum of inputs, then passes it through a **sigmoid function** to output a probability (0–1); a threshold (commonly 0.5) converts this to a class label. Trained by minimizing **log loss (cross-entropy)**.
**Use when:** Binary classification, need probability scores (not just labels), need a fast and interpretable baseline.
**Company-style question:** *"We need to predict loan default risk and regulators require us to explain every decision — which model?"*
→ Logistic Regression — coefficients directly show how much each feature (income, credit score) shifts the probability of default, which is essential for regulatory/explainability requirements in finance.

---

### 2.3 Decision Trees
**How it works:** Recursively splits data on the feature/threshold that best separates classes (using metrics like **Gini impurity** or **information gain/entropy**), building a tree of if-else rules.
**Use when:** Need high interpretability (can literally draw the decision path), data has non-linear relationships, mixed data types (numeric + categorical) without much preprocessing.
**Weakness:** Prone to overfitting (a fully grown tree can memorize training data perfectly).
**Company-style question:** *"Business wants to understand exactly why each customer was flagged as high churn risk — which model?"*
→ Decision Tree — you can trace the exact path of splits that led to the prediction, which is exactly what "explain each decision" requires.

---

### 2.4 Random Forest
**How it works:** Trains many decision trees, each on a random subset of data and features (**bagging**), then averages (regression) or votes (classification) their predictions.
**Why it beats a single tree:** Averaging many slightly-different, overfit trees cancels out their individual errors/noise, producing a much more stable, generalizable model — this is the core idea of **ensemble learning**.
**Use when:** Tabular data, need better accuracy than a single tree, don't need perfect interpretability (though feature importance is still available), robust to outliers and missing data.
**Company-style question:** *"Predict equipment failure from sensor data with lots of noisy features — which algorithm?"*
→ Random Forest — handles noisy/irrelevant features well, gives feature importance to identify which sensors matter, and doesn't require heavy preprocessing.

---

### 2.5 Gradient Boosting (XGBoost / LightGBM / CatBoost)
**How it works:** Builds trees **sequentially** — each new tree is trained to correct the *errors (residuals)* of the combined previous trees, using gradient descent on the loss function. Unlike Random Forest (parallel, independent trees), boosting trees are dependent on each other.
**Why it's often the top performer on tabular data:** It directly optimizes for reducing error at each step rather than averaging independent guesses, allowing it to capture more subtle patterns — this is why it wins the majority of Kaggle competitions on structured data.
**Use when:** Tabular/structured data, accuracy is the top priority, you have time to tune hyperparameters (learning rate, tree depth, number of estimators) and can risk more overfitting if not tuned carefully.
**Company-style question:** *"We need the highest possible accuracy predicting customer lifetime value from structured CRM data, and can spend time tuning"*
→ XGBoost/LightGBM — best-in-class for structured/tabular data when raw accuracy matters more than interpretability or training speed.

---

### 2.6 K-Nearest Neighbors (KNN)
**How it works:** Stores all training data ("lazy learner," no real training phase); to predict a new point, finds the "k" closest points (by distance metric, usually Euclidean) and takes a majority vote (classification) or average (regression).
**Use when:** Small-to-medium datasets, decision boundary is irregular/non-linear, simplicity is valued over speed.
**Weakness:** Slow at prediction time on large datasets (must compare to every point); sensitive to feature scale (must standardize features first) and irrelevant features.
**Company-style question:** *"Recommend similar products based on customer purchase history, small catalog"*
→ KNN — naturally suited to "find similar items" problems since it's fundamentally a similarity-based method.

---

### 2.7 Support Vector Machines (SVM)
**How it works:** Finds the hyperplane that separates classes with the **maximum margin** (largest distance to the nearest points of each class, called "support vectors"). For non-linear data, uses the **kernel trick** (e.g., RBF kernel) to implicitly map data into higher dimensions where it becomes linearly separable.
**Use when:** Clear margin of separation exists, high-dimensional data (e.g., text classification with many features), medium-sized datasets (doesn't scale as well to huge datasets as tree-based methods).
**Company-style question:** *"Classify documents into categories based on thousands of word-frequency features, moderate dataset size"*
→ SVM — performs well in high-dimensional, sparse feature spaces like text (TF-IDF vectors).

---

### 2.8 Naive Bayes
**How it works:** Applies Bayes' theorem, assuming all features are **independent** given the class (a "naive" assumption that's often wrong but works well in practice). Computes the probability of each class given the input features and picks the highest.
**Use when:** Text classification (spam detection, sentiment analysis), need a fast baseline, high-dimensional sparse data.
**Company-style question:** *"Build a spam filter that needs to retrain quickly on new data daily"*
→ Naive Bayes — extremely fast to train (just counts/probabilities), works surprisingly well for text despite its simplifying assumption.

---

### 2.9 K-Means Clustering
**How it works:** Randomly initializes "k" cluster centers, assigns each point to its nearest center, then recomputes centers as the mean of assigned points — repeats until convergence.
**Use when:** Need to group unlabeled data (customer segmentation, market basket grouping), you have a reasonable estimate of "k" (or use the **elbow method** to find it).
**Company-style question:** *"Segment our customer base into groups for targeted marketing, no labels available"*
→ K-Means — the standard first approach for unsupervised customer segmentation.

---

### 2.10 PCA (Principal Component Analysis)
**How it works:** Finds new axes (principal components) that capture the maximum variance in the data, allowing you to reduce dimensionality while retaining most of the information.
**Use when:** Too many features (curse of dimensionality), need to speed up training, want to visualize high-dimensional data in 2D/3D, or need to remove multicollinearity.
**Company-style question:** *"We have 500 correlated sensor features and training is too slow"*
→ PCA — reduces feature count while preserving most variance, directly addressing both speed and multicollinearity.

---

### 2.11 Neural Networks (MLP) / CNN / RNN-LSTM / Transformers — quick recap
(Detailed in the TensorFlow guide — summarized here for algorithm-selection purposes.)

| Data type | Best-suited architecture | Why |
|-----------|---------------------------|-----|
| Tabular/structured | Gradient Boosting (usually) or MLP | Trees handle structured data with less tuning; MLPs need more data to beat trees here |
| Images | CNN | Convolution exploits spatial locality and translation invariance |
| Sequential/text (older) | RNN/LSTM | Captures order/temporal dependency |
| Text/NLP (modern) | Transformer (BERT, GPT-style) | Captures long-range dependencies in parallel, outperforms RNNs on most NLP tasks today |
| Time series forecasting | ARIMA/Prophet (classical) or LSTM/Transformer (complex patterns) | Classical methods work well for simpler seasonal patterns; deep learning helps with complex, multivariate patterns and large data |

---

## PART 3: "WHICH ALGORITHM FOR THIS PROJECT" — SCENARIO CHEAT SHEET

This is the exact style of question asked at 2 YOE — "given this business problem, what would you pick and why?"

| Project / Scenario | Best-suited algorithm(s) | Why |
|---|---|---|
| Credit scoring / loan default (needs explainability, regulated industry) | Logistic Regression, Decision Tree | Interpretable, coefficients/rules can be explained to regulators/auditors |
| Fraud detection (imbalanced, need high recall) | Random Forest / XGBoost + class weighting or SMOTE | Handles imbalance well with tuning, strong accuracy on tabular data |
| Customer churn prediction | XGBoost / Random Forest | Best accuracy on structured CRM data with mixed features |
| Product recommendation (small catalog) | KNN / Collaborative Filtering | Similarity-based approach fits "similar user/item" logic naturally |
| Product recommendation (large scale, e.g., Netflix/Amazon) | Matrix Factorization, or Deep Learning-based recommenders (e.g., neural collaborative filtering) | Classical KNN doesn't scale; need embeddings for millions of users/items |
| Image classification (e.g., defect detection in manufacturing) | CNN (often via transfer learning) | Convolution captures spatial patterns in images; transfer learning saves data/compute |
| Spam / sentiment detection on text | Naive Bayes (fast baseline) or Logistic Regression/SVM on TF-IDF, or a Transformer for higher accuracy | Naive Bayes/SVM are fast, strong baselines; Transformers used when accuracy matters more than latency/cost |
| Customer segmentation, no labels | K-Means clustering | Standard unsupervised grouping approach |
| Predicting a continuous value (price, demand) with interpretability | Linear Regression / Gradient Boosting Regressor | Linear for simple/interpretable cases, boosting for higher accuracy at some interpretability cost |
| Time series demand forecasting | Prophet/ARIMA (simpler seasonality) or LSTM (complex multivariate patterns) | Classical models are simpler and effective for standard seasonal patterns; DL needed for complex multivariate dependencies |
| Anomaly detection (e.g., server monitoring) | Isolation Forest, One-Class SVM, or Autoencoder (deep learning) | Isolation Forest is fast and effective for tabular anomaly detection; autoencoders help with more complex/high-dimensional anomaly patterns |
| Reducing features / speeding up training | PCA | Removes redundant/correlated features while retaining variance |
| Very large dataset, need fast training | Linear models, or LightGBM (faster than XGBoost on large data) | Simpler models scale better; LightGBM is optimized for speed/memory on large data |

**How to answer this type of question in an interview (structure):**
1. State the algorithm(s) you'd try first.
2. Justify based on the **specific constraint mentioned** (interpretability, imbalance, scale, latency).
3. Mention what you'd check/compare against (e.g., "I'd start with logistic regression as a baseline, then try XGBoost to see if the accuracy gain is worth the interpretability trade-off").
4. Mention evaluation metric appropriate to the business problem (e.g., recall for fraud, RMSE for price prediction).

This 4-step structure is what separates a "book answer" from an experienced-engineer answer.

---

## PART 4: MOST-ASKED QUESTIONS FOR A 2-YEARS-EXPERIENCED ML/DATA ENGINEER

### 4.1 Project & Decision-Making Questions
1. **"Walk me through a project where you built a model end-to-end."**
   Structure your answer: business problem → data → preprocessing choices → algorithm(s) tried → evaluation metric chosen (and why) → final model → how it was deployed/monitored.

2. **"Why did you choose [algorithm] over [alternative] in that project?"**
   Always tie back to a concrete constraint: data size, interpretability need, latency requirement, or accuracy gain measured.

3. **"Your model had 95% training accuracy but only 65% in production — what do you check?"**
   Check for: (a) overfitting (train/val gap), (b) data drift (production data distribution differs from training data), (c) data leakage during training (accidentally including future/target-correlated info), (d) train/test split issues (e.g., time-based leakage in time series data).

4. **"How did you handle class imbalance in your dataset?"**
   Techniques: class weighting in the loss function, oversampling minority class (SMOTE), undersampling majority class, choosing recall/F1/AUC instead of accuracy as the evaluation metric.

5. **"How do you decide which evaluation metric to use for a business problem?"**
   Tie metric to business cost: high cost of false negatives (e.g., disease, fraud) → prioritize recall; high cost of false positives (e.g., blocking legitimate transactions) → prioritize precision; balanced needs → F1 or AUC.

6. **"How do you handle missing data, and how did you decide the approach?"**
   Depends on missingness pattern and % missing: drop if minimal and random; impute (mean/median for numeric, mode for categorical) if moderate; use a model-based imputation or a "missing" indicator flag if missingness itself carries information (e.g., a skipped survey question can be meaningful).

7. **"How do you prevent data leakage?"**
   Ensure all preprocessing (scaling, imputation, feature selection) is fit **only on training data**, then applied to test/validation data — never fit preprocessing on the full dataset before splitting. For time series, always split by time, never randomly.

8. **"How would you monitor a model in production over time?"**
   Track prediction distribution drift, input feature drift, live accuracy/business metrics (if ground truth becomes available later), and set up alerts/retraining triggers when performance degrades.

### 4.2 Technical/Conceptual Questions Common at This Level
9. **"Explain bias-variance tradeoff with an example from your work."**
10. **"What's the difference between bagging and boosting?"**
    Bagging (Random Forest) trains trees independently/in parallel on random subsets and averages them to reduce variance. Boosting (XGBoost) trains trees sequentially, each correcting the previous one's errors, reducing bias but with more risk of overfitting if not regularized.
11. **"How does regularization (L1 vs L2) work, and when would you use each?"**
    L1 (Lasso) can shrink some coefficients to exactly zero — useful for feature selection. L2 (Ridge) shrinks coefficients smoothly toward zero without eliminating them — useful when all features contribute somewhat and you want to reduce their combined influence to prevent overfitting.
12. **"How do you choose hyperparameters?"**
    Grid search / random search / Bayesian optimization with cross-validation, guided by a validation metric aligned with the business goal.
13. **"What is cross-validation, and why use it over a single train/test split?"**
    Splits data into multiple folds, training/testing on different combinations, to get a more reliable estimate of model performance and reduce the risk that a single lucky/unlucky split skews your evaluation.
14. **"Explain the difference between parametric and non-parametric models."**
    Parametric models (linear/logistic regression) assume a fixed functional form with a set number of parameters regardless of data size. Non-parametric models (KNN, decision trees) can grow in complexity with more data and don't assume a fixed form.
15. **"How would you explain a complex model's prediction to a non-technical stakeholder?"**
    Mention tools like feature importance (tree-based models), SHAP/LIME values (model-agnostic explainability), or partial dependence plots — showing which features drove a specific prediction in plain language.

### 4.3 System-Level / Scaling Questions (start appearing at 2 YOE)
16. **"How would you scale model training if your dataset grew from 1GB to 1TB?"**
    Consider: distributed training frameworks (Spark MLlib, distributed XGBoost/LightGBM), sampling strategies for prototyping, batch/mini-batch processing, cloud-based distributed compute.
17. **"How would you deploy this model, and how would you retrain it periodically?"**
    Package the model (e.g., via a REST API, TensorFlow Serving, or a batch pipeline), set up a scheduled retraining pipeline as new data arrives, and version models so you can roll back if a new version underperforms.
18. **"What would you do if two features were highly correlated?"**
    For linear models, multicollinearity can distort coefficient interpretation — consider removing one, combining them, or using PCA/regularization (Ridge handles correlated features better than plain linear regression).

---

## PART 5: HOW TO PREPARE — PRACTICAL STEPS

1. Pick 2–3 real projects from your experience and rehearse explaining them using the structure in Q1 above.
2. For each project, be ready to answer: "why this algorithm," "what metric did you optimize for and why," and "what would you improve if you redid it."
3. Practice the **scenario cheat sheet** (Part 3) out loud — cover the algorithm name and try to reconstruct the reasoning from the scenario alone.
4. Be ready to discuss **one failure or debugging story** (e.g., a model that didn't generalize) — this is one of the most common "senior-track" style questions even at 2 years, since it tests real experience over textbook knowledge.
5. Know your metrics cold: precision, recall, F1, ROC-AUC, RMSE, R² — and the business scenario each maps to.

---

*Golden rule for this stage of interview: always answer "which algorithm" questions with reasoning tied to the constraint (data size, interpretability, imbalance, latency, accuracy needs) — never just name-drop an algorithm without justification.*
