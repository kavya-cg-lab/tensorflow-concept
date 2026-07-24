Here's a structured overview of **common** and **advanced** data processing techniques used in machine learning.

## Common Data Processing Techniques

| Technique              | Purpose                       | Common Methods                  | When to Use                                       |
| ---------------------- | ----------------------------- | ------------------------------- | ------------------------------------------------- |
| Remove Duplicates      | Eliminate repeated records    | `drop_duplicates()`             | Duplicate customer records, repeated transactions |
| Missing Value Handling | Deal with missing data        | `dropna()`, `fillna()`          | Null or NaN values                                |
| Data Type Conversion   | Convert data to correct types | `astype()`, `to_datetime()`     | Numbers stored as text, dates as strings          |
| Text Cleaning          | Standardize text              | `lower()`, `upper()`, `strip()` | Names, addresses, categories                      |
| Remove Invalid Values  | Correct incorrect entries     | `replace()`, filtering          | Negative ages, impossible dates                   |
| Data Integration       | Merge datasets                | `merge()`, SQL JOIN             | Combining customer and sales data                 |
| Data Transformation    | Change data format            | Log transform, square root      | Reduce skewness                                   |
| Feature Selection      | Keep useful features          | Correlation, importance scores  | Improve model performance                         |

---

# Advanced Data Processing Techniques

| Technique                | Purpose                         | Popular Techniques                                | Common Libraries                |
| ------------------------ | ------------------------------- | ------------------------------------------------- | ------------------------------- |
| Missing Value Treatment  | Fill missing data intelligently | Mean, Median, Mode, KNN, MICE                     | Pandas, Scikit-learn            |
| Outlier Detection        | Find unusual observations       | IQR, Z-score, Isolation Forest, DBSCAN            | SciPy, Scikit-learn             |
| Feature Scaling          | Put features on similar scales  | Standardization, Normalization, RobustScaler      | Scikit-learn                    |
| Categorical Encoding     | Convert categories to numbers   | Label Encoding, One-Hot, Target Encoding          | Scikit-learn, Category Encoders |
| Feature Engineering      | Create better features          | Polynomial, Date extraction, Interaction features | Pandas, Feature-engine          |
| Feature Selection        | Remove unnecessary features     | RFE, Lasso, PCA                                   | Scikit-learn                    |
| Data Balancing           | Handle imbalanced classes       | SMOTE, ADASYN, Undersampling                      | imbalanced-learn                |
| Dimensionality Reduction | Reduce feature count            | PCA, t-SNE, UMAP                                  | Scikit-learn                    |

---

# 1. Missing Value Treatment

Missing values are represented as:

```text
NaN
NULL
None
```

Example

| Name | Age | Salary |
| ---- | --- | ------ |
| John | 25  | 50000  |
| Mike | NaN | 45000  |
| Sara | 30  | NaN    |

---

## Technique 1: Delete Rows

```python
df = df.dropna()
```

Use when:

* Only a few rows are missing.
* Removing them will not affect the dataset significantly.

---

## Technique 2: Mean Imputation

Example Ages

```
20
25
30
NaN
35
```

Mean

```
(20+25+30+35)/4 = 27.5
```

Replace NaN with 27.5.

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
```

Best for:

* Continuous numerical data.
* Approximately normally distributed data.

---

## Technique 3: Median Imputation

Example Salary

```
40000
45000
50000
55000
5000000
```

Mean is heavily affected by the extreme value, but the **median** remains close to the typical salary.

```python
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
```

Best for:

* Skewed data.
* Data with outliers.

---

## Technique 4: Mode Imputation

Example

| Gender |
| ------ |
| Male   |
| Female |
| Male   |
| NaN    |

Mode = Male

```python
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
```

Best for:

* Categorical features.

---

## Technique 5: Forward Fill

```
10
12
NaN
NaN
15
```

Becomes

```
10
12
12
12
15
```

```python
df.fillna(method="ffill")
```

Useful for:

* Time-series data.

---

## Technique 6: KNN Imputation

Instead of using one global value (mean/median), the algorithm finds the **k most similar rows** and estimates the missing value from those neighbors.

Useful when:

* Features are correlated.
* Relationships between variables matter.

---

# 2. Outlier Detection

Outliers are observations that are unusually far from the rest of the data.

Example

```
45
48
50
52
49
46
900
```

Here **900** is an outlier.

Outliers can result from:

* Data entry errors.
* Sensor errors.
* Fraud.
* Genuine rare events.

---

## Method 1: IQR (Interquartile Range)

The most common statistical method.

Steps:

1. Find **Q1** (25th percentile).
2. Find **Q3** (75th percentile).
3. Compute:

```
IQR = Q3 - Q1
```

4. Compute bounds:

```
Lower = Q1 - 1.5 × IQR

Upper = Q3 + 1.5 × IQR
```

Values outside these bounds are considered outliers.

Example

```
10
12
13
14
15
16
100
```

100 will be detected as an outlier.

Python

```python
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Salary"] >= lower) &
        (df["Salary"] <= upper)]
```

Advantages:

* Simple.
* No assumption about normal distribution.
* Widely used.

---

## Method 2: Z-score

Formula

```
Z = (x − mean) / standard deviation
```

Interpretation:

* |Z| < 3 → Normal
* |Z| > 3 → Outlier

```python
from scipy import stats

z = stats.zscore(df["Salary"])

df = df[(abs(z) < 3)]
```

Use when:

* Data is approximately normally distributed.

---

## Method 3: Isolation Forest

A machine learning algorithm that isolates unusual observations by randomly partitioning the data. Outliers are isolated in fewer splits than normal observations.

Best for:

* Large datasets.
* Multiple numerical features.
* Complex patterns.

---

# 3. Feature Scaling

Different features can have very different ranges.

Example

| Age | Salary |
| --- | ------ |
| 25  | 40000  |
| 40  | 90000  |

The model may give more importance to **Salary** simply because its values are much larger.

Scaling puts features on comparable scales.

---

## Standardization

Formula

```
z = (x − mean) / standard deviation
```

After transformation:

* Mean ≈ 0
* Standard deviation ≈ 1

Example

Original

```
20
30
40
```

After Standardization

```
-1.22
0
1.22
```

Python

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
```

Good for:

* Logistic Regression
* SVM
* PCA
* Neural Networks
* K-Means

---

## Normalization (Min-Max Scaling)

Formula

```
(x − min) / (max − min)
```

Output range:

* 0 to 1

Example

Original

```
10
20
30
```

Normalized

```
0
0.5
1
```

Python

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
```

Good for:

* Neural Networks
* Image processing
* Algorithms expecting bounded input values.

---

### Standardization vs Normalization

| Standardization                                                         | Normalization                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| Mean = 0                                                                | Range = 0–1                                                   |
| Standard deviation = 1                                                  | Preserves relative positions within a fixed range             |
| Can produce negative values                                             | Values stay between 0 and 1                                   |
| Better when data has outliers (though extreme outliers still affect it) | Sensitive to outliers because min and max determine the scale |

---

# 4. Categorical Encoding

ML models work with numbers, so categorical values must be encoded.

Example

| Color |
| ----- |
| Red   |
| Blue  |
| Green |

---

## Label Encoding

```
Red → 0
Blue → 1
Green → 2
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Color"] = le.fit_transform(df["Color"])
```

Best for:

* Ordinal categories (e.g., Small < Medium < Large).

Avoid for:

* Nominal categories with no natural order, because the assigned numbers may imply a false ranking.

---

## One-Hot Encoding

| Color | Red | Blue | Green |
| ----- | --- | ---- | ----- |
| Red   | 1   | 0    | 0     |
| Blue  | 0   | 1    | 0     |
| Green | 0   | 0    | 1     |

```python
pd.get_dummies(df["Color"])
```

or

```python
from sklearn.preprocessing import OneHotEncoder
```

Best for:

* Nominal categories such as city, color, or country.

---

## Ordinal Encoding

Example

```
Low
Medium
High
```

Encoded as

```
1
2
3
```

Appropriate only when the categories have a meaningful order.

---

## Target Encoding

Example

| City     | Average House Price |
| -------- | ------------------- |
| New York | 500000              |
| Chicago  | 300000              |
| Boston   | 400000              |

Each category is replaced by a statistic (often the mean target value) computed from the training data.

Useful for:

* High-cardinality features (many unique categories), such as ZIP codes or product IDs.

Needs careful validation to avoid **data leakage**.

---

## Summary

| Problem                    | Recommended Technique            |
| -------------------------- | -------------------------------- |
| Few missing values         | Drop rows (`dropna`)             |
| Numerical missing values   | Mean or Median imputation        |
| Categorical missing values | Mode imputation                  |
| Time-series missing values | Forward/Backward fill            |
| Extreme values             | IQR or Isolation Forest          |
| Different feature scales   | Standardization or Normalization |
| Ordered categories         | Label/Ordinal Encoding           |
| Unordered categories       | One-Hot Encoding                 |
| Many unique categories     | Target Encoding                  |

### Industry tips

* **Mean/Median/Mode imputation** are the most common missing-value treatments in production because they are simple and reliable.
* **IQR** is the go-to technique for univariate outlier detection; **Isolation Forest** is preferred when you have many numerical features and more complex anomalies.
* **Standardization** is more commonly used than normalization for classical ML algorithms, while **normalization** is especially common for image data and some deep learning workflows.
* **One-Hot Encoding** is the default choice for nominal categories, while **Ordinal Encoding** should only be used when the categories have a real order.
