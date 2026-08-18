# Step 1: Exploratory Data Analysis & Preprocessing in NLP

## 1. Dataset Overview & Source
* **Dataset:** Amazon Fine Food Reviews (Sourced from Kaggle).
* **Objective:** Explore raw textual data, identify anomalies, missing values, and structural noise to prepare for NLP pipelines.
* **Volume:** 5,000 sampled customer review entries.

## 2. Data Preprocessing & Normalization Pipeline
To transform raw, unstructured customer reviews into clean, machine-readable text, a multi-step preprocessing function was implemented using Python, Regular Expressions (`re`), and NLTK:

* **HTML Tag Removal:** Stripped out structural markup (e.g., `<br />`, `<p>`) commonly found in web-scraped data using regular expressions (`<.*?>`)
* **Noise Reduction & Lowercasing:** Removed URLs, numerical digits, and special punctuation symbols.
* **Tokenization & Stopword Filtering:** Split the continuous text string into individual word tokens and filtered out high-frequency, 
* **Lemmatization:** Applied the `WordNetLemmatizer` to reduce words down to their root dictionary form (e.g., converting "tasting", "tasted", or "tastes" into "taste"), preserving semantic integrity better than aggressive stemming algorithms.


## 3. Exploratory Data Analysis (EDA) & Visualizations
Statistical analysis and visualizations were performed on the 5000-review sample to understand key data characteristics.

### 3.1 Review Length Distribution
We analyzed the distribution of word counts per review after cleaning.
* **Observation:** The distribution is right-skewed, indicating that while the average review length is moderate, a significant number of reviews are quite short (under 50 words), with a long tail of very detailed, lengthy reviews.
* **Visualization:** [Word Count Distribution Chart](amazon_word_count.png)

### 3.3 Top 10 Single Words
To identify the most dominant individual terms across the dataset, we extracted the top 10 single words using unigram frequency analysis.
* **Observation:** High-frequency words highlight core subjects and general sentiment indicators (e.g., product, taste, good).
* **Visualization:** [Top 10 Words Chart](top10_words.png)