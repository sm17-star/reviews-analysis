import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')


nltk.download('stopwords')
nltk.download('wordnet')

def load_data():
    try:
        # Load a sample of 5,000 rows from dataset
        df = pd.read_csv('Reviews.csv', nrows=5000)
    except FileNotFoundError:
        print("Error: 'Reviews.csv' not found. Ensure it is in your project directory.")
        return None

    # Keep relevant text and score columns
    df = df[['Text', 'Score']].dropna()
    print(f"Loaded {len(df)} records successfully.")
    print("\nMissing values check:")
    print(df.isnull().sum())
    
    return df

def clean_text(text):
    # Remove HTML tags (like <br />)
    text = re.sub(r'<.*?>', '', str(text))
    
    # Remove URLs, numbers, and special symbols, then convert to lowercase
    text = re.sub(r'http\S+|[^a-zA-Z\s]', '', text).lower()
    
    # Split the text into individual words (tokens)
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords and lemmatize each remaining word
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join the words back together into a single cleaned sentence
    return " ".join(tokens)

def generate_visualizations(df):
    # 1. Set the size of the canvas for word count distribution
    plt.figure(figsize=(6, 3))
    sns.histplot(df['word_count'], kde=True, bins=25, color='teal')
    plt.title('Amazon Review Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('amazon_word_count.png', bbox_inches='tight')
    plt.close()

    # 2. Generate and Save Top 10 Single Words Bar Chart
    print("Generating Top 10 Single Words chart...")
    cv_single = CountVectorizer(ngram_range=(1, 1), max_features=10)
    single_words_matrix = cv_single.fit_transform(df['clean_text'])

    single_counts = pd.DataFrame(
        single_words_matrix.toarray(), 
        columns=cv_single.get_feature_names_out()
    ).sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=single_counts.values, y=single_counts.index, hue=single_counts.index, legend=False, palette='crest')
    plt.title('Top 10 Most Frequent Words in Amazon Reviews', fontsize=14)
    plt.xlabel('Occurrences', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.savefig('top10_words.png', bbox_inches='tight')
    plt.close()

    print("Success! Top 10 words chart saved as 'top10_words.png'.")


#  WEEK 2 TEXT CLASSIFICATION FUNCTION ---
def run_text_classification(df):
    print("\n--- STEP 4: WEEK 2 TEXT CLASSIFICATION ---")
    
    # Drop neutral reviews (Score = 3) to create a clean Binary Sentiment task (1 = Positive, 0 = Negative)
    df_filtered = df[df['Score'] != 3].copy()
    df_filtered['sentiment'] = df_filtered['Score'].apply(lambda x: 1 if x > 3 else 0)
    
    X = df_filtered['clean_text']
    y = df_filtered['sentiment']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Extraction using TF-IDF
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Model 1: Naive Bayes
    print("\nTraining Naive Bayes Model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    
    print("Naive Bayes Results:")
    print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
    print(classification_report(y_test, nb_pred))
    
    # Model 2: Logistic Regression
    print("\nTraining Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    print(classification_report(y_test, lr_pred))



def run_week3_sentiment_and_ner(df):
    print("\n--- STEP 5: WEEK 3 SENTIMENT ANALYSIS & NER ---")
    
    # 1. Sentiment Analysis using NLTK VADER
    print("Running Sentiment Analysis (VADER)...")
    sia = SentimentIntensityAnalyzer()
    
    sample_df = df.head(5).copy()
    sample_df['vader_scores'] = sample_df['Text'].apply(lambda x: sia.polarity_scores(str(x)))
    sample_df['vader_compound'] = sample_df['vader_scores'].apply(lambda score_dict: score_dict['compound'])
    sample_df['predicted_sentiment'] = sample_df['vader_compound'].apply(
        lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral')
    )
    
    print("\nSentiment Analysis Sample Results:")
    for idx, row in sample_df.iterrows():
        print(f"Rating: {row['Score']} | VADER Compound: {row['vader_compound']:.4f} | Sentiment: {row['predicted_sentiment']}")
        print(f"Text Snippet: {row['Text'][:80]}...\n")

    # 2. Named Entity Recognition (NER) using spaCy
    print("Running Named Entity Recognition (NER) via spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm' in your terminal.")
        return

    print("Named Entity Extraction Sample:")
    for idx, text in enumerate(df['Text'].head(3)):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Review {idx + 1} Entities Found: {entities}")


if __name__ == "__main__":
    # STEP 1: load data
    df = load_data()

    if df is not None:
        # STEP 2: Clean the text
        print("\n--- STEP 2: DATA CLEANING ---")
        print("Cleaning text data (this may take a few seconds)...")
        df['clean_text'] = df['Text'].apply(clean_text)
        
        # Preview before vs after
        print("\n--- BEFORE CLEANING ---")
        print(df['Text'].iloc[0][:200])
        
        print("\n--- AFTER CLEANING ---")
        print(df['clean_text'].iloc[0][:200])

        # STEP 3: Exploratory Data Analysis & Statistics
        print("\n--- STEP 3: EXPLORATORY DATA ANALYSIS ---")
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        print("Statistical Summary of Word Counts:")
        print(df['word_count'].describe())

        # Generate Week 1 Visualizations
        generate_visualizations(df)

        # STEP 4: Run Week 2 Classification Pipeline
        run_text_classification(df)

        run_week3_sentiment_and_ner(df)