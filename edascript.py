import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import  CountVectorizer

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
    
    #  Remove URLs, numbers, and special symbols, then convert to lowercase
    text = re.sub(r'http\S+|[^a-zA-Z\s]', '', text).lower()
    
    #  Split the text into individual words (tokens)
    tokens = text.split()


    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    #  Remove stopwords and lemmatize each remaining word
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join the words back together into a single cleaned sentence
    return " ".join(tokens)

def generate_visualizations(df):

        # Distribution of review scores
        # 1. Set the size of the canvas
    plt.figure(figsize=(6, 3))

    # 2. Draw the histogram with a smooth density line
    sns.histplot(df['word_count'], kde=True, bins=25, color='teal')

    # 3. Add titles and labels
    plt.title('Amazon Review Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    # 4. Save the plot as an image file
    plt.savefig('amazon_word_count.png', bbox_inches='tight')

    # 5. Clear the canvas so memory is freed up for the next plot
    plt.close()



    # 4. Generate and Save Top 10 Single Words Bar Chart
    print("Generating Top 10 Single Words chart...")
    
    # A. Use CountVectorizer for single words (ngram_range=(1, 1))
    cv_single = CountVectorizer(ngram_range=(1, 1), max_features=10)
    single_words_matrix = cv_single.fit_transform(df['clean_text'])

    # B. Sum up the occurrences and sort them
    single_counts = pd.DataFrame(
        single_words_matrix.toarray(), 
        columns=cv_single.get_feature_names_out()
    ).sum().sort_values(ascending=False)

    # C. Set up the Matplotlib canvas
    plt.figure(figsize=(8, 4))

    # D. Draw the Seaborn bar chart
    sns.barplot(x=single_counts.values, y=single_counts.index, hue=single_counts.index, legend =False,palette='crest')

    # E. Add titles and labels
    plt.title('Top 10 Most Frequent Words in Amazon Reviews', fontsize=14)
    plt.xlabel('Occurrences', fontsize=12)
    plt.ylabel('Words', fontsize=12)

    # F. Save the chart as an image file
    plt.savefig('top10_words.png', bbox_inches='tight')
    plt.close()

    print("Success! Top 10 words chart saved as 'top10_words.png'.")
     
if __name__ == "__main__":
    # STEP 1: load data
    df = load_data()

    if df is not None:
        # 2. Clean the text
        print("\n STEP 2: DATA CLEANING ---")
        print("Cleaning text data (this may take a few seconds)...")
        df['clean_text'] = df['Text'].apply(clean_text)
        
        # 3. Preview before vs after
        print("\n--- BEFORE CLEANING ---")
        print(df['Text'].iloc[0][:200])
        
        print("\n--- AFTER CLEANING ---")
        print(df['clean_text'].iloc[0][:200])

        # 4. Exploratory Data Analysis & Statistics
        print("STEP 3: EXPLORATORY DATA ANALYSIS ---")
        
        # Calculate word counts
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        print("Statistical Summary of Word Counts:")
        print(df['word_count'].describe())


        generate_visualizations(df)