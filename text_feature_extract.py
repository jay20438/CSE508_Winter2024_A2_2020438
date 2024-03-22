import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from bs4 import BeautifulSoup

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
df = pd.read_csv('A2_Data.csv')

df = df.dropna(subset=['Review Text'])

def preprocess_text(text):

    if pd.isnull(text):
        return
    
    # Lowercasing the text
    text = text.lower()
    
    # Removing HTML tags from the data
    text = BeautifulSoup(text, "html.parser").get_text()
    tokenizer = RegexpTokenizer(r'\w+')
    # Tokenization
    tokens = tokenizer.tokenize(text)
    
    # Removing punctuations
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Removing the leading and trailing spaces
    tokens = [token for token in tokens if token.strip()]   
    
    # Tokens are stored as a list
    processed_text = tokens
    
    return processed_text

# Apply preprocessing to 'Review Text' column and create 'Preprocessed Review' column
df['Preprocessed Review'] = df['Review Text'].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
df.to_csv("preprocessed_text_data.csv", index=False)

# Checking the first few rows
print(df.head())
