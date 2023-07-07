import string
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
# ML Models
from sklearn.naive_bayes import MultinomialNB
# Scikit Learn packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import Pipeline
from utils.wordprocess import WordProcess


class NaiveBayesClassifierUtil():

    @staticmethod
    def preporcessfile(df):
        df['Text2'] = df['Text'].replace('\n',' ')
        df['Text2'] = df['Text2'].replace('\r',' ')
        
        # Remove punctuation signs and lowercase all
        df['Text2'] = df['Text2'].str.lower()
        df['Text2'] = df['Text2'].str.translate(str.maketrans('', '', string.punctuation))

        wp = WordProcess()
        df['Text2'] = df['Text2'].apply(wp.lemmatize_text)
        return df

    @staticmethod
    def train_data(df):
        x_train, x_test, y_train, y_test = train_test_split(df['Text2'], df['Class'], test_size=0.2, random_state=9)
        vector = TfidfVectorizer(stop_words='english', ngram_range = (1,2), min_df = 3, max_df = 1., max_features = 10000)
        x = pd.concat([x_train, x_test])
        y = pd.concat([y_train, y_test])
        pipeline = Pipeline([('vectorize', vector), ('model', MultinomialNB())])
        return pipeline.fit(x, y)
    
    def nb_classify(self, text):
        df = pd.read_csv('news_df.csv', encoding='latin-1')
        processed_df = NaiveBayesClassifierUtil.preporcessfile(df)
        pipeline = NaiveBayesClassifierUtil.train_data(processed_df)
        news_class_name = pipeline.predict([text])[0]
        return news_class_name