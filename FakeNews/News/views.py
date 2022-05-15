from django.shortcuts import render
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string

string.punctuation
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model

reloadModel = load_model('./models/lstmmodel.h5')


def index(request):
    return render(request, 'index.html')


# def output(request):
#     news = {}
#     news['title'] = request.POST.get('headline')
#     news['author'] = request.POST.get('author')
#     news['text'] = request.POST.get('body')
#
#     testData = pd.DataFrame({'x': news}).transpose()
#     testData.fillna('unavailable', inplace=True)
#     testData['comb'] = testData['author'] + "_" + testData['title'] + "_" + testData['text']
#     wordnet = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#
#     def clean(text):
#         # text="".join([char for char in text if char not in string.punctuation])
#         text = "".join([re.sub('[^a-zA-Z]', ' ', char) for char in text])
#         text = text.lower()
#         text = text.split()
#         text = [stemmer.stem(word) for word in text if word not in set(stopwords.words("english"))]
#         text = " ".join(text)
#         return text
#
#     testData['comb'] = testData['comb'].apply(clean)
#     vocab_size = 10000
#     news_title = testData['comb']
#     one_hot_r = [one_hot(words, vocab_size) for words in news_title]
#     sent_len = 150
#     final_input = pad_sequences(one_hot_r, padding='post', maxlen=sent_len)
#     result = reloadModel.predict(final_input)[0]
#     final_result = result[0]
#     return render(request, 'output.html', {'context': final_result})

def news_url(request):
    input_url = request.POST['url']
    html_text = requests.get(input_url).text
    soup = BeautifulSoup(html_text, 'lxml')
    news_headline = soup.find('div', class_='col-sm-8')
    news_article = soup.find_all('section', class_='story-section')
    author_line = soup.find_all('section', class_='story-section')
    for headline in news_headline:
        headline = news_headline.h1.text
    for author in author_line:
        author = soup.find('h5', class_='text-capitalize')
        # author_name = author.a.text
        author_name = 'Abhishek Dangi'
    news_par = []
    for article in news_article:
        news_line = soup.find_all('p')
        for news in news_line:
            news_par.append(news.text)
    output = news_par
    body = ' '.join([str(elem) for elem in news_par])
    news = {'title': headline, 'author': author_name, 'text': body}
    testData = pd.DataFrame({'x': news}).transpose()
    testData.fillna('unavailable', inplace=True)
    testData['comb'] = testData['author'] + "_" + testData['title'] + "_" + testData['text']
    wordnet = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def clean(text):
        # text="".join([char for char in text if char not in string.punctuation])
        text = "".join([re.sub('[^a-zA-Z]', ' ', char) for char in text])
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in set(stopwords.words("english"))]
        text = " ".join(text)
        return text

    testData['comb'] = testData['comb'].apply(clean)
    vocab_size = 10000
    news_title = testData['comb']
    one_hot_r = [one_hot(words, vocab_size) for words in news_title]
    sent_len = 800
    final_input = pad_sequences(one_hot_r, padding='post', maxlen=sent_len)
    result = reloadModel.predict(final_input)[0]
    final_result = result[0]
    return render(request, 'output.html', {'context': final_result})
