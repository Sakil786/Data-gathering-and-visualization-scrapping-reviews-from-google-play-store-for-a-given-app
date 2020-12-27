
# # Data gathering and visualization
# * Here we are web scrapping  reviews from google play store for a given app by using python and we are creating two word clouds - one for words occurring in the most favourable review, and another one for and the lowest reviews.

# # Setup
# * Let's install the required packages and setup the imports:
#installing google play API
pip install google-play-scraper
#install watermark
pip install watermark
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p pandas,matplotlib,seaborn,google_play_scraper')
# importing all the required Libraries
import glob
import json
import csv
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import string
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer 
import re
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from google_play_scraper import Sort, reviews, app

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
#choosing  some apps
# e.g. enter this URL: it.rortos.airfighters
app_name=input("Enter URL from google playstore")
app_packages = [
  
   app_name
    
    
]
# # Scraping App Information
app_infos = []
for ap in tqdm(app_packages):
  info = app(ap, lang='en', country='us')
  del info['comments']
  app_infos.append(info)
def print_json(json_object):
  json_str = json.dumps(
    json_object, 
    indent=2, 
    sort_keys=True, 
    default=str
  )
  print(highlight(json_str, JsonLexer(), TerminalFormatter()))
#print_json(app_infos)
app_infos_df = pd.DataFrame(app_infos)
app_infos_df.head()
app_infos_df.shape
# # Scraping App Reviews
app_reviews = []
for ap in tqdm(app_packages):
  for score in list(range(1, 6)):
    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
      rvs, _ = reviews(
        ap,
        lang='en',
        country='us',
        sort=sort_order,
        count= 200 if score == 3 else 100,
        filter_score_with=score
      )
      for r in rvs:
        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
        r['appId'] = ap
      app_reviews.extend(rvs)
print_json(app_reviews[0])
app_reviews_df = pd.DataFrame(app_reviews)
app_reviews_df.head(2)
app_reviews_df_all_reviews_rating_data=pd.DataFrame(app_reviews_df,columns={"content","score"})
app_reviews_df_all_reviews_rating_data.head()
app_reviews_df_all_reviews_rating_data.to_csv('reviews.csv', index=None, header=True)
reviews_data=pd.read_csv("reviews.csv")
reviews_data.head()
# # WordCloud on rating 5
reviews_data_rating5 = reviews_data[reviews_data["score"]==5]
reviews_data_rating5.tail()
def stemming(tokens):
    #ps=PorterStemmer()
    ps = WordNetLemmatizer()
    stem_words=[]
    for x in tokens:
        stem_words.append(ps.lemmatize(x))
    return stem_words
def create_Word_Corpus(df):
    words_corpus = ''
    for val in reviews_data_rating5["content"]:
        text = val.lower()
        all_stopwords_gensim =STOPWORDS.union(set(['game','nt']))
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in all_stopwords_gensim]
        tokens = stemming(tokens)
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus
def plot_Cloud(wordCloud):
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
import nltk
nltk.download('wordnet')
reviews_data_rating5_wordcloud = WordCloud(background_color='black',width=900, height=500).generate(create_Word_Corpus(reviews_data_rating5))
plot_Cloud(reviews_data_rating5_wordcloud)
# # WordCloud on rating 1
reviews_data_rating1 = reviews_data[reviews_data["score"]==1]
reviews_data_rating1.tail()
def create_Word_Corpus_for_rating1(df):
    words_corpus = ''
    for val in reviews_data_rating1["content"]:
        text = val.lower()
        all_stopwords_gensim =STOPWORDS.union(set(['game','t','n','`','V','e']))
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in all_stopwords_gensim]
        tokens = stemming(tokens)
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus
reviews_data_rating1_wordcloud = WordCloud(background_color='black',width=900, height=500).generate(create_Word_Corpus_for_rating1(reviews_data_rating1))
plot_Cloud(reviews_data_rating1_wordcloud)

