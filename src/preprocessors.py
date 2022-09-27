import pandas as pd
import gzip
import json
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from tqdm import tqdm
import re


def getFirstletterPos(pos_tag):
    if pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('J'):
        return 'a'
    elif pos_tag.startswith('R'):
        return 'r'
    else:
        return None

# for sentiment analysis or nlp
# if noun == True, document-noun matrix

def preprocessReview(review):

    # extract stop words
    stop = stopwords.words('english')

    # for lemmatization 
    lemma = WordNetLemmatizer()
        
    # for stemming
    stemmer = PorterStemmer()

    # Remove new line characters
    review = re.sub('\s+', ' ', review)

    # Remove distracting single quotes
    review = re.sub("\'", "", review)

    # lowercasing
    review = review.lower()

    ReviewSentenceList = nltk.sent_tokenize(review)
    SentenceTokenList = []
    NounTokenList = []

    for sentence in ReviewSentenceList:
        # tokenization & pos-tagging
        PosTaggedTokens = pos_tag(nltk.word_tokenize(sentence))

        PosTaggedTokens = [(token, pos) for token, pos in PosTaggedTokens if token not in stop]

        LemmatizedTokens = [(lemma.lemmatize(token, pos = getFirstletterPos(pos)), pos) for token, pos in PosTaggedTokens if getFirstletterPos(pos) is not None]

        NounTokens = [token for token, pos in LemmatizedTokens if pos.startswith('N')]

        #StemmedTokens = [stemmer.stem(token) for token, _ in LemmatizedTokens]
        #SentenceTokenList.append(StemmedTokens)

        Tokens = [token for token, _ in LemmatizedTokens]

        SentenceTokenList.append(Tokens)
        NounTokenList = NounTokenList + NounTokens
        
    '''
    # remove words less than three letters
    tokens = [(word, tag) for word, tag in tokens if len(word) > 3]
    '''
    return ReviewSentenceList, SentenceTokenList, NounTokenList


class Amazon_preprocessor:

    def __init__(self,num_reviews):
        self.path = PATH_BY_DATASET['Amazon']
        self.num_reviews = num_reviews
    def parse(self):
        g = gzip.open(self.path, 'r')
        for l in g:
            yield json.loads(l)

    def getDF(self):
        i = 0
        df = {}
        for d in self.parse():
            if i == self.num_reviews:
                break
            else:
                df[i] = d
                i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def getDatasets(self):
        df = self.getDF()
        df = df[df.reviewText.isnull() != True]
        #idx_list = df.index.to_list()
        review_data = df.reviewText.to_list()
        star_rating_data = df.overall.to_list()
        return review_data, star_rating_data

class Hotel_preprocessor:

    def __init__(self, num_reviews):
        self.path = PATH_BY_DATASET['Hotel_review']
        self.num_reviews = num_reviews

    def getDF(self):
        i = 0
        df = {}
        f = open(self.path, 'r')
        filelines = f.readlines()
        for line in filelines:
            if i == self.num_reviews:
                break
            else:
                df[i] = json.loads(line)
                i+=1
        return pd.DataFrame.from_dict(df, orient='index')

    def getDatasets(self):
        df = self.getDF()
        df = df[df.text.isnull() != True]
        review_data = df.text.to_list()
        score_list = [scores["overall"] for scores in df.ratings]
        return review_data, score_list


class Yelp_preprocessor:

    def __init__(self, num_reviews):
        self.path = PATH_BY_DATASET['Yelp']
        self.num_reviews = num_reviews

    def parse(self):
        g = open(self.path, 'r')
        for l in g:
            yield json.loads(l)

    def getDF(self):
        i = 0
        df = {}
        for d in self.parse():
            if i == self.num_reviews:
                break
            else:
                df[i] = d
                i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def getDatasets(self):
        df = self.getDF()
        df = df[(df.text.isnull() != True) & (df.stars.isnull() != True)]
        review_data = df.text.to_list()
        score_list = df.stars.to_list()
        return review_data, score_list

class CustomizedPreprocessor:
    def __init__(self, path, num_reviews):
        self.path = path
        self.num_reviews = num_reviews

    def makeClearSent(self, sent):
        sent = str(sent)
        sent = sent.replace("\n", "")
        sent = sent.replace("\r", "")
        sent = sent.replace("  ", " ")
        sent = sent.replace(' "', '')
        return sent

    def makeReviewList(self, df):
        review_list = []
        for i in range(len(df)):
            if i == self.num_reviews:
                break
            else:
                review_list.append(self.makeClearSent(df.loc[i, "title"]) + ". " + self.makeClearSent(df.loc[i, "body"]))

        '''
        --- same as ---- 
        review_list = [makeClearSent(df.loc[i, "title"]) + ". " + makeClearSent(df.loc[i, "body"]) for i in range(len(df))]
        '''
        return review_list

    def getDatasets(self):
        df_star2 = pd.read_excel(self.path, sheet_name= '2class')
        df_star3 = pd.read_excel(self.path, sheet_name= '3class')
        df_star4 = pd.read_excel(self.path, sheet_name= '4class')

        df = pd.concat([df_star2, df_star3, df_star4])
        review_score_df = df.loc[:, ["title","body", "rating"]].drop_duplicates().reset_index(drop = True)
    
        review_list = self.makeReviewList(review_score_df)
        score_list = (review_score_df.loc[:,"rating"]//10).to_list()
        if self.num_reviews != -1:
            score_list = score_list[:self.num_reviews]

        return review_list, score_list


class Preprocessor:

    def __init__(self, datasets, num_reviews = -1, min_words = 3, cust_path= None):
        if datasets == "Custom":
            Data_Preprocessor = CustomizedPreprocessor
            self.preprocessor = Data_Preprocessor(cust_path, num_reviews)
        else:
            Data_Preprocessor = PREPROCESSOR_BY_DATASET[datasets]
            self.preprocessor = Data_Preprocessor(num_reviews)
        self.min_words = min_words
        self.OriginalReviewData = None
        self.StarRatingData = None

    def getRawData(self):
        #output : raw reviews, scores
        return self.preprocessor.getDatasets()

    def preprocessing(self):
        self.OriginalReviewData, self.StarRatingData = self.getRawData()
        
        ReviewSentenceList = []
        ReviewSentenceWordList = []
        ReviewNounList = []
        for review in tqdm(self.OriginalReviewData, desc="Preprocessing Reviews..."):
            SentenceList, SentenceWordList, NounList = preprocessReview(review)
            ReviewSentenceList.append(SentenceList)
            ReviewSentenceWordList.append(SentenceWordList)
            ReviewNounList.append(NounList)

        return ReviewSentenceList, ReviewSentenceWordList, ReviewNounList

# Dataset-preprocessor dictionary
PREPROCESSOR_BY_DATASET = {
    'Amazon_movies' : Amazon_preprocessor,
    'Hotel_review' : Hotel_preprocessor,
    'Yelp' : Yelp_preprocessor
}

# Datasets-path dictionary
PATH_BY_DATASET = {
    'Amazon_movies' : './Datasets/Amazon/Movies_and_TV_5.json.gz',
    'Hotel_review' : './Datasets/Hotel_review/review.txt',
    'Yelp' : './Datasets/Yelp/yelp_academic_dataset_review.json'
}