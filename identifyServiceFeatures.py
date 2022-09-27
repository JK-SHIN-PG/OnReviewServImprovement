import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import imp
import os
import pickle
import argparse
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from src.preprocessors import Preprocessor, CustomizedPreprocessor
from src.main_components import search_OptimalNumTopics
from src.utils import parse_params, getCombination, ensure_path, parse_topic_token
import pandas as pd

# [inheritance] class of CustomizedPreprocessor in src.preprocessors
# just modify this preprocessor if you use your own datasets
class ModifiedCustomizedPreprocessor(CustomizedPreprocessor):
    def __init__(self, path, num_reviews):
        super().__init__(path, num_reviews)

    def getDatasets(self):
        df = pd.read_csv(self.path)
        review_list = list(map(self.makeClearSent, df.Review.tolist()))
        score_list = df.Rating
        return review_list, score_list


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default='Params')
    pargs = parser.parse_args()
    
    args, _ = parse_params(pargs.yaml)

    paramslist = getCombination(args["lda_hyperparams"][0])

    # 1. Review Data Preprocessing
    #ReviewSentenceMatrix : [Review, Sentence]
    #ReviewSentenceWordMatrix : [Review, Sentence, Word]
    #ReviewNounMatrix : [Review, Noun]

    CUSTDATA_PATH = args["CUSTDATA_PATH"]
    RESULT_PATH = args["RESULT_PATH"]

    ensure_path(RESULT_PATH)
    ensure_path(RESULT_PATH + "[1]LDA/")
    f = open(RESULT_PATH + "[1]LDA/summary.txt", "a")
    f.write(str(args) + "\n")
    f.write("The number of combinations : {}\n".format(len(paramslist)))
    f.close()
    #DATASETS = ['Amazon_movies', 'Hotel_review', 'Yelp', 'Custom']
    preprocessor = Preprocessor(args["USE_DATATYPE"], num_reviews=args["n_reviews"], cust_path=CUSTDATA_PATH)
    # you use your datasets.
    if args["USE_DATATYPE"] == "Custom":
        preprocessor.preprocessor = ModifiedCustomizedPreprocessor(num_reviews=args["n_reviews"], path=CUSTDATA_PATH)

    ReviewSentenceMatrix, ReviewSentenceWordMatrix, ReviewNounMatrix = preprocessor.preprocessing()

    StarRating = preprocessor.StarRatingData

    with open(RESULT_PATH + "ReviewSentenceMatrix.pkl", 'wb') as f:
        pickle.dump(ReviewSentenceMatrix, f)

    with open(RESULT_PATH + "ReviewSentenceWordMatrix.pkl", 'wb') as f:
        pickle.dump(ReviewSentenceWordMatrix, f)

    with open(RESULT_PATH + "ReviewNounMatrix.pkl", 'wb') as f:
        pickle.dump(ReviewNounMatrix, f)

    with open(RESULT_PATH + "StarRating.pkl", 'wb') as f:
        pickle.dump(StarRating, f)

    for idx, params in enumerate(paramslist):
        print("progress...[{}/{}]".format(idx+1, len(paramslist)))
        dictionary = corpora.Dictionary(ReviewNounMatrix)
        dictionary.filter_extremes(no_below=params["filter_no_below"], no_above=params["filter_no_above"])
        corpus = [dictionary.doc2bow(text) for text in ReviewNounMatrix]

        params["n_jobs"] = args["n_jobs"]
        _, OptNumfromCV, _, maxCV = search_OptimalNumTopics(ReviewNounMatrix, corpus=corpus, dictionary=dictionary, startNum=args["startNum"],endNum=args["endNum"],step=args["step"], args = params)
        TopicmodelC = LdaModel(corpus = corpus, id2word= dictionary, num_topics=OptNumfromCV, iterations=params["iters"], passes=params["passes"], alpha=params["alpha"], eta=params["eta"])
        
        RawTopicTokenListC = TopicmodelC.print_topics(num_words=args["n_wordsinTopic"])
        resultC = parse_topic_token(RawTopicTokenListC)
        
        f = open(RESULT_PATH + "[1]LDA/summary.txt", "a")
        f.write("idx : [{}]\toptimal number : {}\t coherence score : {}\t".format(idx, OptNumfromCV, round(maxCV,4)))
        f.write(str(params) + "\n")
        f.close()

        DIRPATH = RESULT_PATH + "[1]LDA/{}/".format(idx)
        os.makedirs(DIRPATH)

        f = open(DIRPATH + "report.txt", "a")
        f.write(str(str(params)) + "\n")
        for wordlist in resultC:
            f.write(str(wordlist) + "\n")
        f.close()

        with open(DIRPATH + "TopicNounList.pkl", 'wb') as file:
            pickle.dump(resultC, file)

    print("Finished...")

