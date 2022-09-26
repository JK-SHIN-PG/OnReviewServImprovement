import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import imp
import os
import pickle
import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from src.preprocessors import Preprocessor
from src.main_components import search_OptimalNumTopics
from src.utils import parse_params, getCombination, ensure_path, parse_topic_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default='Params')
    pargs = parser.parse_args()
    
    arg1, _ = parse_params(pargs.yaml)

    paramslist = getCombination(arg1["lda_hyperparams"][0])

    # 1. Review Data Preprocessing
    #ReviewSentenceMatrix : [Review, Sentence]
    #ReviewSentenceWordMatrix : [Review, Sentence, Word]
    #ReviewNounMatrix : [Review, Noun]

    CUSTDATA_PATH = arg1["CUSTDATA_PATH"]
    RESULT_PATH = arg1["RESULT_PATH"]

    ensure_path(RESULT_PATH)
    f = open(RESULT_PATH + "report.txt", "a")
    f.write(str(arg1) + "\n")
    f.write("The number of combinations : {}\n".format(len(paramslist)))
    f.close()
    #DATASETS = ['Amazon_movies', 'Hotel_review', 'Yelp', 'Custom']
    preprocessor = Preprocessor('Custom', num_reviews=arg1["n_reviews"], cust_path=CUSTDATA_PATH) 

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

        params["n_jobs"] = arg1["n_jobs"]
        _, OptNumfromCV, _, maxCV = search_OptimalNumTopics(ReviewNounMatrix, corpus=corpus, dictionary=dictionary, startNum=arg1["startNum"],endNum=arg1["endNum"],step=arg1["step"], args = params)
        TopicmodelC = LdaModel(corpus = corpus, id2word= dictionary, num_topics=OptNumfromCV, iterations=params["iters"], passes=params["passes"], alpha=params["alpha"], eta=params["eta"])
        
        RawTopicTokenListC = TopicmodelC.print_topics(num_words=arg1["n_wordsinTopic"])
        resultC = parse_topic_token(RawTopicTokenListC)
        
        f = open(RESULT_PATH + "report.txt", "a")
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

    print("Finish...")

