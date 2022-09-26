from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm


def calculate_Coherence(model, corpus, dictionary, ReviewNoun):
    
    C_npmi = CoherenceModel(model= model, texts = ReviewNoun, corpus= corpus, dictionary= dictionary, coherence='c_npmi')
    C_V = CoherenceModel(model= model, texts = ReviewNoun, corpus= corpus, dictionary= dictionary, coherence='c_v')
    return C_npmi.get_coherence(), C_V.get_coherence()

def calculate_Perplexity(model, corpus):
    return model.log_perplexity(corpus)

def calculate_TopicDissimilarity():
    pass

def search_OptimalNumTopics(ReviewNoun, corpus, dictionary, startNum, endNum, step, args, plot=False):

    c_npmiList = []
    c_vList = []
    for NumTopic in tqdm(range(startNum, endNum, step), desc="Searching the optimal number of topics"):
        model = LdaMulticore(corpus = corpus, id2word= dictionary, num_topics=NumTopic, workers=args["n_jobs"],iterations=args["iters"], passes=args["passes"], alpha=args["alpha"], eta=args["eta"])
        c_npmi, c_v = calculate_Coherence(model, ReviewNoun=ReviewNoun, corpus=corpus, dictionary=dictionary)
        c_npmiList.append(c_npmi)
        c_vList.append(c_v)
        
    if plot == True:
        x = range(startNum, endNum, step)
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle("Coherence Result")

        ax1.plot(x, c_npmiList)
        ax2.plot(x, c_vList)

        ax1.set_title("U mass (max : 0)")
        ax2.set_title("C_V (max : 1)")


        ax1.set(xlabel="Num Topics")
        ax2.set(xlabel="Num Topics")

        plt.show()

    OptNumfromCVPMI = (c_npmiList.index(max(c_npmiList)) * step) + startNum
    OptNumfromCV = (c_vList.index(max(c_vList)) * step) + startNum

    print("Optimal Number of Topics from C_NPMI : {}".format(OptNumfromCVPMI))
    print("Optimal Number of Topics from C_V : {}".format(OptNumfromCV))

    return OptNumfromCVPMI, OptNumfromCV, max(c_npmiList), max(c_vList)

def convertSentimentLabel(score):
    if (0.525 <= score) and (score <= 1):
        return 4
    elif (0.05 <= score) and (score < 0.525):
        return 3
    elif (-0.525 < score) and (score <= -0.05):
        return 2
    elif (-1 <= score) and (score <= -0.525):
        return 1
    else: 
        return 0

def convertStarLabel(star):
    if (star == 1) or (star == 2) or (star == 3):
        return 0
    else:
        return 1

def CreateReviewFeatureMatrix(TopicNounMatrix, ReviewTokenList, ReviewList):
    analyser = SentimentIntensityAnalyzer()
    
    NumTopic = len(TopicNounMatrix)
    ReviewFeatureMatrix = []
    
    for r_idx, review in enumerate(tqdm(ReviewTokenList)):
        
        SentenceFeatureMatrix = [0]*NumTopic
        SentencecountMatrix = [0]*NumTopic
        for s_idx, sentence in enumerate(review):
            WordFeatureMatrix = [0]*NumTopic
            WordcountMatrix = [0]*NumTopic
            for w_idx, word in enumerate(sentence): # pos tagged data 인지 확인
                for t_idx, topicwordlist in enumerate(TopicNounMatrix):
                    if word in topicwordlist:
                        score = analyser.polarity_scores(ReviewList[r_idx][s_idx])['compound']
                        if score != 0:
                            WordFeatureMatrix[t_idx] += score
                            WordcountMatrix[t_idx] += 1

            if np.sum(WordcountMatrix) == 0: pass
            else:
                templist = []
                for idx, sumScore in enumerate(WordFeatureMatrix):
                    if WordcountMatrix[idx] != 0:
                        templist.append(sumScore/WordcountMatrix[idx])
                    else:
                        templist.append(0)
                WordFeatureMatrix = templist

                SentenceFeatureMatrix = (np.array(SentenceFeatureMatrix) + np.array(WordFeatureMatrix)).tolist()
                arraySFM = np.array(WordFeatureMatrix)
                SentencecountMatrix = (np.array(SentencecountMatrix) + (arraySFM != 0)*np.ones(arraySFM.shape)).tolist()


        if np.sum(SentencecountMatrix) == 0: pass
        else:
            temp = []
            for idx, sumScore in enumerate(SentenceFeatureMatrix):
                if SentencecountMatrix[idx] != 0:
                    temp.append(sumScore/SentencecountMatrix[idx])
                else:
                    temp.append(0)
            SentenceFeatureMatrix = temp 

        ReviewFeatureMatrix.append(SentenceFeatureMatrix)

    return ReviewFeatureMatrix


def getPerformance(input):
    return np.array(input).mean(axis=0)

def estimateImportance(trainedModel, X_input, y_output, TopicNameList, args):
    import sage
    imputer = sage.MarginalImputer(trainedModel, X_input[:args["n_samples"]])
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    sage_values = estimator(X_input, y_output)
    sage_values.plot(TopicNameList)
    return sage_values.values
