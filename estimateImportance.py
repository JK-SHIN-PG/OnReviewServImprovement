from cgi import print_arguments
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from src.main_components import CreateReviewFeatureMatrix, convertSentimentLabel, convertStarLabel, estimateImportance, getPerformance
from src.utils import *

# input TopicNoun matrix
# TopicNounMatrix : [Topic, Noun]
# ReviewToken List : [Review, sentence, word]
# Reveiw List : [Review, sentence]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default='Params')
    pargs = parser.parse_args()
    _, args = parse_params(pargs.yaml)
    TopicNameList = args["TopicList"]

    DIRPATH = args["RESULT_PATH"] + "[2]ML/"
    ensure_path(DIRPATH)

    f = open(DIRPATH + "report.txt", "a")
    f.write(str(args) + "\n")
    f.close()

    #3. Preparing the datasets

    # output Review Feature matrix
    TopicNounMatrix = pickle.load(open(args["RESULT_PATH"] + args["LDA_RESULT_IDX"] + "TopicNounList.pkl", "rb"))
    
    if len(TopicNameList) != len(TopicNounMatrix):
        raise TopicNameListError
    
    ReviewTokenList = pickle.load(open(args["RESULT_PATH"] + "ReviewSentenceWordMatrix.pkl", "rb"))
    ReviewList = pickle.load(open(args["RESULT_PATH"] + "ReviewSentenceMatrix.pkl", "rb")) # for sentiment (service feature satisfaction)
    RatingList = pickle.load(open(args["RESULT_PATH"] + "StarRating.pkl", "rb")) # for overall satisfaction

    ReviewFeatureMatrix = CreateReviewFeatureMatrix(TopicNounMatrix, ReviewTokenList, ReviewList)

    #CvtRFMatrix : CvtReviewFeatureMatrix
    CvtRFMatrix = [list(map(convertSentimentLabel, rowReview)) for rowReview in ReviewFeatureMatrix]

    CvtStarRating = list(map(convertStarLabel, RatingList))

    # create dataframe
    df = pd.DataFrame(CvtRFMatrix)
    df["star"] = CvtStarRating

    range_X = df.shape[1]-2


    # drop zeros in input data
    df = df.loc[~(df.loc[:,:range_X]==0).all(axis=1)].reset_index(drop = True)
    input_X = df.loc[:,:range_X]
    target_y = df.star

    # spliting dataset into train and test dataset
    thres = int(len(df)*0.8)
    X_train = np.array(input_X.iloc[:thres, :])
    y_train = np.array(target_y.iloc[:thres])
    X_test = np.array(input_X.iloc[thres:, :])
    y_test = np.array(target_y.iloc[thres:])

    # 4. training the ML model and find optimal model
    # Possible to add the model that you want
    modelDic = {'LogisticRegression' : LogisticRegression(), 'MLPClassifier' : MLPClassifier(), 'RandomForestClassifier': RandomForestClassifier(), 'LGBMClassifier' : LGBMClassifier()}

    trainedModelDic = {}
    accResults = {}

    CV_option = model_selection.KFold(n_splits = args["n_splits"], random_state= args["random_state"], shuffle= True)

    for argmodel in args["model_hyperparams"]:
        print("Training [{}]...".format(argmodel['model']))
        model = modelDic[argmodel['model']]
        params = argmodel['param'][0]
        if argmodel['model'] == "MLPClassifier":
            params["hidden_layer_sizes"] = [eval(layers) for layers in params["hidden_layer_sizes"]]
        grid = model_selection.GridSearchCV(estimator= model, param_grid= params, cv= CV_option, n_jobs = args["n_jobs"])
        
        trainedModel = grid.fit(X_train, y_train)
        
        # prediction result
        pred_test = trainedModel.predict(X_test)
        
        acc = accuracy_score(y_test, pred_test)
        print("model : {}\tacc : {}%".format(argmodel['model'], round(acc*100,2)))

        f = open(DIRPATH + "report.txt", "a")
        f.write("model : {}\tacc : {}%\n".format(argmodel['model'], round(acc*100,2)))
        f.close()
        
        # store the trained model
        trainedModelDic[argmodel['model']] = trainedModel
        accResults[argmodel['model']] = acc

    # find best model
    bestmodel = [key for key, acc in accResults.items() if max(accResults.values()) == acc][0]
    best_params = trainedModelDic[bestmodel].best_params_
    trainedModel = modelDic[bestmodel].set_params(**best_params).fit(X_train, y_train)

    f = open(DIRPATH + "report.txt", "a")
    f.write("Best model : {}\tacc : {}%\tparams : {}\n".format(bestmodel, round(acc*100,2), best_params))
    f.close()
    
    # 5. Global importance estimation using SAGE

    DIRPATH2 = args["RESULT_PATH"] + "[3]IPA/"
    ensure_path(DIRPATH2)
    if bestmodel == "LogisticRegression":
        Importances = trainedModel.coef_[0]
    Importances = estimateImportance(TopicNameList, modelDic, bestmodel, best_params, CV_option, X_train, y_train, args)
    Performances = getPerformance(X_train)
    plot_IPA(Performances, Importances, TopicNameList, DIRPATH2 + "IPA_Plot.png")

    print("Finished...")
