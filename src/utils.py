import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import yaml

class TopicNameListError(Exception):
    def __init__(self):
        super().__init__('The number of topics in TopicNameList (Params.yaml) and the number of feature in ReviewFeatureMatrix must be same. please check your params.yaml file')

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def parse_params(filename):
    with open('{}.yaml'.format(filename)) as f:
        argsfile = yaml.load_all(f, Loader=yaml.FullLoader)
        argslist = [args for args in argsfile]
        arg1 = argslist[0]
        arg2 = argslist[1]
    return arg1, arg2

def calculate_sparsity(matrix):
    from numpy import count_nonzero
    matrix = np.array(matrix)
    sparsity = 1.0 - (count_nonzero(matrix) / float(matrix.size))
    print("sparsity : {}".format(round(sparsity,3)))


def parse_topic_token(RawTopicTokenList):
    TopicDic = []
    for idx, eqs in RawTopicTokenList:
        tokenlist = []
        eqlist = eqs.split(" + ")
        for idx, eq in enumerate(eqlist):
            word = eq.split("*")[1]
            tokenlist.append(word.split('"')[1])
        TopicDic.append(tokenlist)

    return TopicDic

def getCombination(tdict):
    from itertools import product
    temp = {}
    for key, val in tdict.items():
        
        # for key-value combinations 
        temp[key] = product([key], val)
        #print(list(res[key]))
    
    # computing cross key combinations
    combtuples = list(product(*temp.values()))

    comblist = []
    for tuples in combtuples:
        tempdic = {}
        for key, value in tuples:
            tempdic[key] = value
        comblist.append(tempdic)

    print("\nThe number of hyper-parameter combinations : {} \n".format(len(comblist)))
    return comblist

def plot_IPA(Performance, Importance, TopicNameList, filename):
    
    MeanPerformance = Performance.mean()
    MeanImportance = Importance.mean()
    plt.figure(figsize=(8,8))
    plt.scatter(Performance, Importance, color = 'g')
    for idx, topic_name in enumerate(TopicNameList):
        plt.text(Performance[idx]+0.0005, Importance[idx]+0.0005, topic_name)

    plt.axhline(MeanImportance, color = 'b', linestyle = '--')
    plt.axvline(MeanPerformance, color = 'b', linestyle = '--')
    plt.ylabel("Importance")
    plt.xlabel("Performance")
    plt.title("Importance-Performance Analysis (IPA)", fontsize = 20)
    #plt.text(0.50, 0.26, "Q2", fontweight= "bold", fontsize = 15)
    #plt.text(2.20, 0.26, "Q1", fontweight= "bold", fontsize = 15)
    #plt.text(0.50, -0.10, "Q3", fontweight= "bold", fontsize = 15)
    #plt.text(2.20, -0.10, "Q4", fontweight= "bold", fontsize = 15)
    plt.savefig(filename)
    plt.show()
