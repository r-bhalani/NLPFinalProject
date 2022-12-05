import json
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def plot_exact_matches():
    # create data
    percentages = [0, 25, 50, 75, 100]
    exact_matches_adv = [23.59550561797753, 34.831460674157306, 33.42696629213483, 27.10674157303371, 29.49438202247191]
    exact_matches_orig = [67.13483146067416, 63.90449438202247, 64.18539325842697, 49.859550561797754, 29.775280898876403]
    
    # plot lines
    plt.plot(percentages, exact_matches_adv, label = "Adversarial SQuAD", linestyle="-")
    plt.plot(percentages, exact_matches_orig, label = "Original SQuAD", linestyle="-.")
    # plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
    # plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")

    plt.title("Exact Match Scores for Perfomance on Alternating Train Sets")
    plt.xlabel("Percentage of Adversarial Data in Train Set")
    plt.ylabel("Exact Match Score")

    plt.legend()
    plt.show()
    plt.savefig('exact_match.png')



def plot_f1():
    # Create f1 scores
    percentages = [0, 25, 50, 75, 100]
    f1_adv = [31.343909119121612, 45.24858266856346, 45.98682604299156, 40.14662669892348, 36.555009074424134]
    f1_orig = [76.9494442977546, 74.40498751069204, 72.64932983884096, 60.45653201542083, 40.48994006516695]
    
    # plot lines
    plt.plot(percentages, f1_adv, label = "Adversarial SQuAD", linestyle="-")
    plt.plot(percentages, f1_orig, label = "Original SQuAD", linestyle="-.")
    # plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
    # plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")

    plt.title("F1 Scores for Perfomance on Alternating Train Sets")
    plt.xlabel("Percentage of Adversarial Data in Train Set")
    plt.ylabel("F1 Score")

    plt.legend()
    plt.show()
    plt.savefig('f1.png')

plot_f1()