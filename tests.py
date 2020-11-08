from functions import plot_zipf_top500
from functions import plot_zipf_poet
from bs4 import BeautifulSoup
import requests
import language_tool_python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import string
import pandas as pd
from scipy import optimize
from collections import Counter
import matplotlib

def test_cleaning_top500():
    """
    Tests to see if dataset is being properly split into an array of 
    all words found in the poems and upercase lettering and punctuation 
    is removed from the words
    """
    all_poemtxt = ""
    with open("allpoems.txt", "r") as r:
        for line in r:
            # Clean some punctuation with .strip
            all_poemtxt = all_poemtxt + " " + line.strip()

    # Splits all words in all poems into one list of all words in poems
    one_hot_poem = all_poemtxt.split()

    # Remove any left over punctuation and set all words to lowercase to count
    # them as the same
    one_hot_poem = [each_word.strip(string.punctuation)
                    for each_word in one_hot_poem]
    one_hot_poem = [each_word.lower() for each_word in one_hot_poem]

    assert plot_zipf_top500() == one_hot_poem

def test_cleaning_poet():
    """
    Tests to see if dataset is being properly split into an array of 
    all words found in the poems and upercase lettering and punctuation 
    is removed from the words
    """
    all_poemtxt = ""
    with open("allpoemsmaya-angelou.txt", "r") as r:  # Arbitrary Choice
        for line in r:
            # Clean some punctuation with .strip
            all_poemtxt = all_poemtxt + " " + line.strip()

    # Splits all words in all poems into one list of all words in poems
    one_hot_poem = all_poemtxt.split()

    # Remove any left over punctuation and set all words to lowercase to count
    # them as the same
    one_hot_poem = [each_word.strip(string.punctuation)
                    for each_word in one_hot_poem]
    one_hot_poem = [each_word.lower() for each_word in one_hot_poem]

    assert plot_zipf_poet("maya-angelou") == one_hot_poem