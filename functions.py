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

def scrape_poet(poet_name):
    links = []
    poem_links = []
    soup = []
    poem_txt = []
    tool = language_tool_python.LanguageTool('en-US')
    for k in range(35):
        r = requests.get( f"https://www.poemhunter.com/{poet_name}/poems/page-{k}/?")
        if r.status_code == 200:
            src = r.content
            soup.append(BeautifulSoup(src, 'html.parser'))
    for soups in soup:
        for link in soups.find_all('a'):
            if link != None:
                links.append(link.get('href'))
   
    links = [j for j in links if j] 
    cleaned_links = [a for n, a in enumerate(links) if a not in links[:n]] 
    for i in range(len(cleaned_links)):
        if '/poem/' in cleaned_links[i]:
            poem_links.append("http://poemhunter.com"+ cleaned_links[i]+"?")
    for links in poem_links:
        current_poem_link = requests.get(links)
        poem_stuff = current_poem_link.content
        soup = BeautifulSoup(poem_stuff, 'html.parser')
        text = soup.find("p")
        poem_txt.append(tool.correct(text.get_text()))
    for poems in poem_txt:
        f= open(f"allpoems{poet_name}.txt","a")
        f.write(poems)

# Code to Scrape links from Poemhunter's top 500 poems

def scrape_top500():
    links = []
    poem_links = []
    soup = []
    for k in range(20):
        r = requests.get( f"https://www.poemhunter.com/p/m/l.asp?a=0&l=top500&order=title&p={k}")
        src = r.content
        soup.append(BeautifulSoup(src, 'html.parser'))
    for soups in soup:
        for link in soups.find_all('a'):
                if link != None:
                    links.append(link.get('href'))
    links = [j for j in links if j] 
    
    for i in range(len(links)):
        if '/poem/' in links[i]:
            poem_links.append("http://poemhunter.com"+ links[i]+"?")
    poem_txt = []
    tool = language_tool_python.LanguageTool('en-US')
    for links in poem_links:
        current_poem_link = requests.get(links)
        poem_stuff = current_poem_link.content
        soup = BeautifulSoup(poem_stuff, 'html.parser')
        text = soup.find("p")
        poem_txt.append(tool.correct(text.get_text()))
    for poems in poem_txt:
        f= open("allpoems.txt","a")
        f.write(poems)
    
def plot_zipf_top500():
    all_poemtxt = ""
    with open("allpoems.txt", "r") as r:
        for line in r:
            # Do something with line here, like the following.
            all_poemtxt = all_poemtxt + " " + line.strip()
    one_hot_poem = all_poemtxt.split()
    one_hot_poem = [each_word.strip(string.punctuation) for each_word in one_hot_poem]
    one_hot_poem = [each_word.lower() for each_word in one_hot_poem]
    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
    frequency_df1 = pd.DataFrame(list(cnt.items()), columns = ["Words", "Frequency"])
    frequency_df1 = frequency_df1.sort_values('Frequency', ascending=False)
    frequency_df1 = frequency_df1.reset_index(drop=True)

    frequency_df=frequency_df1[0:50]

    x = frequency_df.index+1
    y = frequency_df["Frequency"]
    fits1 = optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  x,  y, p0 = (5000,-.002,500))
    figure(num=None, figsize=(20, 4), dpi=1000, facecolor='w', edgecolor='k',frameon = False)

    width = .5
    plt.xticks(rotation=45)

    plt.bar(frequency_df["Words"],frequency_df["Frequency"],width, color = 'g')
    lobf = plt.plot(x-1, np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2] )
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title("Word Frequency Analysis of Top 500 Poem Hunter Poems, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 10}

    matplotlib.rc('font', **font)
    residuals = y - (np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])


def plot_zipf_poet(poetname):
    plot_name = poetname.replace("-"," ")
    names = plot_name.split()
    plot_name = names[0].capitalize() + " " + names[1].capitalize()
    all_poemtxt = ""
    with open(f"allpoems{poetname}.txt", "r") as r:
        for line in r:
            all_poemtxt = all_poemtxt + " " + line.strip()
    one_hot_poem = all_poemtxt.split()
    one_hot_poem = [each_word.strip(string.punctuation) for each_word in one_hot_poem]
    one_hot_poem = [each_word.lower() for each_word in one_hot_poem]

    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—": #because of emily dickinson
            del cnt[i]
    frequency_df = pd.DataFrame(list(cnt.items()), columns = ["Words", "Frequency"])
    frequency_df = frequency_df.sort_values('Frequency', ascending=False)
    frequency_df = frequency_df.reset_index(drop=True)

    for i in range(50,len(frequency_df)):
        frequency_df.drop([i],axis=0,inplace=True)
    x = frequency_df.index+1
    y = frequency_df["Frequency"]
    fits1 = optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  x,  y, p0 = (y[0],-0.01,50))
    figure(num=None, figsize=(20, 4), dpi=1000, facecolor='w', edgecolor='k',frameon = False)

    width = .5
    plt.xticks(rotation=45)

    plt.bar(frequency_df["Words"],frequency_df["Frequency"],width, color = 'g')
    plt.plot(x-1, np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2] )
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title(f"Word Frequency Analysis of {plot_name} Poems, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 10}

    matplotlib.rc('font', **font)
    residuals = y - (np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])

def all_english_zipf():
    frequency_df1 = pd.read_csv("unigram_freq.csv",names = ["Words", "Frequency"])
    frequency_df=frequency_df1[0:50]
    x = frequency_df.index+1
    y = frequency_df["Frequency"]
    fits1 = optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  x,  y, p0 = (y[0],-0.01,50))
    residuals = y - (np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)

    figure(num=None, figsize=(20, 4), dpi=1000, facecolor='w', edgecolor='k',frameon = False)

    width = .5
    plt.xticks(rotation=45)

    plt.bar(frequency_df["Words"],frequency_df["Frequency"],width, color = 'g')
    plt.plot(x-1, np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2] )
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title("Word Frequency Analysis of the Gutenburg Dictionary, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 10}

    matplotlib.rc('font', **font)
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])

'''
    def plot_all_dict():
    all_txt = ""
    with open("dictionary.txt", "r") as r:
        for line in r:
            # Do something with line here, like the following.
            all_txt = all_txt + " " + line.strip()
    one_hot_dict = all_txt.split()
    one_hot_dict = all_txt.strip(string.punctuation)
    cnt = Counter()
    for words in one_hot_dict:
        cnt[words] += 1
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—": #because of emily dickinson
            del cnt[i]
        if str(i) in "bcdefghjklmnopqrstuvwsyzBCDEFGHJKLMNOPQRSTUVWSYZ12345679":
            del cnt[i]
    frequency_df = pd.DataFrame(list(cnt.items()), columns = ["Words", "Frequency"])
    frequency_df = frequency_df.sort_values('Frequency', ascending=False)
    frequency_df = frequency_df.reset_index(drop=True)

    for i in range(50,len(frequency_df)):
        frequency_df.drop([i],axis=0,inplace=True)
    x = frequency_df.index+1
    y = frequency_df["Frequency"]
    fits1 = optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  x,  y, p0 = (y[0],-0.01,50))
    figure(num=None, figsize=(20, 4), dpi=1000, facecolor='w', edgecolor='k',frameon = False)

    width = .5
    plt.xticks(rotation=45)

    plt.bar(frequency_df["Words"],frequency_df["Frequency"],width, color = 'g')
    plt.plot(x-1, np.exp(fits1[0][1]*x)*fits1[0][0]+fits1[0][2] )
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title("Word Frequency Analysis of the Gutenburg Dictionary, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 10}

    matplotlib.rc('font', **font)
    '''