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
    """
    Takes in a poet name and scrapes their entire catalog from
    poethunter.com and saves it as a .txt file

    Input a poet name in a string with each part of a poet's name
    separated by hyphens. Obtains every poem that poet authored
    saved in poethunter.com and saves them all as plaintext in a
    file named "allpoems{poet-name}.txt". Code also does a bit of
    formatting through "Python Language Tool" to resolve templating
    erros when scraping

    Args:
        poet_name:
            A string, representing a poet's name, formatted as the
            poet's full name with hyphens separating each part.
            i.e. "edgar-allen-poe", "maya-angelou"

    """
    # Declaring all the lists used
    links = []
    poem_links = []
    soup = []
    poem_txt = []
    tool = language_tool_python.LanguageTool('en-US')

    # Obtains the name of every poem authored by input poet by sweeping
    # Through the poet's catalog and saving each poem name, then inputing
    # poem names into a list of link stubs
    for k in range(35):
        r = requests.get(
            f"https://www.poemhunter.com/{poet_name}/poems/page-{k}/?")
        if r.status_code == 200:  # checks to see if the link is real
            src = r.content

            # appends webpage into a list of Beautiful Soup objects
            soup.append(BeautifulSoup(src, 'html.parser'))

    for soups in soup:
        # for each link, find all link stubs in the webpage
        for link in soups.find_all('a'):
            if link is not None:  # discard false links stubs
                links.append(link.get('href'))  # append link info to a list

    # Cleans link stubs again and removes repeats or empty ones
    links = [j for j in links if j]
    cleaned_links = [a for n, a in enumerate(links) if a not in links[:n]]

    # Takes the list of clean link stubs and concatenates them into actual
    # links further filtering for links that have the /poem/ chunk which means
    # the link leads to a poem on the site
    for i in range(len(cleaned_links)):
        if '/poem/' in cleaned_links[i]:
            # Place all the working poem link stubs into a poem links list
            poem_links.append("http://poemhunter.com" + cleaned_links[i] + "?")

    # Takes list of all extracted poem names, scrapes the poem,
    # formats the text, and saves it on to a list of saved poems.
    for links in poem_links:
        current_poem_link = requests.get(links)
        poem_stuff = current_poem_link.content
        soup = BeautifulSoup(poem_stuff, 'html.parser')
        text = soup.find("p")  # "p" tag is html text
        # tool.correct using Language Tool Python to format spacing errors
        poem_txt.append(tool.correct(text.get_text()))

    # Save the text of each poem line by line into a .txt file
    for poems in poem_txt:
        f = open(f"allpoems{poet_name}.txt", "a")
        f.write(poems)


def scrape_top500():
    """
    Scrapes the top 500 poems catalog from poethunter.com and saves
    it as a .txt file

    Obtains all poems in the top 500 poems list on poethunter.com,
    and saves them all as plaintext in a file named "allpoems.txt".
    Code also does a bit of formatting through "Python Language Tool"
    to resolve templating erros when scraping
    """

    # Declaring all the lists used
    links = []
    poem_links = []
    soup = []
    poem_txt = []
    tool = language_tool_python.LanguageTool('en-US')

    # Obtains the name of every poem in the top 500 poem list by sweeping
    # Through the poethunter.com catalog and saving each poem name, then
    # inputing poem names into a list of links stubs
    for k in range(20):
        r = requests.get(
            f"https://www.poemhunter.com/p/m/l.asp?a=0&l=top500&order=title&p={k}")
        src = r.content
        # appends webpage into a list of BS objects
        soup.append(BeautifulSoup(src, 'html.parser'))
    for soups in soup:
        for link in soups.find_all(
                'a'):  # for each link, find all link stubs in the webpage
            if link is not None:  # discard false links stubs
                links.append(link.get('href'))  # append link info to a list

    # get repeats out of the list
    links = [j for j in links if j]

    # Takes the list of clean link stubs and contactonates
    # them into actual links further filtering for links
    # that have the /poem/ chunk which means the link leads
    # to a poem on the site
    for i in range(len(links)):
        if '/poem/' in links[i]:
            # Place all the working poem link stubs into a poem links list
            poem_links.append("http://poemhunter.com" + links[i] + "?")

    # Takes list of all extracted poem names,
    # scrapes the poem, formats the text,
    # and saves it on to a list of saved poems.
    for links in poem_links:
        current_poem_link = requests.get(links)
        poem_stuff = current_poem_link.content
        soup = BeautifulSoup(poem_stuff, 'html.parser')
        text = soup.find("p")  # "p" tag is html text
        # tool.correct using Language Tool Python to format spacing errors
        poem_txt.append(tool.correct(text.get_text()))

    # Save the text of each poem line by line into a .txt file
    for poems in poem_txt:
        f = open("allpoems.txt", "a")
        f.write(poems)


def plot_zipf_top500():
    """
    Plots a bar-graph with curve fit of the top 50
    most used words in the top 500 poems list of
    poethunter.com

    Assumes that you have the file allpoems.txt containing
    the top 500 poems from poethunter.com, if not in the repo
    use the "scrape_top500" to obtain the file. Code opens the
    file, takes out punctuation and turns all words lower-case,
    puts all words in a list, counts the frequency of appearance
    of each word. Next turns the count object into a pandas dataframe
    which organizes it based on decending frequency and then drops all
    but the top 50 words from the dataframe. The data is then curve fit
    to a function of the form f(r) = a*e^(r*b) + c, where f is frequency
    and r is rank, a, b, and c are curve fit constants. Finally plots the
    curve fit as a line and the entire dataframe with a bar graph, with a
    calculated r^2 value for the fit.

    Returns:
        one_hot_poem:
            List of strings containg an index for every word in the
            poems, used exclusively for unit testing
    """
    # Open allpoems.txt file and concatenates all the lines into a single
    # string
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

    # Creates a counter object and counts the word appearance
    # frequency in all the words of all the poems saved
    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1

    # final punctuation clean (for leftover punctuation that gets counted as a
    # word)
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—":  # because of emily dickinson
            del cnt[i]

    # Converts the counter object into a Pandas dataframe with labels for the
    # columns
    frequency_df_all = pd.DataFrame(
        list(
            cnt.items()),
        columns=[
            "Words",
            "Frequency"])

    # Organize the words in decending frequency order and make sure the index
    # are numbered
    frequency_df_all = frequency_df_all.sort_values(
        'Frequency', ascending=False)
    frequency_df_all = frequency_df_all.reset_index(drop=True)

    # Drops all but the top 50 values
    frequency_df = frequency_df_all[0:50]

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the
    # form described above uses the optimize.curve_fit function from
    # Scipy to create curve fit values
    fits1 = optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(5000, -.002, 500))

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    width = .5
    plt.xticks(rotation=45)

    # Actually plots the word frequency bar graph and the appropriate curve fit
    plt.bar(frequency_df["Words"], frequency_df["Frequency"], width, color='g')
    plt.plot(x - 1, np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])

    # More visual plot stuff
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title("Word Frequency Analysis of Top 500 Poem Hunter Poems, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}
    matplotlib.rc('font', **font)

    # Calculates r^2 with curve fit data and uses string comprehention to print
    # to plot it on the legend
    residuals = y - (np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])
    return one_hot_poem  # For unit test


def plot_zipf_poet(poetname):
    """
    Takes a poet name in a string with each part of a poet's name
    separated by hyphens and plots a bar-graph with curve fit of the
    top 50 most used words in the input poet's saved poem list

    Assumes that you have the file allpoems{poetname}.txt containing
    the plaintext of all poems from that author on poethunter.com,
    if not in the repo use the "scrape_poet(poetname)" to obtain the file.
    Code opens the file, takes out punctuation and turns all words lower-case,
    puts all words in a list, counts the frequency of appearance
    of each word. Next turns the count object into a pandas dataframe
    which organizes it based on decending frequency and then drops all
    but the top 50 words from the dataframe. The data is then curve fit
    to a function of the form f(r) = a*e^(r*b) + c, where f is frequency
    and r is rank, a, b, and c are curve fit constants. Finally plots the
    curve fit as a line and the entire dataframe with a bar graph, with a
    calculated r^2 value for the fit.

    Args:
        poet_name:
            A string, representing a poet's name, formatted as the
            poet's full name with hyphens separating each part.
            i.e. "edgar-allen-poe", "maya-angelou"

    Returns:
        one_hot_poem:
            List of strings containg an index for every word in the
            poems, used exclusively for unit testing

    """
    # Gets first and last name of input poet to use for
    # The plot
    plot_name = poetname.replace("-", " ")
    names = plot_name.split()
    plot_name = names[0].capitalize() + " " + names[-1].capitalize()
    all_poemtxt = ""

    # Open allpoems{poetname}.txt file and concatenates all the lines into a
    # single
    with open(f"allpoems{poetname}.txt", "r") as r:
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

    # Creates a counter object and counts the word appearance
    # frequency of all the words in all the poems saved
    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1

    # final punctuation clean (for leftover punctuation that gets counted as a
    # word)
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—":  # because of emily dickinson
            del cnt[i]

    # Converts the counter object into a Pandas dataframe with labels for the
    # columns
    frequency_df = pd.DataFrame(
        list(
            cnt.items()),
        columns=[
            "Words",
            "Frequency"])

    # Organize the words in decending frequency order and make sure the index
    # are numbered
    frequency_df = frequency_df.sort_values('Frequency', ascending=False)
    frequency_df = frequency_df.reset_index(drop=True)

    # Drops all but the top 50 values
    for i in range(50, len(frequency_df)):
        frequency_df.drop([i], axis=0, inplace=True)

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the form
    # described above uses the optimize.curve_fit function from Scipy to
    # create curve fit values
    fits1 = optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(y[0], -0.01, 50))

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    width = .5
    plt.xticks(rotation=45)

    # Actually plots the word frequency bar graph and the appropriate curve fit
    plt.bar(frequency_df["Words"], frequency_df["Frequency"], width, color='g')
    plt.plot(x - 1, np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])

    # More visual and label plot stuff
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title(
        f"Word Frequency Analysis of {plot_name} Poems, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}
    matplotlib.rc('font', **font)

    # Calculates r^2 with curve fit data and uses string comprehention to print
    # to plot it on the legend
    residuals = y - (np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])
    return one_hot_poem


def all_english_zipf():
    """
    Plots a bar-graph with curve fit of the top 50
    most used words in the Google Web Trillion Word Corpus

    Assumes that you have the file unigram_freq.csv containing
    all word frequency in the Google Web Trillion Word Corpus.
    If not in your repo, download from kaggle with the link:

    https://www.kaggle.com/rtatman/english-word-frequency

    Code opens the file, turns its contents into a pandas dataframe
    which drops all but the top 50 words from the dataframe. The data
    is then curve fit to a function of the form f(r) = a*e^(r*b) + c,
    where f is frequency and r is rank, a, b, and c are curve fit constants.
    Finally plots the curve fit as a line and the entire dataframe with a bar
    graph, with a calculated r^2 value for the fit.
    """

    # Reads the word frequency file and creates a labeled pandas dataframe
    frequency_df1 = pd.read_csv(
        "unigram_freq.csv", names=[
            "Words", "Frequency"])

    # Drops all but the top 50 values
    frequency_df = frequency_df1[0:50]

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the form
    # described above uses the optimize.curve_fit function from Scipy to
    # create curve fit values
    fits1 = optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(y[0], -0.01, 50))

    # Calculates r^2 value with curvefit data
    residuals = y - (np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    width = .5
    plt.xticks(rotation=45)

    # Actually plots the word frequency bar graph and the appropriate curve fit
    plt.bar(frequency_df["Words"], frequency_df["Frequency"], width, color='g')
    plt.plot(x - 1, np.exp(fits1[0][1] * x) * fits1[0][0] + fits1[0][2])

    # More visual and label plot stuff
    plt.ylabel("Word Frequency")
    plt.xlabel("Top 50 most frequent Words")
    plt.title("Word Frequency Analysis of the English Language, Top 50 Words, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}
    matplotlib.rc('font', **font)

    # Using String Comprehension to plot r^2 value on legend
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])


def all_english_zipf_all():
    """
    Plots a loglog scatterplot with curve fit for the entirety
    of the word frequency dataset of the Google Web Trillion Word Corpus

    Assumes that you have the file unigram_freq.csv containing
    all word frequency in the Google Web Trillion Word Corpus.
    If not in your repo, download from kaggle with the link:

    https://www.kaggle.com/rtatman/english-word-frequency

    Code opens the file, turns its contents into a pandas dataframe. The data
    is then curve fit to a function of the form log(f(r)) = e^(m) * log(r)^k
    where f is frequency and r is rank, and m and k are curve fit constants.
    Finally plots the curve fit as a line and the entire dataframe of frequency
    over rank with a lolog scatterplot, with a calculated r^2 value for the fit
    """

    # Reads the word frequency file and creates a labeled pandas dataframe
    frequency_df = pd.read_csv(
        "unigram_freq.csv", names=[
            "Words", "Frequency"])

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the
    # form described above uses the polyfit function from Numpy to
    # create curve fit values
    k, m = np.polyfit(np.log(x), np.log(y), 1)

    # Calculates r^2 value with curvefit data
    residuals = np.log(y) - np.log((np.exp(m) * (x**(k))))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    r_sq = 1 - (ss_res / ss_tot)

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    ax = plt.gca()
    width = 15
    plt.xticks(rotation=45)

    # Ensures the plot has loglog scales
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Actually plots the word frequency over rank scatterplot and the
    # appropriate curve fit
    plt.scatter(x, y, width, color='g')
    plt.loglog(x, np.exp(m) * (x**(k)))

    # More visual and label plot stuff
    plt.ylabel("Word Frequency")
    plt.xlabel("Word Rank")
    plt.title(
        "Word Frequency Analysis of the English Language, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}
    matplotlib.rc('font', **font)

    # Using String Comprehension to plot r^2 value on legend
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])


def plot_zipf_poet_all(poetname):
    """
    Takes a poet name in a string with each part of a poet's name
    separated by hyphens and plots a loglog scatterplot with curve fit
    of word frequency over word rank for all poems in file

    Assumes that you have the file allpoems{poetname}.txt containinh
    the plaintext of all poems from that author on poethunter.com, if
    not in the repo use the "scrape_poet(poetname)" to obtain the file.

    Code opens the file, takes out punctuation and turns all words lower-case,
    puts all words in a list, counts the frequency of appearance of each word.
    Next turns the count object into a pandas dataframe, organizes it based
    on decending frequency. The data is then fit to a function of the form
    log(f(r)) = e^(m) * log(r)^k where f is frequency, r is rank, and m and k
    are curve fit constants. Finally plots the curve fit line and the entire
    dataframe of frequency over rank with a lolog scatterplot, with
    a calculated r^2 value for the fit.

    Args:
        poet_name:
            A string, representing a poet's name, formatted as the
            poet's full name with hyphens separating each part.
            i.e. "edgar-allen-poe", "maya-angelou"

    """
    # Gets first and last name of input poet to use for
    # The plot
    plot_name = poetname.replace("-", " ")
    names = plot_name.split()
    plot_name = names[0].capitalize() + " " + names[-1].capitalize()
    all_poemtxt = ""

    # Open allpoems{poetname}.txt file and concatenates all the lines into a
    # single
    with open(f"allpoems{poetname}.txt", "r") as r:
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

    # Creates a counter object and counts the word appearance frequency
    # of all the words in all the poems saved
    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1

    # final punctuation clean (for leftover punctuation that gets counted as a
    # word)
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—":  # because of emily dickinson
            del cnt[i]

    # Converts the counter object into a Pandas dataframe
    # with labels for the columns
    frequency_df = pd.DataFrame(
        list(
            cnt.items()),
        columns=[
            "Words",
            "Frequency"])
    frequency_df = frequency_df.sort_values('Frequency', ascending=False)
    frequency_df = frequency_df.reset_index(drop=True)

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the
    # form described above uses the polyfit function from Numpy to
    # create curve fit values
    k, m = np.polyfit(np.log(x), np.log(y), 1)

    # Calculates r^2 value with curvefit data
    residuals = np.log(y) - np.log((np.exp(m) * (x**(k))))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    r_sq = 1 - (ss_res / ss_tot)

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    width = 20
    plt.xticks(rotation=45)
    ax = plt.gca()

    # Ensures the plot has loglog scales
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Actually plots the word frequency over rank scatterplot and the
    # appropriate curve fit
    plt.scatter(x, y, width, color='g')
    plt.plot(x - 1, np.exp(m) * (x**(k)))

    # More visual and label plot stuff
    plt.ylabel("Word Frequency(log10)")
    plt.xlabel("Word Rank(log10)")
    plt.title(
        f"Word Frequency Analysis of {plot_name} Poems, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    # Using String Comprehension to plot r^2 value on legend
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])


def plot_zipf_top500_all():
    """
    Plots a loglog scatterplot with curve fit
    of word frequency over word rank for thetop 50
    most used words in the top 500 poems list of
    poethunter.com


    Assumes that you have the file allpoems.txt containing
    the top 500 poems from poethunter.com, if not in the repo
    use the "scrape_top500" to obtain the file.

    Code opens the file, takes out punctuation and turns all words lower-case,
    puts all words in a list, counts the frequency of appearance of each word
    Next turns count object into a pandas dataframe which organizes it based
    on decending frequency. The data is then fit to a function of the form:
    log(f(r)) = e^(m) * log(r)^k where f is frequency, r is rank, and m and k
    are curve fit constants. Finally plots the curve fit line and the entire
    dataframe of frequency over rank in a lolog scatterplot, with a calculated
    r^2 value for the fit.
    """

    # Open allpoems.txt file and concatenates all the lines into a single
    # string
    all_poemtxt = ""
    with open("allpoems.txt", "r") as r:
        for line in r:
            all_poemtxt = all_poemtxt + " " + line.strip()

    # Splits all words in all poems into one list of all words in poems
    one_hot_poem = all_poemtxt.split()

    # Remove any left over punctuation and set all words to lowercase to count
    # them as the same
    one_hot_poem = [each_word.strip(string.punctuation)
                    for each_word in one_hot_poem]
    one_hot_poem = [each_word.lower() for each_word in one_hot_poem]

    # Creates a counter object and counts the word appearance frequency
    # in all the words of all the poems saved
    cnt = Counter()
    for words in one_hot_poem:
        cnt[words] += 1

    # final punctuation clean (for leftover punctuation that gets counted as a
    # word)
    for i in list(cnt.keys()):
        if str(i) in string.punctuation:
            del cnt[i]
        if str(i) in "—":  # because of emily dickinson
            del cnt[i]

    # Converts the counter object into a Pandas dataframe with labels for the
    # columns
    frequency_df = pd.DataFrame(
        list(
            cnt.items()),
        columns=[
            "Words",
            "Frequency"])

    # Organize the words in decending frequency order and make sure the index
    # are numbered
    frequency_df = frequency_df.sort_values('Frequency', ascending=False)
    frequency_df = frequency_df.reset_index(drop=True)

    # Sets up the curve fit axis into variables x and y
    x = frequency_df.index + 1
    y = frequency_df["Frequency"]

    # Uses x and y variables to create a curve fit of the data in the form
    # described above uses the polyfit function from Numpy to create curve fit
    # values
    k, m = np.polyfit(np.log(x), np.log(y), 1)

    # Calculates r^2 value with curvefit data
    residuals = np.log(y) - np.log((np.exp(m) * (x**(k))))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    r_sq = 1 - (ss_res / ss_tot)

    # Makes the plot more legible, better formatted
    figure(
        num=None,
        figsize=(
            20,
            4),
        dpi=1000,
        facecolor='w',
        edgecolor='k',
        frameon=False)
    width = 15
    plt.xticks(rotation=45)
    ax = plt.gca()

    # Ensures the plot has loglog scales
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Actually plots the word frequency over rank scatterplot and the
    # appropriate curve fit
    plt.scatter(x, y, width, color='g')
    plt.plot(x - 1, np.exp(m) * (x**(k)))

    # More visual and label plot stuff
    plt.ylabel("Word Frequency(log10)")
    plt.xlabel("Word Rank(log10)")
    plt.title(
        "Word Frequency Analysis of Top 500 Poem Hunter Poems, Descending Order, With Line of Best Fit")
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    # Using String Comprehension to plot r^2 value on legend
    plt.legend([f'Line Of best Fit, R-Square = {r_sq}'])
