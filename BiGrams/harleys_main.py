import re
import random
import pandas as pd


def generate_wordlist(corpora):
    """
    Generates a wordlist from a corpora. Removes punctuation except for periods and changes all words to lowercase
    :param corpora: the corpora txt file
    :return: the wordlist
    """
    file = open(corpora, 'r')
    words = []
    for line in file.readlines():
        words = words + (line.split())
    words_periods = []
    for word in words:
        word = re.sub(r'[\'\",():;-]', '', word.lower())    # Remove punctuation (not periods), make all lowercase
        if word.endswith('.'):      # Split periods from words to signify end of sentence characters
            words_periods.append(word[0:-1])
            words_periods.append('.')
        elif word != '':
            words_periods.append(word)
    return words_periods


def create_grams(wordlist, gram_type):
    """
    Creates a list of unigrams or bigrams from a given wordlist
    :param wordlist: the wordlist
    :param gram_type: whether the user wants unigrams ('1') or bigrams ('2')
    :return: A list of the unigrams
    """
    if gram_type == '1':
        grams = create_unigrams(wordlist)
    else:
        grams = create_bigrams(wordlist)
    return grams


def create_unigrams(words):
    """
    Creates a list of unigrams from a given wordlist
    :param words: the wordlist
    :return: a dictionary representing the unigrams. Key is the word. Value is it's probability of occuring.
    """
    gram_dict = {}
    for word in words:
        gram_dict[word] = gram_dict.get(word, 0) + 1
    for key, value in gram_dict.items():
        gram_dict[key] = gram_dict[key] / len(words)
    print(len(gram_dict), 'unigrams created.')
    return gram_dict


def create_bigrams(words):
    """
    Creates a pandas dataframe representing the bigrams from a given wordlist
    :param words: the wordlist
    :return: a pandas dictionary containing the bigram probabilities
    """
    num_words = len(words)
    word_set = set(words)
    df = pd.DataFrame(0, index=word_set, columns=word_set)
    df[words[0]]['.'] = 1       # Make first word follow a period. Set appropriate cell to 1
    for x in range(1, num_words):
        df[words[x]][words[x-1]] = df[words[x]][words[x-1]] + 1     # Increment each appropriate cell value by 1
    return df.div(df.sum(axis=1), axis=0)       # Divide each column by the corresponding row sums to get probs


def print_random_sentence(grams, gram_type):
    """
    Creates and prints a random sentence
    :param grams: the data structure of the n-grams
    :param gram_type: whether the user wants unigrams ('1') or bigrams ('2')
    """
    if gram_type == '1':
        print_random_sentence_unigram(grams)
    else:
        print_random_sentence_bigram(grams)


def print_random_sentence_unigram(grams):
    """
    Generates and prints a random sentence created from unigrams
    :param grams: a map of unigrams
    """
    sentence = ''
    word = ''
    while word != '.':          # Chooses a word according to the probabilities. Terminates on a '.'
        num = random.random()
        prob = 0
        for key, value in grams.items():
            prob = prob + value
            if prob > num:
                word = key
                break
        if word == '.':
            sentence = sentence + word
        else:
            sentence = sentence + ' ' + word
    print(sentence.strip())


def print_random_sentence_bigram(grams):
    """
    Generates and prints a random sentence created from bigrams
    :param grams: a pandas dataframe of bigrams
    """
    word = ''
    num = random.random()
    prob = 0
    row = grams.loc['.']        # First word is chosen based on the previous word being '.'
    for x in range(row.size):
        prob = prob + row[x]
        if prob > num:
            word = row.index[x]
            break
    sentence = word
    while word != '.':          # Next words are chosen based off the previous word. Terminates if a '.' is selected
        num = random.random()
        prob = 0
        row = grams.loc[word]
        for x in range(row.size):
            prob = prob + row[x]
            if prob > num:
                word = row.index[x]
                break
        if word == '.':
            sentence = sentence + word
        else:
            sentence = sentence + ' ' + word
    print(sentence)


def main():
    corpora = input('Enter a corpora: ')
    words = generate_wordlist(corpora)
    print(len(words), 'total words in text file.')
    gram_type = input('Do you want to create unigrams (1) or bigrams (2): ')
    grams = create_grams(words, gram_type)
    # print(grams.to_string(), '\n')
    choice = input('Do you want to create a random sentence (y/n)? ')
    while choice == 'y':
        print_random_sentence(grams, gram_type)
        choice = input('Do you want to create another random sentence (y/n)? ')
    return


if __name__ == "__main__":
    main()
