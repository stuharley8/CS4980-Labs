import pandas as pd
import re
import time

regex = r'[_\'\".?!,():;-]'


def create_vocabulary(documents):
    vocabulary = set()
    for doc in documents:
        for line in doc.readlines():
            words = line.split()
            for word in words:
                word = re.sub(regex, '', word.lower())   # Remove punctuation, make all lowercase
                vocabulary.add(word)
    vocabulary.remove('pos')
    vocabulary.remove('neg')
    vocabulary.add('COUNT')
    vocabulary.add('TOTAL')
    return vocabulary


def add_lexicon_to_vocabulary(vocabulary, document):
    for line in document.readlines():
        words = line.replace('=', ' ').split()
        words[5] = re.sub(regex, '', words[5].lower())
        vocabulary.add(words[5])  # 6th "word" in each line of the lexicon is the actual word


def create_vocabulary_amazon(document):
    vocabulary = set()
    l = 0
    with open(document, 'r', encoding='utf8', errors='ignore') as doc:
        for line in doc.readlines():
            l = l + 1
            words = line.split()
            for word in words:
                word = re.sub(regex, '', word.lower())  # Remove punctuation, make all lowercase
                vocabulary.add(word)
            if l % 10000 == 0:
                print('line', l, 'added to vocab')
    # vocabulary.remove('label1')
    # vocabulary.remove('label2')
    print(len(vocabulary), 'words in vocabulary')
    vocabulary.add('COUNT')
    vocabulary.add('TOTAL')
    return vocabulary


def count_words_in_classes(vocabulary, documents):
    df = pd.DataFrame(0, index=vocabulary, columns=['pos', 'neu', 'neg'])
    count_pos = 0
    count_neu = 0
    count_neg = 0
    for doc in documents:
        for line in doc.readlines():
            words = line.split()
            if words[0] == 'POS':
                count_pos = count_pos + 1
                for word in words[1:]:
                    word = re.sub(regex, '', word.lower())
                    df['pos'][word] = df['pos'][word] + 1
            elif words[0] == 'NEU':
                count_neu = count_neu + 1
                for word in words[1:]:
                    word = re.sub(regex, '', word.lower())
                    df['neu'][word] = df['neu'][word] + 1
            else:
                count_neg = count_neg + 1
                for word in words[1:]:
                    word = re.sub(regex, '', word.lower())
                    df['neg'][word] = df['neg'][word] + 1
    df['neg']['TOTAL'] = df['neg'].sum()
    df['neu']['TOTAL'] = df['neu'].sum()
    df['pos']['TOTAL'] = df['pos'].sum()
    df['neg']['COUNT'] = count_neg
    df['neu']['COUNT'] = count_neu
    df['pos']['COUNT'] = count_pos
    return df


def count_words_in_classes_lexicon(df, document):
    count_pos = df['pos']['COUNT']
    count_neu = df['neu']['COUNT']
    count_neg = df['neg']['COUNT']
    for line in document.readlines():
        words = line.replace('=', ' ').split()
        words[5] = re.sub(regex, '', words[5].lower())
        if words[-1] == 'positive':
            count_pos = count_pos + 1
            df['pos'][words[5]] = df['pos'][words[5]] + 1
        elif words[-1] == 'neutral':
            count_neu = count_neu + 1
            df['neu'][words[5]] = df['neu'][words[5]] + 1
        elif words[-1] == 'negative':
            count_neg = count_neg + 1
            df['neg'][words[5]] = df['neg'][words[5]] + 1
    df['neg']['COUNT'] = 0
    df['neu']['COUNT'] = 0
    df['pos']['COUNT'] = 0
    df['neg']['TOTAL'] = df['neg'].sum()
    df['neu']['TOTAL'] = df['neu'].sum()
    df['pos']['TOTAL'] = df['pos'].sum()
    df['neg']['COUNT'] = count_neg
    df['neu']['COUNT'] = count_neu
    df['pos']['COUNT'] = count_pos


def count_words_in_classes_amazon(vocabulary, document):
    df = pd.DataFrame(0, index=vocabulary, columns=['pos', 'neg'])
    count_pos = 0
    count_neg = 0
    l = 0
    with open(document, 'r', encoding='utf8', errors='ignore') as doc:
        for line in doc.readlines():
            l = l + 1
            words = line.split()
            if words[0] == '__label__2':    # label2 = Positive
                count_pos = count_pos + 1
                for word in words[1:]:
                    word = re.sub(regex, '', word.lower())  # Remove punctuation, make all lowercase
                    df['pos'][word] = df['pos'][word] + 1
            else:                       # label1 = Negative
                count_neg = count_neg + 1
                for word in words[1:]:
                    word = re.sub(regex, '', word.lower())  # Remove punctuation, make all lowercase
                    df['neg'][word] = df['neg'][word] + 1
            if l % 10000 == 0:
                print('line', l, 'counts added')
    df['neg']['COUNT'] = 0
    df['pos']['COUNT'] = 0
    df['neg']['TOTAL'] = df['neg'].sum()
    df['pos']['TOTAL'] = df['pos'].sum()
    df['neg']['COUNT'] = count_neg
    df['pos']['COUNT'] = count_pos
    print(df['pos']['COUNT'], 'positive documents')
    print(df['neg']['COUNT'], 'negative documents')
    print(df['pos']['TOTAL'], 'positive words in table')
    print(df['neg']['TOTAL'], 'negative words in table')
    return df


def classifySentence(sentence, df):
    total_docs = df.loc['COUNT'].sum()
    prob_neg = df['neg']['COUNT'] / total_docs
    prob_neu = df['neu']['COUNT'] / total_docs
    prob_pos = df['pos']['COUNT'] / total_docs
    total_words = df.loc['TOTAL'].sum()
    words = sentence.split()
    for word in words:
        try:
            word = re.sub(regex, '', word.lower())
            prob_neg = prob_neg * ((df['neg'][word] + 1) / (df['neg']['TOTAL'] + total_words))
            prob_neu = prob_neu * ((df['neu'][word] + 1) / (df['neu']['TOTAL'] + total_words))
            prob_pos = prob_pos * ((df['pos'][word] + 1) / (df['pos']['TOTAL'] + total_words))
        except:     # ignores Out-Of-Vocabulary terms
            pass
    if max(prob_pos, prob_neu, prob_neg) == prob_pos:
        classify = 'Positive'
    elif max(prob_pos, prob_neu, prob_neg) == prob_neu:
        classify = 'Neutral'
    else:
        classify = 'Negative'
    print(sentence[:-1], ':', classify)
    # print('Prob Positive:', prob_pos)
    # print('Prob Neutral:', prob_neu)
    # print('Prob Negative:', prob_neg)
    # print()


def classify_amazon(df, document):
    cm = pd.DataFrame(0, index=['act_pos', 'act_neg'], columns=['pred_pos', 'pred_neg'])
    total_docs = df.loc['COUNT'].sum()
    prob_pos = df['pos']['COUNT'] / total_docs
    prob_neg = df['neg']['COUNT'] / total_docs
    total_words = df.loc['TOTAL'].sum()
    l = 0
    with open(document, 'r', encoding='utf8', errors='ignore') as doc:
        for line in doc.readlines():
            l = l + 1
            words = line.split()
            if words[0] == '__label__2':    # __label__2 is positive
                actual = 'act_pos'
            else:                           # __label__1 is negative
                actual = 'act_neg'
            for word in words[1:]:
                try:
                    word = re.sub(regex, '', word.lower())
                    prob_word_neg = prob_neg * ((df['neg'][word] + 1) / (df['neg']['TOTAL'] + total_words))
                    prob_word_pos = prob_pos * ((df['pos'][word] + 1) / (df['pos']['TOTAL'] + total_words))
                except:  # ignores Out-Of-Vocabulary terms
                    pass
            if prob_word_pos > prob_word_neg:
                classify = 'pred_pos'
            else:
                classify = 'pred_neg'
            cm[classify][actual] = cm[classify][actual] + 1
            if l % 10000 == 0:
                print('line', l, 'sentences classified')
            # print(line, ':', classify)
            # print('Prob Positive:', prob_word_pos)
            # print('Prob Negative:', prob_word_neg)
            # print()
    return cm


def main():
    """
    # The following lines use the training set and lexicon files to classify the test set
    vocabulary = create_vocabulary([open('trainingSet.txt', 'r')])
    add_lexicon_to_vocabulary(vocabulary, open('lexicon.txt', 'r'))
    df = count_words_in_classes(vocabulary, [open('trainingSet.txt', 'r')])
    count_words_in_classes_lexicon(df, open('lexicon.txt', 'r'))
    # print(df.to_string(), '\n')
    testSet = open('testSet.txt', 'r')
    for line in testSet.readlines():
        classifySentence(line, df)
    """
    # The following lines use the amazon training set to classify the amazon test set
    # This code was not optimized for performance. It took ~9.5 hours to run locally on my laptop
    start = time.time()
    vocabulary = create_vocabulary_amazon('AmazonTrain.txt')
    print()
    df = count_words_in_classes_amazon(vocabulary, 'AmazonTrain.txt')
    print()
    cm = classify_amazon(df, 'AmazonTest.txt')
    print()
    print(time.time() - start, 'seconds')
    print()
    print(cm.to_string(), '\n')


if __name__ == "__main__":
    main()
