#!usr/bin/python

# === load libraries ===
# key libraries
import re
from functools import reduce

# === data ===
sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

# === test functions ===

# === helper functions ===

def BayesRule(h, tp, tn):
    '''p(h|+) test illness given test positive for illness'''
    return format(h * tp /(h * tp + (1-h) * (1-tn)),'.3')

def tokenize(phrase, ver=1):
    '''changes phrase to lowercase and removes all punctuation, except apostrophe'''
    if      ver == 1 : return re.sub("[^\w\d'\s]+",'',phrase.lower()).split()
    elif    ver == 2 : return phrase.split()
    else             : return phrase.strip().split()

def next_word_list(phrase, test_word):
    ''' sub for obtaining next words and counts '''
    next_word_dict      = {}
    for index, word in enumerate(phrase):
        if test_word == word:
            next_word = phrase[index + 1]
            if next_word in next_word_dict  : next_word_dict[next_word] += 1
            else                            : next_word_dict[next_word] = 1
    return sorted(next_word_dict.items(), key=lambda x: x[1], reverse=True)

def LaterWords(sample, word, distance, parse_ver=1):
    '''
    @param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''
    # TODO: Given a word, collect the relative probabilities of possible following words from @sample.
    phrase = tokenize(sample, parse_ver)
    count_test_word     = phrase.count(word)
    total_words         = len(phrase)
    for i in range(distance):
        result = next_word_list(phrase, word)    #need to change the word each time
        for item in result: print(result)
        word=result[0][0] 
    
    # TODO: Repeat the above process--for each distance beyond 1, evaluate the words that might come after each word, and combine them weighting by relative probability into an estimate of what might appear next.
    return result[0][0]
#    return {}       # word


def NextWordProb(sampletext,test_word, parse_ver=1):
    end_reached         = False
    next_word_dict      = {}
    phrase = tokenize(sampletext, parse_ver)
    count_test_word     = phrase.count(test_word)
    total_words         = len(phrase)
    for index, word in enumerate(phrase):
        if test_word == word:
            next_word = phrase[index + 1]
            if next_word in next_word_dict  : next_word_dict[next_word] += 1
            else                            : next_word_dict[next_word] = 1
    print(next_word_dict)
    result = sorted(next_word_dict.items(), key=lambda x: x[1], reverse=True) 
    print("{0}\twords in memo".format(total_words))
    print("{0}\toccurances of '{1}'".format(count_test_word, test_word))
    for item in result:
        print("\t{}".format(item))
    print()
    if end_reached  : print("end of memo was reached given distance provided")
    return result

def nearestWordProb(memo, test_word, distance=1, parse_ver=1):
    '''provide frequency of next word in phrase'''
    end_reached         = False
    nearest_word_sets   = []
    phrase              = tokenize(memo, parse_ver)
    phrase_dict         = {}
    total_words         = len(phrase)
    count_test_word     = phrase.count(test_word)
    for index, word in enumerate(phrase):
        if word in phrase_dict  : phrase_dict[word] += 1
        else                    : phrase_dict[word] = 1
    phrase_dict_prob    = {key:value/total_words for (key,value) in phrase_dict.items()}                 # probabilities for each word in phrase
    for item in sorted(phrase_dict.items(), key=lambda x: x[1], reverse=True):
        print("\t{}".format(item))
    for index, word in enumerate(phrase):
        if test_word == word:
            try:
                nearest_word_sets.append([phrase[index:index+distance]])
#                nearest_word_sets.append((reduce( (lambda x, y: x * y), [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)] ), phrase[index + distance]))
#                nearest_word_sets.append((format(reduce( (lambda x, y: x * y), [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)] ),'.4%'), phrase[index + distance], index))
#                nearest_word_sets[index] = (phrase[index + distance], reduce( (lambda x, y: x * y), [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)] )) # this should be done as pandas DateFrame
#                nearest_word_sets[phrase[index + distance]] = reduce( (lambda x, y: x * y), [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)] ) # this assumes only one path to word...
#                nearest_word_sets[index] = reduce( (lambda x, y: x * y), [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)] )
#                nearest_word_sets[index] = [(phrase[index + i + 1],phrase_dict_prob[phrase[index + i + 1]]) for i in range(distance)]
#                nearest_word_sets[index] = [phrase_dict_prob[phrase[index + i + 1]] for i in range(distance)]
#                nearest_word_sets[index] = [phrase[index + i + 1] for i in range(distance)]
#                reduce( (lambda x, y: x * y), test )
#                next_word = phrase[index + distance]
#                if next_word in nearest_word_dict   : nearest_word_dict[next_word] += 1
#                else                                : nearest_word_dict[next_word] = 1
            except IndexError:
                end_reached = True
    result = sorted(nearest_word_sets, reverse=True)
#    nearest_word_prob   = {key:(value, format(value/total_words,'0.2%')) for (key,value) in nearest_word_dict.items()}
    print("{0}\twords in memo".format(total_words))
    print("{0}\toccurances of '{1}'".format(count_test_word, test_word))
    for item in result: print("\t{}".format(item))
#    print(nearest_word_prob)
    for item in nearest_word_sets: print("\t{}".format(item))
    if end_reached  : print("end of memo was reached given distance provided")
    return(result)