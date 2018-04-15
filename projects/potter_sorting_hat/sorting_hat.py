# Import modules
import pandas as pd
import requests
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import copy
from itertools import combinations

def get_forms(lemma):
    ''' function to get different word forms from a lemma, need to bump up against dictionary to drop improper conjugations '''
    drfs = lemma.derivationally_related_forms()
    output_list = []
    if drfs:
        for drf in drfs:
            drf_pos = str(drf).split(".")[1]
            if drf_pos in ['n', 's', 'a']:
                output_list.append(drf.name().lower())
                if drf_pos in ['s', 'a']:
                    # Adverbs + "-ness" nouns + comparative & superlative adjectives
                    if len(drf.name()) == 3:
                        last_letter = drf.name()[-1:]
                        output_list.append(drf.name().lower() + last_letter + 'er')
                        output_list.append(drf.name().lower() + last_letter + 'est')
                        output_list.append(drf.name().lower()+'ness')
                        output_list.append(drf.name().lower()+'ly')
                    elif drf.name()[-4:] in ['able', 'ible']:
                        output_list.append(drf.name().lower()+'r')
                        output_list.append(drf.name().lower()+'st')
                        output_list.append(drf.name().lower()+'ness')
                        output_list.append(drf.name()[:-1].lower()+'y')
                    elif drf.name()[-1:] == 'e':
                        output_list.append(drf.name().lower()+'r')
                        output_list.append(drf.name().lower()+'st')
                        output_list.append(drf.name().lower()+'ness')
                        output_list.append(drf.name().lower()+'ly')
                    elif drf.name()[-2:] == 'ic':
                        output_list.append(drf.name().lower()+'er')
                        output_list.append(drf.name().lower()+'est')
                        output_list.append(drf.name().lower()+'ness')
                        output_list.append(drf.name().lower()+'ally')
                    elif drf.name()[-1:] == 'y':
                        output_list.append(drf.name()[:-1].lower()+'ier')
                        output_list.append(drf.name()[:-1].lower()+'iest')
                        output_list.append(drf.name()[:-1].lower()+'iness')
                        output_list.append(drf.name()[:-1].lower()+'ily')
                    else:
                        output_list.append(drf.name().lower()+'er')
                        output_list.append(drf.name().lower()+'est')
                        output_list.append(drf.name().lower()+'ness')
                        output_list.append(drf.name().lower()+'ly')
        return output_list
    else:
        return output_list

# Tests that the trait dictionary and the antonym dictionary don't have any repeats among houses
def testOverlap(dict):
    results = []
    house_combos = combinations(list(dict.keys()), 2)
    for combo in house_combos:
        results.append(set(dict[combo[0]]).isdisjoint(dict[combo[1]]))
    return results

# sl_flat = [item for subset in [synset.lemmas() for synset in sh.relevant_synsets['Slytherin']] for item in subset]
# gl_flat = [item for subset in [synset.lemmas() for synset in sh.relevant_synsets['Gryffindor']] for item in subset]
# hl_flat = [item for subset in [synset.lemmas() for synset in sh.relevant_synsets['Hufflepuff']] for item in subset]
# rl_flat = [item for subset in [synset.lemmas() for synset in sh.relevant_synsets['Ravenclaw']] for item in subset]
# 
# sl = [synset.lemmas() for synset in sh.relevant_synsets['Slytherin']]
# gl = [synset.lemmas() for synset in sh.relevant_synsets['Gryffindor']]
# hl = [synset.lemmas() for synset in sh.relevant_synsets['Hufflepuff']]
# rl = [synset.lemmas() for synset in sh.relevant_synsets['Ravenclaw']]
# [item for sublist in l for item in sublist]

class SortingHat(object):
    ''' create the sorting hat8 '''
    # Set variables
    houses          = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    site_url        = "http://harrypotter.wikia.com/api/v1/Articles/List?expand=1&limit=1000&category="
    site_articles   = 'http://harrypotter.wikia.com/api/v1/Articles/AsSimpleJson?id='
    df              = pd.DataFrame()
    trait_dict      = {'Gryffindor'     : ['bravery', 'nerve', 'chivalry', 'daring', 'courage'],
                       'Slytherin'      : ['resourcefulness', 'cunning', 'ambition', 'determination', 'self-preservation', 'fraternity', 'cleverness'],
                       'Ravenclaw'      : ['intelligence', 'wit', 'wisdom', 'creativity', 'originality', 'individuality', 'acceptance'],
                       'Hufflepuff'     : ['dedication', 'diligence', 'fairness', 'patience', 'kindness', 'tolerance', 'persistence', 'loyalty']}
    # Creates a copy of our trait dictionary, If we don't do this, then we constantly update the dictariony we are looping through, causing an infinite loop
    new_trait_dict  = copy.deepcopy(trait_dict)
    antonym_dict    = {}

# Manually select the synsets that are relevant to us ## shouldn't this be created dynamically...???
    relevant_synsets = {'Ravenclaw'     : [ wn.synset('intelligence.n.01'),
                                            wn.synset('wit.n.01'),
                                            wn.synset('brain.n.02'),
                                            wn.synset('wisdom.n.01'),
                                            wn.synset('wisdom.n.02'),
                                            wn.synset('wisdom.n.03'),
                                            wn.synset('wisdom.n.04'),
                                            wn.synset('creativity.n.01'),
                                            wn.synset('originality.n.01'),
                                            wn.synset('originality.n.02'),
                                            wn.synset('individuality.n.01'),
                                            wn.synset('credence.n.01'),
                                            wn.synset('acceptance.n.03')],
                        'Hufflepuff'    : [ wn.synset('dedication.n.01'),
                                            wn.synset('commitment.n.04'),
                                            wn.synset('commitment.n.02'),
                                            wn.synset('diligence.n.01'),
                                            wn.synset('diligence.n.02'),
                                            wn.synset('application.n.06'),
                                            wn.synset('fairness.n.01'),
                                            wn.synset('fairness.n.02'),
                                            wn.synset('patience.n.01'),
                                            wn.synset('kindness.n.01'),
                                            wn.synset('forgivingness.n.01'),
                                            wn.synset('kindness.n.03'),
                                            wn.synset('tolerance.n.03'),
                                            wn.synset('tolerance.n.04'),
                                            wn.synset('doggedness.n.01'),
                                            wn.synset('loyalty.n.01'),
                                            wn.synset('loyalty.n.02')],
                        'Gryffindor'    : [ wn.synset('courage.n.01'),
                                            wn.synset('fearlessness.n.01'),
                                            wn.synset('heart.n.03'),
                                            wn.synset('boldness.n.02'),
                                            wn.synset('chivalry.n.01'),
                                            wn.synset('boldness.n.01')],
                        'Slytherin'     : [ wn.synset('resourcefulness.n.01'),
                                            wn.synset('resource.n.03'),
                                            wn.synset('craft.n.05'),
                                            wn.synset('cunning.n.02'),
                                            wn.synset('ambition.n.01'),
                                            wn.synset('ambition.n.02'),
                                            wn.synset('determination.n.02'),
                                            wn.synset('determination.n.04'),
                                            wn.synset('self-preservation.n.01'),
                                            wn.synset('brotherhood.n.02'),
                                            wn.synset('inventiveness.n.01'),
                                            wn.synset('brightness.n.02'),
                                            wn.synset('ingenuity.n.02')]}

    # Add synonyms and word forms to the (new) trait dictionary; also add antonyms (and their word forms) to the antonym dictionary
    for house, traits in trait_dict.items():
        antonym_dict[house] = []
        for trait in traits:
            synsets = wn.synsets(trait, pos=wn.NOUN)
            for synset in synsets:
                if synset in relevant_synsets[house]:
                    for lemma in synset.lemmas():
                        new_trait_dict[house].append(lemma.name().lower())
                        if get_forms(lemma):
                            new_trait_dict[house].extend(get_forms(lemma))
                        if lemma.antonyms():
                            for ant in lemma.antonyms():
                                antonym_dict[house].append(ant.name().lower())
                                if get_forms(ant):
                                    antonym_dict[house].extend(get_forms(ant))
        new_trait_dict[house]   = sorted(list(set(new_trait_dict[house])))
        antonym_dict[house]     = sorted(list(set(antonym_dict[house])))            # no antonyms found for slytherin lemmas...

    def __init__(self):
        self.prepTests()
        for house in self.houses:
            url                 = self.site_url + house + 's'
            requested_url       = requests.get(url)
            json_results        = requested_url.json()
            info                = json_results['items']
            house_df            = pd.DataFrame(info)
            house_df            = house_df[house_df['type'] == 'article']
            house_df.reset_index(drop=True, inplace=True)
            house_df.drop(['abstract', 'comments', 'ns', 'original_dimensions', 'revision', 'thumbnail', 'type'], axis=1, inplace=True)
            house_df['house']   = pd.Series([house]*len(house_df))
            self.df             = pd.concat([self.df, house_df])
        self.df.reset_index(drop=True, inplace=True)
        self.printArticleSummary()
        self.getArticles()          # get's article details
    def prepTests(self):
        # Print some of our results
        for house in self.houses:
            print("\n\t{} traits: {}".format(house, self.new_trait_dict[house]))
            print("\n\t{} anti-traits: {}".format(house, self.antonym_dict[house]))
#        print("Gryffindor traits: {}".format(new_trait_dict['Gryffindor']))
#        print("Gryffindor anti-traits: {}".format(antonym_dict['Gryffindor']))

        # Outputs results from our test; should output "False"
        print("\tAny words overlap in trait dictionary? {}".format(sum(testOverlap(self.new_trait_dict)) != 6))
        print("\tAny words overlap in antonym dictionary? {}".format(sum(testOverlap(self.antonym_dict)) != 6))
    def printArticleSummary(self):
        print('\tNumber of student articles: {}'.format(len(self.df)))
        print('')
        print(self.df.head())
        print('')
        print(self.df.tail())
    def getArticles(self):
        ''' Gets article ids, url, and house. Loops through student articles, pulls "Personality and traits" section ... This takes a few minutes to run'''
        print('\n\tgetting articles...')
        text_dict = {}
        for iden in self.df['id']:
            url             = self.site_articles + str(iden)
            requested_url   = requests.get(url)
            json_results    = requested_url.json()
            sections        = json_results['sections']
            contents        = [sections[i]['content'] for i, x in enumerate(sections) if sections[i]['title'] == 'Personality and traits']
            if contents:
                paragraphs  = contents[0]
                texts       = [paragraphs[i]['text'] for i, x in enumerate(paragraphs)]
                all_text    = ' '.join(texts)
            else:
                all_text    = ''
            text_dict[iden] = all_text
        print('\tretrieved {} articles'.format(len(text_dict)))

        #Places data into a DataFrame and computes the length of the "Personality and traits" section
        text_df                = pd.DataFrame.from_dict(text_dict, orient='index')
        text_df.reset_index(inplace=True)
        text_df.columns        = ['id', 'text']
        text_df['text_len']    = text_df['text'].map(lambda x: len(x))

        #Merges our text data back with the info about the students
        self.mydf_all               = pd.merge(self.df, text_df, on='id')
        self.mydf_all.sort_values('text_len', ascending=False, inplace=True)

        #Creates a new DataFrame with just the students who have a "Personality and traits" section
        self.mydf_relevant          = self.mydf_all[self.mydf_all['text_len'] > 0]

        # Turns off a warning
        pd.options.mode.chained_assignment = None
        # run sort student as mapped lambda function
        self.mydf_relevant['new_house'] = self.mydf_relevant['text'].map(lambda x: self.sort_student(x))

        print('\tNumber of useable articles: {}'.format(len(self.mydf_relevant)))
        print('')
        print(self.mydf_relevant.head())
        print("\n\tMatch rate: {}".format(sum(self.mydf_relevant['house'] == self.mydf_relevant['new_house']) / len(self.mydf_relevant)))
        print("\tPercentage of ties: {}".format(sum(self.mydf_relevant['new_house'] == 'Tie!') / len(self.mydf_relevant)))
    def sort_student(self, text):
        '''Function that sorts the students'''
        text_list = word_tokenize(text)
        text_list = [word.lower() for word in text_list]
        score_dict = {}
        for house in self.houses:
            score_dict[house] = (sum([True for word in text_list if word in self.new_trait_dict[house]]) -
                                      sum([True for word in text_list if word in self.antonym_dict[house]]))
        sorted_house = max(score_dict, key=score_dict.get)
        sorted_house_score = score_dict[sorted_house]
        if sum([True for i in score_dict.values() if i==sorted_house_score]) == 1:
            return sorted_house
        else:
            return "Tie!"
# 
# # Test our function
# print(sort_student('Alice was brave'))
# print(sort_student('Alice was British'))


