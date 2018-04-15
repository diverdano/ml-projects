#!usr/bin/env python

# == load libraries ==
import util
import util_data
import util_plot

# == load data ==
# s&p 500 symbols
def getSP500():
    '''use pandas to parse wikipedia S&P 500 web pages'''
    url         = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response    = util.requests.get(url)
    result      = util_data.pd.read_html(response.content)      # currently page contains "component stocks" & "recent changes"
#     for df in result:
#         df.columns  = df.iloc[0]                                # first row contains the column names
# #        df.reindex(df.index.drop(0))
    return result

class PBGCrime(object):
    '''get crime statistics for Palm Beach County crime from website'''
    url         = 'http://www.city-data.com/crime/crime-Palm-Beach-Gardens-Florida.html'
    def __init__(self):
        pass
    def getStats(self):
        self.response   = util.requests.get(self.url)
#        self.status      = self.response.status_code
    def clenseStats(self):
        '''table 7 is the relevant stats table'''
        self.stats      = util_data.pd.read_html(self.response.content)[7]


class RegVoters(object):
    '''get list of voters from voterecords.com using street (list) and cityst'''
    domain      = 'https://voterrecords.com/street/'
    cityst      = '-west+palm+beach-fl/'
    filename    = 'projects/preserve/voters.csv'
    co_streets  = [
        "oldham+way",
        "wharton+way"]
    streets     = [
        "bay+hill+dr",
        "blackwoods+ln",
        "buckhaven+ln",
        "carnegie+pl",
        "dunbar+ct",
        "eagles+xing",
        "gullane+ct",
        "keswick+way",
        "leeth+ct",
        "littlestone+ct",
        "marlamoor+ln",
        "northlake+blvd",
        "riverchase+run",
        "sanbourn+ct",
        "stonehaven+way",
        "torreyanna+cir"]
    url_sets        = []
    test_urls       = []
    test_results    = []
    pages           = {}
    people          = []
    def __init__(self, run=True, filename=None):
        if run:
            self.testURLs()
            self.setupURLs()
            self.getVoters()
            self.mergeVoters()
        else:
            if filename:
                self.voters = util_data.pd.read_csv(filename)
            else:
                self.voters = util_data.pd.read_csv(self.filename)
                self.voters.drop(self.voters[[0]], axis=1, inplace=True)   # drop index column
                self.cleansVoters()
    def testURLs(self):
        '''iterate streets, creating test urls, validate http status'''
        print('scraping pages and people counts')
        for street in self.streets:
            url = self.domain+street+self.cityst+"/1"
            self.test_urls.append(url)
            response    = util.requests.get(url)
            status      = response.status_code
            char_loc    = response.text.find('Page 1 of ')                        # find pages
            people_loc  = response.text.find(' people who live')
            if char_loc == -1   : pages = '1'       # keep string value, used in url
            else                : pages = response.text[char_loc:char_loc + 12].split('<')[0].split(' of ')[1]    # propably better way to do this
            people_count = int(response.text[people_loc - 10:people_loc].split(' are ')[1])
            self.pages[street] = {'pages':pages, 'people':people_count}
            print("\tstatus: {0} | {1} | {2} | {3}".format(str(status), pages, people_count, street))
        return self.pages
    def setupURLs(self):
        '''iterate streets, creating urls'''
        print('setting up urls for each street')
        for street in self.streets:
            if self.pages[street]['people']>0 :
                included=True
                pages   = list(range(1,int(self.pages[street]['pages'])+1)) # assumes 9 pages of names for each street
                self.url_sets.append([self.domain+street+self.cityst+str(page) for page in pages])
            else: included=False
            inc_skip = 'included' if included == True else '--skipped--'
            print('\t{}'.format(street).ljust(17) + inc_skip)
    def getVoters(self):
        print('scraping people from web pages')
        for i, urls in enumerate(self.url_sets):
            print('\n\t' + self.streets[i-1])
            for j, url in enumerate(urls):
                print(str(j), sep=' ', end='', flush=True)
                self.people.append(util_data.pd.read_html(util.requests.get(url).content))
    def mergeVoters(self):
        '''apply data merge to scraped voter records'''
        self.voters = util_data.pd.concat([self.people[item][1] for item in list(range(len(self.people)))]) # drop summary tables and concat list of dfs to one df
        self.voters.rename(columns=lambda x: x.strip(), inplace=True)         # strip spaces in column names
        self.voters.drop(self.voters[[3]], axis=1, inplace=True)   # drop extra column
    def cleansVoters(self):
        '''apply data scrubbing to scraped voter records'''
        self.voters[['name','age']] = self.voters.Person.str.split('  ', expand=True)
        self.voters.age.replace(regex=True, inplace=True, to_replace=r'[\(\)]', value=r'')
        self.voters.drop(self.voters[['Person']], axis=1, inplace=True)
        self.voters[['number','street']] = self.voters.Address.str.split(' ', n=1, expand=True)
        self.voters['street'].replace(regex=True, inplace=True, to_replace=r' West Palm Beach, Fl 33412', value=r'')
        self.voters.drop(self.voters[['Address']], axis=1, inplace=True)
        cols = ['age','number']
        self.voters[cols] = self.voters[cols].apply(util_data.pd.to_numeric, errors='coerce', axis=1)
        # self.voters.age = util_data.pd.to_numeric(self.voters.age)
    def setAddress(self):
        '''concat number and street, mapping number to a string'''
        self.voters['address'] = self.voters.number.map(str) + ' ' + self.voters.street
