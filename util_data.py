#!usr/bin/env python

# == load libraries ==

# key libraries
import pandas as pd
import numpy as np
import simplejson as json
import csv                                      # for csv sniffer
import logging
import html5lib

# data sets
from sklearn.datasets import load_linnerud      # linear regression data set
from sklearn.datasets import load_digits        # learning curve data set
import quandl

# == set logging ==
logger = logging.getLogger(__name__)
#logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

# == data ==

def readHTML2df(html):      # requires html5lib
    '''load html table data to pandas DataFrame'''
    return pd.read_html(html)

def getQuandl(symbol):
    '''access quandl data via api
    examples:
        vix futures: "CHRIS/CBOE_VX1", "CHRIS/CBOE_VX2"
        s&p 500 e-mini: "CME/ESU2018"'''
    return quandl.get(symbol)

def mergeDFs(df1, df2, df1_cols, df2_cols, join_type='outer'):
    '''merge df1 and df2 on df1_columns (list) and df2_columns (list) using join type'''
    df3 = pd.merge(df1, df2, left_on=df1_cols, right_on=df2_cols, how=join_type)
    return df3

def crosstabDF(df):
    ''' create cross tab of two attributes, remove zeros for readability '''
    return(pd.crosstab(tx.Offer, tx.Cust_ln).apply(lambda x: x, axis=1).replace(0,''))

def obj2float(df, columns):
    '''convert dataframe columns from object to float'''
    for column in columns:
        df.column = df.column.str.replace(',','').astype(float)

def isolateMissing():
    '''isolate columns that aren't mapping'''
    return df3[df3.isnull().any(axis=1)][['name','company']]

def getFirstLast(name):
    '''separates the first and last names'''
    if isinstance(name, str):
        name        = name.split()
    else:
        name    = ('na','na')
    return{"firstname":name[0], "lastname":name[-1]}
    # firstName   = name[0]
    # lastName    = name[-1]
    # # now figure out the first initial, we're assuming that if it has a dot it's an initialized name, but this may not hold in general
    # if "." in firstName:
    #     firstInitial = firstName
    # else:
    #     firstInitial = firstName[0] + "."
    # lastName = name[2]
    # return {"FirstName":firstName, "FirstInitial":firstInitial, "LastName": lastName}

# == test functions ==
# def SniffDelim(file):
#     '''use csv library to sniff delimiters'''
#     with open(file, 'r') as infile:
#         dialect = csv.Sniffer().sniff(infile.read())
#     return dict(dialect.__dict__)

# == helper functions ==

def strFormat(text, color):
    key = {
        'head'  : "\x1b[",
        'end'   : "\x1b[0m",
        'red'   : "0;30;41m",
        'green' : "0;30;42m",
        'orange': "0;30;43m",
        'blue'  : "0;30;44m",
        'purple': "0;30;45m",
        'gold'  : "0;30;46m",
        'white' : "0;30;47m"
    }
    return (key['head'] + key[color] + text + key['end'])

def setDF(w=None, c=None, r=None, f='{:,.1f}'):
    '''set width, max columns and rows'''
    pd.set_option('display.width', w)               # show columns without wrapping
    pd.set_option('display.max_columns', c)         # show all columns without elipses (...)
    pd.set_option('display.max_rows', r)            # show default number of rows for summary
    pd.options.display.float_format = f.format
    np.set_printoptions(formatter={'float': lambda x: f.format(x)})
def summarizeData(desc, data):
    '''summarize dataframe data, separate data summary/reporting'''     # test if this works for data set that is not a dict
    logger.info("\n\n" + strFormat(" {} ".format(desc),'green') +
        strFormat(" dataset has {} records with {} features".format(*data.shape),'white') + "\n")
    for index, item in enumerate(sorted(data.columns)): logger.info("\t{}\t'{}'".format(index + 1, item))
    logger.info("\n\n== DataFrame Description ==\n\n" + str(data.describe(include='all')) + "\n\n")
    # logger.info(data.describe(include='all'))
    logger.info("\n\n== DataFrame, head ==\n\n" + str(data.head()) + "\n\n")


def dfCol2Numeric(df, cols):
    '''use pandas apply to convert columns to numeric type'''
    # cols = df.columns.drop('id')
    # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df[cols].apply(pd.to_numeric, errors='coerce')

# == i/o ==

def sniffDelim(file):
    '''helper functionuse csv library to sniff delimiters'''
    with open(file, 'r') as infile:
        dialect = csv.Sniffer().sniff(infile.read())
    return dict(dialect.__dict__)

def readFile(file):
    '''simple read file'''
    with open(file, 'r') as infile: data=infile.read()
    return data

def loadData(file):
    '''check file delimiter and load data set as pandas.DataFrame'''
    try:
        logger.debug('\n\tchecking delimiter')
        delimiter = sniffDelim(file)['delimiter']
        logger.debug('\tdelimiter character identified: {}'.format(delimiter))
        try:
            data = pd.read_csv(file, sep=delimiter)
            logger.debug("\tfile loaded")
            return data
        except UnicodeDecodeError:
            logger.error('\tunicode error, trying latin1')
            data           = pd.read_csv(file, encoding='latin1', sep=delimiter) # including sep in this call fails...
            logger.debug("\tfile loaded")
            return data
    except Exception as e:
        raise e

# == create data ==

def loadRegSample(self):
    ''' load regression sample dataset '''
    self.data           = load_linnerud()
    logger.info(self.data.DESCR)

def loadLearningCurveSample(self):
    ''' load learning curve sample dataset '''
    self.data           = load_digits()
    logger.info(self.data.DESCR)

def makeTerrainData(self, n_points=1000):
    '''make the toy dataset'''
    self.data = {}
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0
### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    self.data['X_train'] = X[0:split]
    self.data['X_test']  = X[split:]
    self.data['y_train'] = y[0:split]
    self.data['y_test']  = y[split:]
#     grade_sig = [self.data['X_train'][ii][0] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==0]
#     bumpy_sig = [self.data['X_train'][ii][1] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==0]
#     grade_bkg = [self.data['X_train'][ii][0] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==1]
#     bumpy_bkg = [self.data['X_train'][ii][1] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==1]
# #         training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
# #                 , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}
#     grade_sig = [self.data['X_test'][ii][0] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==0]
#     bumpy_sig = [self.data['X_test'][ii][1] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==0]
#     grade_bkg = [self.data['X_test'][ii][0] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==1]
#     bumpy_bkg = [self.data['X_test'][ii][1] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==1]
# #         test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
# #                 , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}
# #         return X_train, y_train, X_test, y_test

# == data object ==
class ProjectData(object):
    ''' get and setup data '''
    infile          = 'ml_projects.json'                # should drop target/features from json? lift from data with pd.columns[:-1] & [-1]
    outfile         = 'ml_projects.json'                # write to same file, use git to manage versions
    DF              = {}
    def __init__(self, project='boston_housing', file=None):
        if file:
            self.desc           = 'file used'
            self.file           = file
            self.data           = loadData(file)
        else:
            try:
                self.loadProjects()
                if project in self.projects.keys():
                    self.desc       = project # if exists project in self.projects ...
                    if 'files' in self.projects[self.desc]:
                        self.files  = self.projects[self.desc]['files']
                        for file in self.files.keys():
                            try:
                                self.DF[file]   = loadData(self.files[file])
                                logger.debug("file is: {}".format(file))
                                setDF()
                                summarizeData(desc=file, data=self.DF[file])
                            except Exception as e:
                                logger.error("issue loading data from: {}".format(file), exc_info=True)
                                return
                        return
                    self.file       = self.projects[self.desc]['file']
                    try:
                        self.DF                 = loadData(self.file)
#                        self.loadData()
                    except Exception as e:
                        logger.error("issue loading data: ", exc_info=True)
                        return
                    if all (k in self.projects[self.desc] for k in ('target', 'features')):
                        self.target     = self.projects[self.desc]['target']        # make y or move this to data, or change reg & lc samples?
                        self.features   = self.projects[self.desc]['features']      # make X or move this to data, or change reg & lc samples?
                        self.prepData()
                        self.preprocessData()
                    else: logger.warn("'target' and 'features' need to be specified for prepping model data")
                else:
                    logger.warn('"{}" project not found; list of projects:\n'.format(project))
                    logger.warn("\t" + "\n\t".join(sorted(list(self.projects.keys()))))
            except Exception as e: # advanced use - except JSONDecodeError?
                logger.error("issue reading project file:", exc_info=True)
    def loadProjects(self):
        ''' loads project meta data from file '''
        with open(self.infile) as file: self.projects  = json.load(file)
    def saveProjects(self):
        ''' saves project meta detail to file '''
        with open(self.outfile, 'w') as outfile: json.dump(self.projects, outfile, indent=4)
    def joinDataSets(self):
        ''' combine transactional data and meta attributes into a single array'''
        pass
    def prepData(self):
        '''split out target and features based on known column names in project meta data'''
        self.y              = self.DF[self.target]
        self.X              = self.DF.drop(self.target, axis = 1)
    def preprocessData(self):
        '''transpose objects to numerical data -> binary where appropriate '''
        # convert yes/no to 1/0
        logger.info("\n\tprocessing target/y variable")
        logger.info("\n\tpreprocessing X & y, inputs and target values, replacing yes/no with 1/0")
        if self.y.dtype == object:          self.y.replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        logger.info("\t\ty (target) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X[col].replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        # use separate for loop to complete in place changes before processing 'get_dummies'
        logger.info("\t\tX (input) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X = self.X.join(pd.get_dummies(col_data, prefix = col))
        logger.info("\tconverted categorical variable into dummy/indicator variables")
        # cast float64 to int64 for memory use and ease of reading (issue with get_dummies)
        for col in self.X.select_dtypes(['float64']):
            self.X[col] = self.X[col].astype('int64')
        logger.info("\tconverted float to integer")
        # if all else fails: _get_numeric_data()       # limit to numeric data
        # remove remaining object columns
        for col in self.X.select_dtypes(['O']):
            del self.X[col]
        logger.info("\tremoved initial columns, now that they have been converted")
        self.all        = pd.concat((self.X, self.y), axis=1)
        logger.info("\tcreated 'all' dataframe, adding target as final column")
        self.features   = list(self.X.columns)
        self.label      = self.y.name
        logger.info("\n\tTarget/Label Description (numerical attribute statistics)\n")
        logger.info(self.y.describe())
