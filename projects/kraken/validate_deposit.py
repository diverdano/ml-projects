#!/usr/env python
import pandas as pd
import numpy as np
import argparse
import logging
import simplejson as json
from database_setup import Base, Deposits, RefData, createDBsession

# -- logging --
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -- argument construction and parsing --
ap = argparse.ArgumentParser()
ap.add_argument('--app',                default='deposits_app',                 help="name of application")
ap.add_argument('--db',                 default='sqlite:///deposits.db',        help="designates the database to use")
args = ap.parse_args()
logger.info('\n\tlogging is on\n')

# -- helper functions --

def readJSONFile(file):
    '''simple read file'''
    with open(file, 'r') as infile: data=infile.read()
    return(json.loads(data))

def setDF(w=None, c=None, r=None, f='{:,.1f}'):
    '''set width, max columns and rows'''
    pd.set_option('display.width', w)               # show columns without wrapping
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', c)         # show all columns without elipses (...)
    pd.set_option('display.max_rows', r)            # show default number of rows for summary
    pd.options.display.float_format = f.format
    np.set_printoptions(formatter={'float': lambda x: f.format(x)})

# -- delete data from table --

def deleteFromtable():
    '''delect all records from deposits'''
    logger.info('deposits records: {} - deleting'.format(deposits.count()))
    session.execute('delete from deposits;')
    logger.info('deposits records: {}'.format(deposits.count()))

# -- load data to table --

def loadDF2table():
    '''load both files to table'''
    logger.info('deposits records: {} - loading'.format(deposits.count()))
    df1.to_sql('deposits', con, if_exists='append', index=False) # doesn't convert integer to bytes...
    df2.to_sql('deposits', con, if_exists='append', index=False)
    logger.info('deposits records: {}'.format(deposits.count()))
    # logger.info("deposits loaded: {}".format(str(len(list(deposits)))))

def loadDFfromtable():
    '''load df from table allowing parameter injection'''
    d = {'fields': "id, address, amount, blockhash, blockindex, blocktime,\
        category, confirmations, time, txid, vout",
        'table': "deposits"}
    s = "select {fields} from {table};"
    s.format(**d)
    return(pd.read_sql(s.format(**d), con, index_col='id'))

# def df2deposit(df):
#     '''load df records into orm class'''
#     deposits = []
#     for record in df.to_records(index=False):      # class uses autoincrement in table
#         deposits.append(Deposits(*record))          # stores bytes instead of integer...
#     logger.info('deposits created: {}'.format(len(deposits)))
#     return(deposits)
    # recs1 = df2deposit(df1)                                    # converts integer to bytes...
    # recs2 = df2deposit(df2)

# -- transform data --

def evalMaxCol(df):
    '''evaluate df, identify max column widths for string fields'''
    max_col_len = df.columns.str.len().max() + 10
    for column in df.columns:
        try:
            logger.info('column: {0}'.format(column).ljust(max_col_len) + 'dtype: {0} len: {1}'.format(df[column].dtype, df[column].str.len().max()))
        except AttributeError:
            logger.info('column: {0}'.format(column).ljust(max_col_len) + 'dtype: {0}'.format(df[column].dtype))

def evalDFcols(df):
    '''evaluate DF columns, return str length, convert list to str'''
    for column in df.columns:
        if type(df[column][0]) is list:
            df[column] = df[column].apply(','.join)             # convert list to string (these are all empty...)

def validDeposits(df):
    '''identify valid deposits - has 6 confirmations, need to de-dup responses'''
    logger.info('processing transactions')
    # time always equals timereceived for sample data
    df.drop(['timereceived'], axis=1, inplace=True)

#     airlines_df = df.groupby('AIRLINECODE', as_index=False )['AIRLINENAME'].last()
#     airlines_df.sort_values('name', inplace=True)
#     logger.info('airlines found: {:,}'.format(len(airlines_df)))
#     if airlines_df.isnull().values.any():
#         nan_airlines_df = airlines_df[pd.isnull(airlines_df).any(axis=1)]
#         logger.error(['airlines with NaN values:', nan_airlines_df])
#     if type == 'df' : return airlines_df
#     else            :
#         airlines = airlines_df[['id_airline','name']].to_records(index=False)
#         return(airlines)

# -- objects --

if __name__== '__main__':
    setDF(r=30)
    df1     = pd.DataFrame(readJSONFile('transactions-1.json')['transactions'])
    df2     = pd.DataFrame(readJSONFile('transactions-2.json')['transactions'])
    evalDFcols(df1)
    evalDFcols(df2)
    df1.rename(columns={"bip125-replaceable": "bip125_replaceable"}, inplace=True)
    df2.rename(columns={"bip125-replaceable": "bip125_replaceable"}, inplace=True)
    try:
        session     = createDBsession(args.db)
        # session.commit()  # commit kills session
        con         = session.connection()
        deposits    = session.query(Deposits)                                   # load global objects for convenience
        deleteFromtable()
        loadDF2table()
        dep_df_old  = pd.read_sql('deposits', con, index_col='id')
        dep_df      = loadDFfromtable()
        ref_data    = RefData().addresses
        data_ref    = dict((v,k) for k,v in ref_data.items())
        spock       = dep_df[dep_df.address == data_ref['Spock']]
    except:
        logger.exception("issue loading database items", exc_info=1)
