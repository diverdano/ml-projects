#!/usr/env python
import pandas as pd
import numpy as np
import argparse
import logging
import simplejson as json
from database_setup import Base, Deposits, RefData, createDBsession
from datetime import datetime, timezone

# -- logging --
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # set to "off" when deploy
logger = logging.getLogger(__name__)
logger.propagate = False
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

def checkDatetime(time):
    '''check Datetime stamp'''
    return(datetime.fromtimestamp(float(time)/1000, timezone.utc)) # assume need to move decimal 3 places to get propper time

def timeDiff(row):
    '''return difference in times - like timedelta but for kraken times'''
    return(checkDatetime(row['blocktime']) - checkDatetime(row['time']))

def timeDiff_old(start, end):
    '''return difference in times - like timedelta but for kraken times'''
    return(checkDatetime(end) - checkDatetime(start))


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
    # time always equals timereceived for sample data                   --> drop timereceived
    # account, label, walletconflicts always empty                      --> drop all
    # bip125_replaceable always 'no', involvesWatchonly always 'yes'    --> drop both
    d = {'fields': "id, address, amount, blockhash, blockindex, blocktime,\
        category, confirmations, time, txid, vout",
        'table': "deposits"}
    s = "select {fields} from {table};"
    s.format(**d)
    return(pd.read_sql(s.format(**d), con, index_col='id'))

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

def createSummary(df, ref_df):
    '''generate stdout result per example'''
    largest     = df.amount.max()
    smallest    = df.amount.min()
    unref_df    = df[~df['address'].isin(ref_df.keys())]    # will progressivly do these? or remove with list?
    # print known accounts
    for ref in ref_df:
        print('Deposited for {}: count={} sum={:0.8f}'.format(ref_df[ref], df[df.address == ref].amount.count(), df[df.address == ref].amount.sum()))
    # print remaining
    print('Deposited without reference: count={} sum={:0.8f}'.format(unref_df.amount.count(), unref_df.amount.sum()))
    print('Smallest valid deposit: {:0.8f}'.format(smallest))
    print('Largest valid deposit: {:0.8f}'.format(largest))

def validDeposits(df):
    '''"detect all valid incoming deposits" - has 6 confirmations, need to de-dup responses, exclude send'''
    logger.info('processing transactions')
    # 1. remove 'duplicates' between two json files
    logger.info('remove duplicates')
    df.drop_duplicates(keep='last', inplace=True, subset = 'txid')  # keep "newer" txid record
    # what about duplicate blockhash - odds of the two blocks hashing to the same value is about 2^256 which is about 1.15 * 10^77 -- test scenario?
    logger.info('deposits records: {}'.format(df.shape[0]))
    # 2. remove all tx where confirmations < required per category
    logger.info('remove low confirms')
    df.loc[(df['category'] == 'receive') & (df['confirmations'] > 6), 'confirmed']      = True # + 167
    # df.loc[(df['category'] == 'send') & (df['confirmations'] > 6), 'confirmed']         = True # + 27 # removed per explicit requirement "incoming"
    df.loc[(df['category'] == 'immature') & (df['confirmations'] > 120), 'confirmed']   = True # + 0
    df.loc[(df['category'] == 'generate') & (df['confirmations'] > 100), 'confirmed']   = True # + 1
    logger.info('deposits records: {}'.format(df[(df.confirmed == True)].shape[0]))
    # 3. remove tx where blockindex is zero
    logger.info('remove zero blockindex')
    df.loc[(df['blockindex'] > 0), 'gt_zeroblock'] = True # only genesis blocks should be zero
    logger.info('deposits records: {}'.format(df[(df.confirmed == True) & (df.gt_zeroblock == True)].shape[0]))
    # 4. remove zero amount tx (is there another rule for min amount - given I don't have 'full raw transaction file'?)
    df.loc[(df['amount'] >= 0.00000546), 'gt_zero_amount'] = True # min spend block is 546 satoshi
    # set result dataframe
    valid = df[(df.confirmed == True) & (df.gt_zeroblock == True) & (df.gt_zero_amount == True)]
    logger.info('deposits records: {}'.format(valid.shape[0]))
    return(valid)

# -- objects --

if __name__== '__main__':
    setDF(r=40)
    df1     = pd.DataFrame(readJSONFile('transactions-1.json')['transactions'])
    df2     = pd.DataFrame(readJSONFile('transactions-2.json')['transactions'])
    # remove these
    json1   = readJSONFile('transactions-1.json')
    json2   = readJSONFile('transactions-2.json')
    # --
    lastblocks = {
        'json1' : '4f66926440f1b39fcd5db66609737f877ce32abfc68a945fbd049996ce7d0da2', # need to do anything with this?
        'json2' : '3125fc0ebdcbdae25051f0f5e69ac2969cf910bdf5017349ef55a0ef9d76d591' # need to do anything with this?
    }
    evalDFcols(df1)
    evalDFcols(df2)
    df1.rename(columns={"bip125-replaceable": "bip125_replaceable"}, inplace=True)
    df2.rename(columns={"bip125-replaceable": "bip125_replaceable"}, inplace=True)
    try:
        session     = createDBsession(args.db)
        # session.commit()  # commit kills session
        con         = session.connection()
        deposits    = session.query(Deposits)                           # load global objects for convenience, used for counts
        deleteFromtable()
        loadDF2table()
        # dep_df_old  = pd.read_sql('deposits', con, index_col='id')    # use function to eliminate unnecessary columns
        dep_df      = loadDFfromtable()                                 # df easier to read/manipulate than records
        valid_df    = validDeposits(dep_df)
        ref_data    = RefData().addresses
        createSummary(valid_df, ref_data)
        data_ref    = dict((v,k) for k,v in ref_data.items())           # reverse k,v -> v,k for reverse lookup - for data mining, not part of solution
        spock       = valid_df[valid_df.address == data_ref['Spock']]
        kirk        = valid_df[valid_df.address == data_ref['James T. Kirk']]
        bones       = valid_df[valid_df.address == data_ref['Leonard McCoy']]
        dax         = valid_df[valid_df.address == data_ref['Jadzia Dax']]
        scotty      = valid_df[valid_df.address == data_ref['Montgomery Scott']]
        wesley      = valid_df[valid_df.address == data_ref['Wesley Crusher']]
        archer      = valid_df[valid_df.address == data_ref['Jonathan Archer']]

    except:
        logger.exception("issue loading database items", exc_info=1)
