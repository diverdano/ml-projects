#!/usr/env python
import pandas as pd
import numpy as np
# import psycopg2
import argparse
import logging
import re
# from sqlalchemy import create_engine
import io
import simplejson as json

# -- logging --
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -- argument construction and parsing --
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--run",          default=False,                          help="run script")
ap.add_argument("-d", "--dataset",      default='projects/image_class/knn-classifier/kaggle_dogs_vs_cats/train/',   help="path to input dataset")
ap.add_argument("-k", "--neighbors",    type=int, default=1,                    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs",         type=int, default=-1,                   help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())
logger.info('\n\tlogging is on\n')

# -- helper functions --

def readJSONFile(file):
    '''simple read file'''
    with open(file, 'r') as infile: data=infile.read()
    return(json.loads(data))

def getMaxLengths(df):
    '''return max length for each column in df'''
    df_lengths = {}
    for column in df.columns:
        df_lengths[column] = df[column].str.len().max() # fails on df object type,
    return(df_lengths)

def setDF(w=None, c=None, r=None, f='{:,.1f}'):
    '''set width, max columns and rows'''
    pd.set_option('display.width', w)               # show columns without wrapping
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', c)         # show all columns without elipses (...)
    pd.set_option('display.max_rows', r)            # show default number of rows for summary
    pd.options.display.float_format = f.format
    np.set_printoptions(formatter={'float': lambda x: f.format(x)})

# -- load data --

# pd1.groupby(['address','txid','confirmations','blocktime','category'])['txid'].count()

# -- transform data --

def getAirlines(df, type='df'):
    '''return airlines dataset - several options (to_csv, to_records, to_string, to_gbq, to_parquet)'''
    logger.info('processing airlines')
    airlines_df = df.groupby('AIRLINECODE', as_index=False )['AIRLINENAME'].last()
    airlines_df.columns = (['id_airline','old_name'])
    airlines_df['name'] = airlines_df['old_name'].apply(lambda x: x.split(':')[0])   # strip ':' and trailing text
    airlines_df.drop(['old_name'], axis=1, inplace=True)
    airlines_df.sort_values('name', inplace=True)
    logger.info('airlines found: {:,}'.format(len(airlines_df)))
    if airlines_df.isnull().values.any():
        nan_airlines_df = airlines_df[pd.isnull(airlines_df).any(axis=1)]
        logger.error(['airlines with NaN values:', nan_airlines_df])
    if type == 'df' : return airlines_df
    else            :
        airlines = airlines_df[['id_airline','name']].to_records(index=False)
        return(airlines)


# -- objects --

def sqlalchemy2sql(df, table):
    '''use psycopg2 instead?'''
    host        = 'iw-recruiting-test.cygkjm9anrym.us-west-2.rds.amazonaws.com'
    port        = '5432'
    database    = 'tests_data_engineering'
    schema      = 'candidate7206'
    user        = 'candidate7206'
    password    = 'vhWPmzt5a60cOZY5'
    con_str     = 'postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(user, password, host, port, database)
    logger.info('connecting to IW db')
    engine = create_engine(con_str)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, table, null="") # null values become ''
    conn.commit()
    logger.info('records written')


class IWDB(object):
    '''object for working with database, preference is to use as a class'''
    credentials = {
        'host'      :'iw-recruiting-test.cygkjm9anrym.us-west-2.rds.amazonaws.com',
        'database'  :'tests_data_engineering',
        'schema'    :'candidate7206',
        'user'      :'candidate7206',
        'password'  :'vhWPmzt5a60cOZY5'}
    def __init__(self):
        logger.info('initializing IWDB object')
        self.conn = psycopg2.connect(
                    host        = self.credentials['host'],
                    database    = self.credentials['database'],
                    user        = self.credentials['user'],
                    password    = self.credentials['password'])
        logger.info('connection made')
        self.cur = self.conn.cursor()
        logger.info('cursor constructed')
    def select(self, sql = 'select nspname from pg_catalog.pg_namespace;'):
        '''generic execute sql statement'''
        logger.info('executing select statement: \n\t{0}'.format(sql))
        self.cur.execute(sql)                                       # execute a statement
        result = self.cur.fetchone()                                # decide how to display
        logger.info(result)                                         # decide how to display
    def copyFrom(data, table):
        '''copy from data object to db table'''
        self.cur.copy_from(data, table)
    def getDBver(self): self.select(sql='select version()')

if __name__== '__main__':
    setDF(r=30)
    df1     = pd.DataFrame(readJSONFile('transactions-1.json')['transactions'])
    df2     = pd.DataFrame(readJSONFile('transactions-2.json')['transactions'])
    # flights = getFlights()
    # airlines = getAirlines(flights)
    # airports = getAirports(flights)
    # airplanes = getAirplanes(flights)
