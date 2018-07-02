#!/usr/env python
import pandas as pd
import numpy as np
import psycopg2
import argparse
import logging
import re

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

def getFlights(file = 'flights.csv', sep = '|'):
    '''return pd dataframe with parsed file data'''
    logger.info('processing flights')
    flights = pd.read_csv(filepath_or_buffer = file, sep = sep)
    flights = flights.rename(columns = {'ORIGAIRPORTNAME':'ORIGINAIRPORTNAME'})     # column name inconsistency in source
    logger.info('flights found: {:,}'.format(len(flights)))
    return(flights)

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

def getAirports(df, type='df'):
    '''return airports dataset'''
    #dft1 = dft1.astype({'a': np.bool, 'c': np.float64})
    logger.info('processing airports')
    airports_df_orig = df.groupby('ORIGINAIRPORTCODE', as_index=False)['ORIGINAIRPORTNAME', 'ORIGINCITYNAME', 'ORIGINSTATE', 'ORIGINSTATENAME'].last()
    airports_df_dest = df.groupby('DESTAIRPORTCODE', as_index=False)['DESTAIRPORTNAME', 'DESTCITYNAME', 'DESTSTATE', 'DESTSTATENAME'].last()
    airports_df_orig.columns = (['id_airport','name','city','st','state'])      # could join these into 1 statement
    airports_df_dest.columns = (['id_airport','name','city','st','state'])
    airports_df = pd.concat([airports_df_orig, airports_df_dest])
    airports_df['name'] = airports_df['name'].apply(lambda x: x.split(':')[1])  # strip ':' and trailing text
    airports_df.sort_values(by=['id_airport'], axis=0, inplace=True)
    airports_df.drop_duplicates(inplace=True)                                   # perhaps force columns to exclude index...
    airports_df.reset_index(drop=True, inplace=True)
    logger.info('airports found: {:,}'.format(len(airports_df)))
    if airports_df.isnull().values.any():
        nan_airports_df = airports_df[pd.isnull(airports_df).any(axis=1)]
        nan_airports    = list(airports_df[pd.isnull(airports_df).any(axis=1)]['id_airport'])     # set aside bad airport names
        logger.error(['airports with NaN values:', nan_airports_df])
        OK_airports     = ['LAW', 'OKC','TUL']
        KS_airports     = ['FOE', 'GCK', 'HYS', 'ICT', 'MHK']
        airports_df['st'][airports_df.id_airport.isin(KS_airports)]       = "KS"
        airports_df['state'][airports_df.id_airport.isin(KS_airports)]    = "Kansas"
        airports_df['st'][airports_df.id_airport.isin(OK_airports)]       = "OK"
        airports_df['state'][airports_df.id_airport.isin(OK_airports)]    = "Oklahoma"
        nan_airports_df = airports_df[pd.isnull(airports_df).any(axis=1)]                   # retest for Nan
        logger.error(['airports with NaN values:', nan_airports_df])

    if type == 'df' : return airports_df
    else            :
        airports = airports_df.to_records(index=False)
        return(airlines)

# 128  FOE         Forbes Field                          Topeka               NaN  NaN --> KS
# 133  GCK         Garden City Regional                  Garden City          NaN  NaN --> KS
# 163  HYS         Hays Regional                         Hays                 NaN  NaN --> KS
# 167  ICT         Wichita Dwight D Eisenhower National  Wichita              NaN  NaN --> KS
# 195  LAW         Lawton-Fort Sill Regional             Lawton/Fort Sill     NaN  NaN --> OK
# 227  MHK         Manhattan Regional                    Manhattan/Ft. Riley  NaN  NaN --> KS
# 254  OKC         Will Rogers World                     Oklahoma City        NaN  NaN --> OK
# 345  TUL         Tulsa International                   Tulsa                NaN  NaN --> OK

def getAirplanes(df, type='df', replace=True):                         # probably quicker in db
    '''return airplane dataset'''
    logger.info('processing airplanes')
#    airplanes_df = df.groupby(['TAILNUM', 'AIRLINECODE'], as_index=False)['AIRLINECODE'].last()      # removing records where plane flew under different airline
    airplanes_df = df.groupby('TAILNUM', as_index=False)['AIRLINECODE'].last()      # removing records where plane flew under different airline
    airplanes_df.columns = (['id_tailnum','id_airline'])
    logger.info('airplanes records found: {:,}'.format(len(airplanes_df)))
    logger.info('column names: {}'.format(airplanes_df))
    # FAA website: http://registry.faa.gov/aircraftinquiry/NNum_Inquiry.aspx
    if replace:
        # possible bad tail numbers: 'NKNO', 'UNKNOW', all beginning with '-', 'N998R@', 'NEIDLA'
        char_list = ["@", "'", "-", "UNKNOW", "NKNO"]                                    # list of characters to replace
        airplanes_df = airplanes_df['id_tailnum'].apply(lambda x: re.sub("|".join(char_list), "", x).upper()) # should store original and 'fixed'
    #    airplanes_df['id_tailnum'] = airplanes_df['id_tailnum'].apply(lambda x: x.strip('-'))
#        airplanes_df = airplanes_df.groupby('id_tailnum', as_index=False)['id_airline'].last()
#        logger.info('unique airplanes found: {:,}'.format(len(airplanes_df)))
    if type == 'df' : return airplanes_df
    else            :
        airplanes = airplanes_df.to_records(index=False)
        return(airplanes)

def trimFlights(df, type='df'):
    '''return flight fact dataset'''
    logger.info('processing flights - for fact table')
    drop_columns = [
        'AIRLINENAME',
        'ORIGINAIRPORTNAME',          # fixed in getFlights()
        'ORIGINCITYNAME',
        'ORIGINSTATE',
        'ORIGINSTATENAME',
        'DESTAIRPORTNAME',
        'DESTCITYNAME',
        'DESTSTATE',
        'DESTSTATENAME']
    new_col_names = [
        'id_trans',
        'date_flight',
        'id_airline',
        'id_tailnum',
        'id_flightnum',
        'id_airport_orig',
        'id_airport_dest',
        'time_depart_crs',
        'time_depart',
        'time_depart_delay',
        'time_taxi_out',
        'time_wheelsoff',
        'time_wheelson',
        'time_taxi_in',
        'time_arrive_crs',
        'time_arrive',
        'time_arrive_delay',
        'time_elapsed_crs',
        'time_elapsed_act',
        'stat_cancelled',
        'stat_diverted',
        'stat_miles']               # strip miles and keep imperial measure vs metric
    flight_log                      = df.drop(drop_columns, axis=1, errors='ignore')        # don't need inplace with assignment
    flight_log.columns              = (new_col_names)
#    df[['col.name1', 'col.name2'...]] = df[['col.name1', 'col.name2'..]].astype('data_type')
#   to_datetime
#   to_timedelta
    tf_dict                         = {'T':1, 'True':1, 'F':0, 'False':0, 'NA':0, '1':1, '0':0}     # change all T, True to 1 and F, False to 0
    flight_log['stat_diverted']     = flight_log['stat_diverted'].map(tf_dict)
    flight_log['stat_cancelled']    = flight_log['stat_cancelled'].map(tf_dict)#.astype('int64', errors='ignore')  # convert upon upload?
    flight_log['stat_miles']        = flight_log['stat_miles'].apply(lambda x: int(x.split()[0]))
    cols_time                       = ['time_depart_crs','time_depart','time_arrive_crs','time_arrive','time_wheelsoff','time_wheelson']
    cols_timedelta                  = ['time_elapsed_crs','time_elapsed_act']
    cols_int                        = ['id_trans','id_flightnum','time_depart_delay','time_arrive_delay','time_taxi_out','time_taxi_in','time_arrive_crs','time_arrive']
    flight_log['date_flight']       = flight_log.date_flight.apply(lambda x: pd.to_datetime(x, format='%Y%M%d').date())
#    flight_log[cols_time]           = flight_log[cols_time].apply(lambda x: pd.to_datetime(x, format='%h%m').time()) # fails on nulls
#    flight_log[cols_time]           = flight_log[cols_time].apply(lambda x: pd.to_datetime(x, unit='m'))
    flight_log[cols_timedelta]      = flight_log[cols_timedelta].apply(lambda x: pd.to_timedelta(x, unit='m'))
    flight_log[cols_int]            = flight_log[cols_int].apply(lambda x: pd.to_numeric(x))
    return(flight_log)

# -- objects --

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
    def getDBver(self): self.select(sql='select version()')
    # def getDBversion(self):
    #     '''create cursor, get db version, log and close cursor'''
    #     conn = None
    #     try:
    #         logger.info('Connecting to the PostgreSQL database...') # connect to the PostgreSQL server
    #         conn = psycopg2.connect(**params)
    #         cur = conn.cursor()                                     # create cursor
    #         logger.info('PostgreSQL database version:')
    #         cur.execute('SELECT version()')                         # execute a statement
    #         db_version = cur.fetchone()                             # display the PostgreSQL database server version
    #         logger.info(db_version)
    #         cur.close()                                             # close the communication with the PostgreSQL
    #     except (Exception, psycopg2.DatabaseError) as error:
    #         logger.info(error)
    #     finally:
    #         if conn is not None:
    #             conn.close()
    #             logger.info('Database connection closed.')


'''
source table:
    TRANSACTIONID          int64    f
    FLIGHTDATE             int64    f
    AIRLINECODE           object    f
    AIRLINENAME           object    d
    TAILNUM               object    f
    FLIGHTNUM              int64    f - could become a dimension (these codes are reused)
    ORIGINAIRPORTCODE     object    f
    ORIGAIRPORTNAME       object    d
    ORIGINCITYNAME        object    d
    ORIGINSTATE           object    d
    ORIGINSTATENAME       object    d
    DESTAIRPORTCODE       object    f
    DESTAIRPORTNAME       object    d
    DESTCITYNAME          object    d
    DESTSTATE             object    d
    DESTSTATENAME         object    d
    CRSDEPTIME             int64    f
    DEPTIME              float64    f
    DEPDELAY             float64    f
    TAXIOUT              float64    f
    WHEELSOFF            float64    f
    WHEELSON             float64    f
    TAXIIN               float64    f
    CRSARRTIME             int64    f
    ARRTIME              float64    f
    ARRDELAY             float64    f
    CRSELAPSEDTIME       float64    f
    ACTUALELAPSEDTIME    float64    f
    CANCELLED             object    f
    DIVERTED              object    f
    DISTANCE              object    f - could become a dimension

fact table flights
    id_trans
    date_flight
    id_airline
    id_tailnum
    id_flightnum
    id_airport_orig
    id_airport_dest
    time_depart_crs
    time_depart
    time_depart_delay
    time_taxi_out
    time_wheelsoff
    time_wheelson
    time_taxi_in
    time_arrive_crs
    time_arrive
    time_arrive_delay
    time_elapsed_crs
    time_elapsed_act
    stat_cancelled
    stat_diverted
    stat_distance

dim table airlines
    id_airline
    name

dim table airports
    id_airport
    name
    city
    st
    state (iso standard, useful?)

dim table airplanes (hmmm - history... given incidents)
    id_tailnum
    id_airline (could be transferred)
    date_service_first
    date_service_last
    cum_miles (calc from all segments)


'''
# get airlines: group by 'AIRLINENAME',
# parse on ':' to strip name from code,
# parse on '()' to strip note
# get airports: group by 'ORIGINAIRPORTCODE' AND 'DESTAIRPORTCODE' (need both for full coverage),
# then lift

# -- problem definition/setup --
'''
Case Requirements
1. Load the provided data into the PostgreSQL instance using the provided credentials.
2. Create and load one Fact table to contain data about the flights.
3. Create and load appropriate Dimension table(s).
4. Create a view that joins your Fact table to your Dimension tables and returns columns useful for analysis.
5. Prepare a 10-15 minute presentation to show your work, discuss your approach and any data quality issues that were encountered and your resolution to these issues.
Fact Table
1. Create an additional column called distance_group that bins the distance values into groups in 100 mile increments. Example: 94 miles is 0-100 miles. 274 miles is 201-300 miles.
2. Create an additional column that indicates if the departure delay in minutes (DEPDELAY) is greater than 15.
3. Create an additional column that indicates if the flight arrival time (ARRTIME) is the next day after the departure time (DEPTIME).
4. Choose appropriate data types and perform conversions to load the data from the source into these types.
5. Fix obviously bad data when encountered, if possible. Note these instances.
Dimension Table(s)
1. Create at least one dimension table and load it from the source data.
2. Use your judgment about what columns from the source data should end up in the dimension tables. Be prepared to explain your decisions.
3. Clean up the Airline Name column by removing the Airline Code from it.
4. Clean up the Airport Name fields by removing the concatenated city and state.
5. Fix obviously bad data when encountered, if possible. Note these instances.

Postgres: iw-recruiting-test.cygkjm9anrym.us-west-2.rds.amazonaws.com
Database: tests_data_engineering
Username: candidate7206
Password: vhWPmzt5a60cOZY5
Schema: candidate7206
(port n/a, defaults to 5432 if not provided)
'''

if __name__== '__main__':
    setDF()
    flights = getFlights()
    airlines = getAirlines(flights)
    airports = getAirports(flights)
    # airplanes = getAirplanes(flights)
