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

    # what about duplicate blockhash ???

    logger.info('deposits records: {}'.format(df.shape[0]))
    # 2. remove all tx where confirmations < required per category
        # generate    1         # must have 100 confirmations
        # immature    1         # must have 120 confirmations
        # receive     180       # must have 6 confirmations - deposits
        # send        27        # must have 6 confirmations - withdrawls
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

def validDepositsAdd(df):
    '''additional validation tests'''
    df.loc[(df['amount'] == 'generate') & (df['confirmations'] > 100), 'valid']   = True # + 1
    pass
    # time difference - only 'generate' category was "long", i.e. greater than a minute
    # vout vs amount "1/3"?

# blockhash                                                        blocktime     time          amount blockindex category confirmations vout txid
# 00ae9906635f6bc44a278327ab5323ff23d2d4efecf17d021a428a6278b79f26 1524896887278 1524896845186 81.8   53         receive  28            16   7c293b31771cb4ee7e400290023698ad5204789f141488f1f5f3948907266d49  1        1
# # 08c54bf5346467760aad4257ae8b0d87c8c78ee8f5c1f5be1409adfe50a9ce5a 1524889087278 1524889044705 90.2   50         receive  41            80   c652318119ac68b903ba47862308fa175f5762d65af0e4fcfa962e065996ed1e  1        1
# 090ff6cda58f49d103839ebb1168816c9a979016b7ca773457e811726e789a7b 1524905887278 1524905855382 71.7   77         receive  13            46   89d3906ecfda1655a5ef48d5a0a485034bb88a28a718b927b5b842bb65a2c4e9  1        1
# # 0b72458ef92a1a3fe6a09b2f37dea6e30808b9aca17eaad45bbbc3446bf70d22 1524903487278 1524903449203 0.0    51         receive  17            49   3d0e372ec56b1f06099ca23158f090cac14a7ee471be47e42d5a8a5385d3c4fc  1        1
# # 1310680b2511e7fa402ff6a08ae82c686ad65aa36d97aa758a8716f10816e14f 1524863887278 1524863876727 0.0    5          receive  83            72   231aa7362b7224d3aa094068efa7c21981dc666262108775cd27aef93675339e  1        1
# 15a495ec19b42ea21995e594a1e46a269a52ca826d48e58d008fb7df14093ac6 1524866887278 1524866864037 81.9   47         receive  78            26   a9660dc330082d2a458720fc6a4e823bc82853c9388aebbd83cd6c062b977e9e  1        1
# 194d60f41f6552c073eaf68ef74c34b885c383bb6ecbf70bb628995c4ce0b50e 1524899887278 1524899839696 46.9   48         receive  23            21   06f8e389126cb143ccb4e8b84fe0e873cd12d95d3d30acb324819e0053aa6887  1        1
#                                                                                # 1524899876195 2.0    41         receive  23            46   2a0ce6ba8be6dd89db5c5426e21bb4f7760995bf3151c978f7c0112a8022b719  1        1
# # 1a75bcf5d02db866011676d0ccca7cd1dc7a34a7f6b229b6d1ebd73e1e8be330 1524904087278 1524904059011 0.0    47         receive  16            88   b92d21be0666b7ecb5ce9f4e098042605f776f3bb3685909bace7b3d643f34f1  1        1
# 1b5a1f643d410838fd0a6541347c95950b3d109e8acf2854fb96fa9725add95c 1524895087278 1524895073309 62.1   20         receive  31            8    ef09794a11ac5355120e2990278bdc9b80ea3998fbdc3e1cafd937e5329ae842  1        1
# 363439f30ac5627b9291ccd7d56d70d2b51702c8b3fdb82ef8c1b992d5148087 1524886687278 1524886650207 81.5   59         receive  45            42   ba9320403b5a7b4e4d97eefe7e3ca72eadee299c18b5f51375097a7ad9b7d17f  1        1
# # 4a95b3ea7db4520448de1e518f31827b2941741290bd0a20f88b7086c7784d6f 1524871687278 1524871644784 39.3   63         receive  70            94   c8f092e5eed639f3cfd34f261762d8ba4bd5fe5a424a609fdb20fb00f8e345a1  1        1
# #                                                                                1524871671569 46.9   26         receive  70            39   2ff00061c93090c84f1b898d86bab6693d10ea9beb77d00562ff7cbe78e1f348  1        1
# 5cb99aa54c41439a3896c18d6347f8c495c73f7d707f513fae603f800a9bd93b 1524907687278 1524907640127 8.5    43         receive  10            42   0ccea6e0abc78e00f1f362a055a48fabfe97c8c906d74c24a7b3a52e92445353  1        1
# # 75c62a262bbf31f07637387f6c92022a7150e29203be23da7fcf415cc58bf596 1524877087278 1524877057628 70.1   3          receive  61            71   8aa309954de985defc4e1a8707ff43ef85e68f6cab829851b9cd41fafdd0e782  1        1
# # 7f8dd62a3c35fb46f2ead7a398b3ba96aa6023db6f7801055ff9892f30151cc3 1524890287278 1524890241551 77.9   12         receive  39            66   ccf8db56e0b7d67b527708925c7f144f499a0e546b666bac038eca0fc96048e3  1        1
#                                                                                # 1524890255941 5.4    71         receive  39            75   0e45ece597e3c3ca3ae86c467946ef5ef9463a72889f88d25811cf2ae90cf9ea  1        1
# 833ff1c3e9270a7b014b0f684a89e3f751a58a268dc6f127597d538db69e0a3b 1524868087278 1524868037826 84.9   30         receive  76            28   3b9f0ab947ead33764dfd0a84be66462dac7d007c6b44c8680411a6266900ea8  1        1
#                                                                                1524868044099 68.9   83         receive  76            17   3e0836e5cdfc57f7784c4a72bc9f8ba3f498b185b9951145e3d884361ef8307e  1        1
# 9038c8aefc091d6170dd9610257247259a9bad08a5fa5056ee755c372f25a547 1524895687278 1524895653054 50.3   44         receive  30            14   0321f14134b35868898084ddcd673a5676bd13bf0e06aac36d49d5b776a89766  1        1
# 95d1cb27ab156f53660d9b7cdc26fa07e3550fd5560e5904f27ec0ee5864baa7 1524862087278 1524862054076 97.0   97         receive  86            25   9e7f6710f51413e7834ecefd98bad8ef1b8d7b8342e1b90268590b2fd6980c83  1        1
# b474df60c9b07663097ccaf0e2eb90ad86b0970f8df766da706a8c80576868ce 1524902887278 1524902871033 8.9    66         receive  18            28   dea87932e924f243c4ea2018de8247952009009f96db88cbf0290c5e5e3c3fa3  1        1
# # cba8dd05aaf72bf8dd203a2dbddf7069e24ff90f70ccada40792f4b3c07d179e 1524908887278 1524908877059 0.0    13         receive  8             37   d1579f9602d3a6c783daac95fcea3390f57adff85455233de000f90c128c4cfe  1        1
# d2382dcc16106ec07823995edc0a8f548decd70d3a59ef9549c171741a937a43 1524865087278 1524865049557 4.7    76         receive  81            7    f4a595cdd4b4c9ed7733d8a44def0455ef906b20b8984889c1b255ff063184c0  1        1
# e47ae5d958fc0e8d87e775694d53fe78463a9163fe62d14d580a379d3ae79080 1524862687278 1524862647103 99.5   52         receive  85            6    78e310efda2c933abdbedf195cf7da34a8cf36ef5b1750d01d6dc0224506674c  1        1
# f570848e8b27388453979a2b3b649310a0ffe9e26e9fbfa7f88be75890a48bea 1524863287278 1524863261484 49.3   45         receive  84            42   71eccb99aa66e0fdaca57d6a199bcde22e7c9e4f2d0ad76cfc86d32b9fe5c0d0  1        1
# f8f1da1e5a722a975aa008e43c35e948593bb3e10945f56b8ebf691f2bc62ce8 1524906487278 1524906444397 18.0   82         receive  12            61   2bba16c820dec4ff9ff68256cfe3672a67144dee0681e5db3adff20436fa6e0f  1        1
#                                                                                # 1524906453619 19.2   0          receive  12            94   bf3765997ac109480b13262182bbbc3fbe993cff5567e40772961558b136fc8b  1        1

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
