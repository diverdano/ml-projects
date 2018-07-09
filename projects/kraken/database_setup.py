#import sys
import sqlite3 # alternative to sqlalchemy
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Deposits(Base):
    __tablename__       = 'deposits'
    id                  = Column(Integer, primary_key = True, autoincrement = True)
    account             = Column(String(5))
    address             = Column(String(35))
    amount              = Column(Float)
    bip125_replaceable  = Column(String(2))
    blockhash           = Column(String(64))
    blockindex          = Column(Integer)
    blocktime           = Column(Integer)
    category            = Column(String(8))
    confirmations       = Column(Integer)
    involvesWatchonly   = Column(Boolean)
    label               = Column(String(5))
    time                = Column(Integer)
    timereceived        = Column(Integer)
    txid                = Column(String(64))
    vout                = Column(Integer)
    walletconflicts     = Column(String(5))
    def __repr__(self):
        return(str(self.__dict__))
        # return("\n\tasset id: {0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(self.id, self.type, self.exchange, self.symbol, self.description, self.size))
    def __init__(self, account, address, amount,
        bip125_replaceable,
        blockhash,
        blockindex,
        blocktime,
        category,
        confirmations,
        involvesWatchonly,
        label,
        time,
        timereceived,
        txid,
        vout,
        walletconflicts):
        '''load attributes into object'''
        self.account = account
        self.address    = address
        self.amount     = amount
        self.bip125_replaceable = bip125_replaceable
        self.blockhash  = blockhash
        self.blockindex = blockindex
        self.blocktime  = blocktime
        self.category   = category
        self.confirmations  = confirmations
        self.involvesWatchonly  = involvesWatchonly
        self.label      = label
        self.time       = time
        self.timereceived = timereceived
        self.txid       = txid
        self.vout       = vout
        self.walletconflicts = walletconflicts

class RefData(object):
    ''' reference data, could be moved to DB if needed '''
    addresses = {
        'mvd6qFeVkqH6MNAS2Y2cLifbdaX5XUkbZJ': 'Wesley Crusher',
        'mmFFG4jqAtw9MoCC88hw5FNfreQWuEHADp': 'Leonard McCoy',
        'mzzg8fvHXydKs8j9D2a8t7KpSXpGgAnk4n': 'Jonathan Archer',
        '2N1SP7r92ZZJvYKG2oNtzPwYnzw62up7mTo': 'Jadzia Dax',
        'mutrAf4usv3HKNdpLwVD4ow2oLArL6Rez8': 'Montgomery Scott',
        'miTHhiX3iFhVnAEecLjybxvV5g8mKYTtnM': 'James T. Kirk',
        'mvcyJMiAcSXKAEsQxbW9TYZ369rsMG6rVV': 'Spock'
    }

def createDBsession(db_info):
    ''' create session to db '''
    return sessionmaker()(bind=create_engine(db_info))
engine = create_engine('sqlite:///deposits.db')
Base.metadata.create_all(engine)        # suppress this if creating DB via SQL script?
