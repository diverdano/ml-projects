#!usr/bin/python


# === load libraries ===
import util_data

# === test functions ===
def testFirstN(count=1000000):
    return(sum(firstn(count)))

# === model object ===

class firstn(object):
    def __init__(self, n):
        self.n = n
        self.num, self.nums = 0, []

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num+1
            return cur
        else:
            raise StopIteration()

class BankData(object):
    name        = "Bank Name / Holding Co Name"
    number      = "Bank ID"
    site        = 'Charter'
    main        = 'Bank Location'
    assets      = "Consol Assets (Mil $)"
    dom_assets  = "Domestic Assets (Mil $)"
    branches    = "Domestic Branches"
    def __init__(self, banks):
        self.banks          = util_data.ProjectData(banks).data[[self.number, self.name, self.site, self.main, self.branches, self.assets]]
        self.banks.columns = ['bankID','name','charter','location','branches', 'assets']
        self.banks.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    def sort_assets(self, df_col=None):
        pd.set_option('display.max_rows', df_col)               # show default number of rows for summary
        return self.banks.sort_values('assets', ascending=False)


class CreditUnionData(object):
    ''' base model object '''
    name        = "CU_NAME"
    number      = "CU_NUMBER"
    site        = 'SiteTypeName'
    main        = 'MainOffice'
    city        = "PhysicalAddressCity"
    state       = "PhysicalAddressStateCode"
    assets      = "ACCT_010"
#     merge_type  = 'inner'
    def __init__(self, cus, accounts):
        self.cus            = util_data.ProjectData(cus).data[[self.number, self.name, self.city, self.state, self.site, self.main]]
        self.accounts       = util_data.ProjectData(accounts).data[[self.number, self.assets]]
        self.merge()
    def merge(self):
        self.parents        = self.cus.loc[self.cus[self.main] == 'Yes']
        self.merged         = pd.merge(left=self.parents, right=self.accounts, how='inner', left_on=self.number, right_on=self.number)
        self.merged.columns = ['number','name','city','state', 'type', 'main', 'assets']
        self.CM             = self.merged.loc[self.merged['assets'] > 100000000]
    def sort_assets(self, df_col=None):
        pd.set_option('display.max_rows', df_col)               # show default number of rows for summary
        return self.CM.sort_values('assets', ascending=False)
