#!usr/bin/env python

# == load libraries ==
from statsmodels.tsa.arima_model import ARIMA

# custom
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

def parser(x):
	return util_data.pd.datetime.strptime('190'+x, '%Y-%m')
	# return datetime.strptime('190'+x, '%Y-%m')

def createARIMA(file):
    series = util_data.pd.read_csv(file, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = util_data.pd.DataFrame(model_fit.resid)
    # residuals = DataFrame(model_fit.resid)
    residuals.plot()
    util_plot.plt.show()
    # pyplot.show()
    residuals.plot(kind='kde')
    util_plot.plt.show()
    # pyplot.show()
    print(residuals.describe())
