#!usr/bin/env python

# === load libraries ===

# key libraries
import logging

# science
import numpy as np                  # can remove this once its removed from plot_learning_curve
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

# plot
# import matplotlib as mpl
# mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.sankey import Sankey
#import graphviz                     # decision tree node diagram
#import pylab as pl                 # explore this
#import visuals as vs               # explore this
#import bokeh                	    # for data visualizations

# data
import pandas as pd                 # used for a type test in plotData function

# == set logging ==
logger = logging.getLogger(__name__)

def testLog():
    '''test logging feature'''
    logger.info('this is a test')

# === plot ===
def plotCorr(data):
    '''plot correlation matrix for data (pandas DataFrame), exludes non-numeric attributes'''
    correlations    = data.corr()
    names           = list(correlations.columns)
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

def plotData(data, title=None, xlabel=None, ylabel=None, grid=True, legend=True):
    ''' plot data series from pandas.Series: TODO: test alt data structures'''
    plt.figure()
    # dressing
    if title    == None : title  = 'Title of graph'
    if xlabel   == None : xlabel = 'x axis label'
    if ylabel   == None : ylabel = 'y axis label'
    if grid             : plt.grid()
    if legend           : plt.legend(loc="best")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plot/chart type
    if type(data) == pd.core.series.Series: plt.plot(data.index, data)  # need to improve the type test
    # display plot
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean   = np.mean(train_scores, axis=1)
    train_scores_std    = np.std(train_scores, axis=1)
    test_scores_mean    = np.mean(test_scores, axis=1)
    test_scores_std     = np.std(test_scores, axis=1)
    logger.info('train score mean & std: ', train_scores_mean, train_scores_std)
    logger.info('test score mean & std : ', test_scores_mean, test_scores_st)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
#    return plt

# basic sankey chart
def createSankey(flows = None, labels = None, orientations = None):
    '''create simple sankey diagram
        default example:
            flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
            labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
            orientations=[-1, 1, 0, 1, 1, 1, 0,-1]'''
    if flows == None    : flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10]
    if labels == None   : labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth']
    if orientations == None:orientations=[-1, 1, 0, 1, 1, 1, 0,-1]
    Sankey(flows=flows, labels=labels, orientations=orientations).finish()
    plt.title("Sankey diagram with default settings")
    plt.show()


# === plot functions ===
def output_image(name, format, bytes):
    '''not sure what this function was for... face recognition?'''
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    logger.info(image_start+json.dumps(data)+image_end)

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)
    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
#    plt.savefig("test.png")

def correlation_matrix(df):
    ''''''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 130)
#    cmap = cm.get_cmap('jet', 30)
    cax = ax.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    # also check out matshow - plot a matrix as an image
    # also check out xcorr - plot a correlation x & y
    # minorticks_on()
#    ax.grid(True, markevery=1)
    ax.grid(True, markevery=1)
    plt.title('Project Data Feature Correlation')
    labels=list(df.columns)
#    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax.set_xticklabels(labels, fontsize=6, minor=True, rotation='vertical')
    ax.set_yticklabels(labels, fontsize=6, minor=True, rotation='vertical')
    ax.set_xticks(range(0,len(labels)), minor=True)
    ax.set_yticks(range(0,len(labels)), minor=True)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.show()

def viewRegPlot(self, feature=0, color='yellow', alpha=0.4):
    ''' setup chart for plotting feature vs target variable '''
    feature         = self.project.features[feature]
    feature_series  = self.X[feature].values.reshape(-1,1)
    reg = LinearRegression()
    reg.fit(feature_series, self.y)
    coef            = reg.coef_[0] # with multiple regression model there are multiple coefficients
    inter           = reg.intercept_
    score           = reg.score(feature_series, self.y)
    r_score         = "r-score: {0:.3f}".format(score)
    title           = "Regression Plot of {0} vs {1}".format(feature, self.project.target)
    reg_label       = "coefficient: {0:,.0f}, intercept: {1:,.0f}".format(coef, inter)
    scatter_label   = "plot of {0} vs {1}".format(feature, self.project.target)
    bbox            = {'facecolor':color, 'alpha':alpha}
    # plot
    plt.plot(feature_series, reg.predict(feature_series), color='red', linewidth=1, label=reg_label)
    plt.scatter(feature_series, self.y, alpha=0.5, c=self.y, label=scatter_label)
    # labels
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel(self.project.target)
    plt.figtext(x=0.5, y=0.88, s = r_score, bbox=bbox, horizontalalignment='center', verticalalignment='top')
    plt.figtext(x=0.5, y=0.12, s = reg_label, bbox=bbox, horizontalalignment='center', verticalalignment='bottom')
#        plt.legend(loc = 'upper center')
    plt.show()
def viewScatterPlots(self, newX=None, newY=None, color_xy='blue', color_newxy='red'):
    ''' setup charts for plotting input features vs scatterplot of historical values '''
    for i, feat in enumerate(self.X.keys()):
        plt.scatter(self.X[feat], self.y, color=color_xy)
        if newX != None:
            plt.scatter(newX[i], newY, color=color_newxy)       # for
        plt.xlabel('feature {}'.format(feat))
        plt.show()
def viewRegPlots(project, color='yellow', alpha=0.4, featureStart=0, features=4):    # TODO: allow graphing more than 4 features
    ''' setup charts for plotting features vs target variable '''
    fig         = plt.figure(figsize=(16,10))
    featureEnd  = min(featureStart + features, len(project.features))               # need to take sqrt of features
    len(project.features)
    # Create three different models based on max_depth
    for k, feature in enumerate(project.features[featureStart:featureEnd]):
        feature_series  = project.X[feature].values.reshape(-1,1)
        reg = LinearRegression()
        reg.fit(feature_series, project.y)
        coef            = reg.coef_[0] # with multiple regression model there are multiple coefficients
        inter           = reg.intercept_
        score           = reg.score(feature_series, project.y)
        r_score         = "r-square: {0:.3f}".format(score)
        title           = "{0} given {1} (r-square: {2:.3f})".format(project.target, feature, score)
        reg_label       = "coef: {0:,.0f}, intercept: {1:,.0f}".format(coef, inter)
        scatter_label   = "plot of {0} given {1}".format(project.target, feature)
        bbox            = {'facecolor':color, 'alpha':alpha}
        # plot
#        ax = fig.add_subplot(2, 2, k+1)
        ax = fig.add_subplot(4, 4, k+1)                                         # need to take sqrt of features
        ax.plot(feature_series, reg.predict(feature_series), color='red', linewidth=1, label=reg_label)
        ax.legend(loc='lower right', borderaxespad = 0.) # plot legend without the scatter plot
        ax.scatter(feature_series, project.y, alpha=0.5, c=project.y, label=scatter_label)
        # labels
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel(project.target)
#            ax.figtext(x=0.5, y=0.88, s = r_score, bbox=bbox, horizontalalignment='center', verticalalignment='top')
#            ax.figtext(x=0.5, y=0.12, s = reg_label, bbox=bbox, horizontalalignment='center', verticalalignment='bottom')
#        ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower right', borderaxespad = 0.)
    fig.suptitle('Regression ScatterPlots', fontsize = 16, y = 1)
    fig.set_tight_layout(tight='tight')
    fig.show()


# == silhouette plots ==
