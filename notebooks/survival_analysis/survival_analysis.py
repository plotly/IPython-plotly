
# coding: utf-8

# # Survival Analysis with Plotly: R vs Python

# <h2 id="tocheading">Table of Contents</h2>
# <div id="toc"><ul class="toc"><li><a href="#Survival-Analysis-with-Plotly:-R-vs.-Python">I. Survival Analysis with Plotly: R vs Python</a><a class="anchor-link" href="#Survival-Analysis-with-Plotly:-R-vs.-Python">¶</a></li><ul class="toc"><li><a href="#Introduction">I. Introduction</a><a class="anchor-link" href="#Introduction">¶</a></li><li><a href="#Censoring">II. Censoring</a><a class="anchor-link" href="#Censoring">¶</a></li><li><a href="#Loading-data-into-python-and-R">III. Loading data into Python and R</a><a class="anchor-link" href="#Loading-data-into-python-and-R">¶</a></li></ul><li><a href="#Estimating-survival-with-Kaplan-Meier">II. Estimating survival with Kaplan-Meier</a><a class="anchor-link" href="#Estimating-survival-with-Kaplan-Meier">¶</a></li><ul class="toc"><ul class="toc"><li><a href="#Using-R">I. Using R</a><a class="anchor-link" href="#Using-R">¶</a></li><li><a href="#Using-Python">II. Using Python</a><a class="anchor-link" href="#Using-Python">¶</a></li></ul></ul><li><a href="#Multiple-Types">III. Multiple Types</a><a class="anchor-link" href="#Multiple-Types">¶</a></li><ul class="toc"><ul class="toc"><li><a href="#Using-R">I. Using R</a><a class="anchor-link" href="#Using-R">¶</a></li><li><a href="#Using-Python">II. Using Python</a><a class="anchor-link" href="#Using-Python">¶</a></li></ul></ul><li><a href="#Testing-for-Difference">IV. Testing for Difference</a><a class="anchor-link" href="#Testing-for-Difference">¶</a></li><ul class="toc"><ul class="toc"><li><a href="#Using-R">I. Using R</a><a class="anchor-link" href="#Using-R">¶</a></li><li><a href="#Using-Python">II. Using Python</a><a class="anchor-link" href="#Using-Python">¶</a></li></ul></ul><li><a href="#Estimating-Hazard-Rates">V. Estimating Hazard Rates</a><a class="anchor-link" href="#Estimating-Hazard-Rates">¶</a></li><ul class="toc"><ul class="toc"><li><a href="#Using-R">I. Using R</a><a class="anchor-link" href="#Using-R">¶</a></li><li><a href="#Using-Python">II. Using Python</a><a class="anchor-link" href="#Using-Python">¶</a></li></ul></ul></ul></div>

# In this notebook, we introduce survival analysis and we show application examples using both R and Python. We will compare the two programming languages, and leverage Plotly's Python and R APIs to convert static graphics into interactive `plotly` objects.
# 
# [Plotly](https://plot.ly) is a platform for making interactive graphs with R, Python, MATLAB, and Excel. You can make graphs and analyze data on Plotly’s free public cloud. For collaboration and sensitive data, you can run Plotly [on your own servers](https://plot.ly/product/enterprise/).
# 
# For a more in-depth theoretical background in survival analysis, please refer to these sources:
# 
# - [Lecture Notes by John Fox](http://socserv.mcmaster.ca/jfox/Courses/soc761/survival-analysis.pdf)
# - [Wikipedia article](http://en.wikipedia.org/wiki/Survival_analysis)
# - [Presentation by Kristin Sainani](www.pitt.edu/~super4/33011-34001/33051-33061.ppt)
# - [Lecture Notes by Germán Rodríguez](http://data.princeton.edu/wws509/notes/c7.pdf)
# 
# Need help converting Plotly graphs from R or Python?
# - [R](https://plot.ly/r/user-guide/)
# - [Python](https://plot.ly/python/matplotlib-to-plotly-tutorial/)
# 
# For this code to run on your machine, you will need several R and Python packages installed.
# 
# - Running `sudo pip install <package_name>` from your terminal will install a Python package.
# 
# - Running `install.packages("<package_name>")` in your R console will install an R package.
# 
# You will also need to create an account with [Plotly](https://plot.ly/feed/) to receive your API key.

# In[1]:

# You can also install packages from within IPython!

# Install Python Packages
get_ipython().system(u'pip install lifelines')
get_ipython().system(u'pip install rpy2')
get_ipython().system(u'pip install plotly')
get_ipython().system(u'pip install pandas')

# Load extension that let us use magic function `%R`
get_ipython().magic(u'load_ext rpy2.ipython')

# Install R packages
get_ipython().magic(u'R install.packages("devtools")')
get_ipython().magic(u'R devtools::install_github("ropensci/plotly")')
get_ipython().magic(u'R install.packages("IOsurv")')


# ## Introduction

# [Survival analysis](http://en.wikipedia.org/wiki/Survival_analysis) is a set of statistical methods for analyzing the occurrence  of events over time. It is also used to determine the relationship of co-variates to the time-to-events, and accurately compare time-to-event between two or more groups. For example:
# 
# - Time to death in biological systems.
# - Failure time in mechanical systems.
# - How long can we expect a user to be on a website / service?
# - Time to recovery for lung cancer treatment.
# 
# The statistical term 'survival analysis' is analogous to 'reliability theory' in engineering, 'duration analysis' in economics, and 'event history analysis' in sociology.

# The two key functions in survival analysis are the *survival function* and the *hazard function*.
# 
# The **survival function**, conventionally denoted by $S$, is the probability that the event (say, death) has not occurred yet:
# 
# $$S(t) = Pr(T > t),$$
# 
# where $T$ denotes the time of death and $Pr$ the probability. Since $S$ is a probability, $0\leq S(t)\leq1$. Survival times are non-negative ($T \geq 0$) and, generally, $S(0) = 1$.
# 
# 
# The **hazard function** $h(t)$ is the event (death) rate at time $t$, conditional on survival until $t$ (i.e., $T \geq t$):
# 
# \begin{align*}
# h(t) &= \lim_{\Delta t \to 0} Pr(t \leq T \leq t + \Delta t \, | \, T \geq t) \\
#      &= \lim_{\Delta t \to 0} \frac{Pr(t \leq T \leq t + \Delta t)}{S(t)} = \frac{p(t)}{S(t)},
# \end{align*}
# 
# where $p$ denotes the probability density function.
# 
# In practice, we do not get to observe the actual survival function of a population; we must use the observed data to estimate it. A popular estimate for the survival function $S(t)$ is the [Kaplan–Meier estimate](http://en.wikipedia.org/wiki/Kaplan–Meier_estimator):
# 
# \begin{align*}
# \hat{S}(t) &= \prod_{t_i \leq t} \frac{n_i − d_i}{n_i}\,,
# \end{align*}
# 
# where $d_i$ is the number of events (deaths) observed at time $t_i$ and $n_i$ is the number of subjects at risk observed at time $t_i$.

# ## Censoring

# Censoring is a type of missing data problem common in survival analysis. Other popular comparison methods, such as linear regression and t-tests do not accommodate for censoring. This makes survival analysis attractive for data from randomized clinical studies. 
# 
# In an ideal scenario, both the birth and death rates of a patient is known, which means the lifetime is known.
# 
# **Right censoring** occurs when the 'death' is unknown, but it is after some known date. e.g. The 'death' occurs after the end of the study, or there was no follow-up with the patient.
# 
# **Left censoring** occurs when the lifetime is known to be less than a certain duration. e.g. Unknown time of initial infection exposure when first meeting with a patient.
# 
# <hr>

# For following analysis, we will use the [lifelines](https://github.com/CamDavidsonPilon/lifelines) library for python, and the [survival](http://cran.r-project.org/web/packages/survival/survival.pdf) package for R. We can use [rpy2](http://rpy.sourceforge.net) to execute R code in the same document as the python code.
# 
# 

# In[2]:

# OIserve contains the survival package and sample datasets
get_ipython().magic(u'R library(OIsurv)')
get_ipython().magic(u'R library(devtools)')
get_ipython().magic(u'R library(plotly)')
get_ipython().magic(u'R library(IRdisplay)')

# Authenticate to plotly's api using your account
get_ipython().magic(u'R py <- plotly("rmdk", "0sn825k4r8")')

# Load python libraries
import numpy as np
import pandas as pd
import lifelines as ll

# Plotting helpers
from IPython.display import HTML
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls   
from plotly.graph_objs import *

from pylab import rcParams
rcParams['figure.figsize']=10, 5


# ## Loading data into Python and R
# 
# We will be using the `tongue` dataset from the `KMsurv` package in R, then convert the data into a pandas dataframe under the same name.
# 
# 
# This data frame contains the following columns:
# 
# - type: Tumor DNA profile (1=Aneuploid Tumor, 2=Diploid Tumor) 
# - time: Time to death or on-study time, weeks
# - delta Death indicator (0=alive, 1=dead)

# In[3]:

# Load in data
get_ipython().magic(u'R data(tongue)')
# Pull data into python kernel
get_ipython().magic(u'Rpull tongue')
# Convert into pandas dataframe
from rpy2.robjects import pandas2ri

tongue = pandas2ri.ri2py_dataframe(tongue)


# We can now refer to `tongue` using both R and python.

# In[4]:

get_ipython().run_cell_magic(u'R', u'', u'summary(tongue)')


# In[5]:

tongue.describe()


# We can even operate on R and Python within the same code cell.

# In[6]:

get_ipython().magic(u'R print(mean(tongue$time))')

print tongue['time'].mean()


# In R we need to create a `Surv` object with the `Surv()` function. Most functions in the `survival` package apply methods to this object. For right-censored data, we need to pass two arguments to `Surv()`:
# 
# 1. a vector of times
# 2. a vector indicating which times are observed and censored

# In[7]:

get_ipython().run_cell_magic(u'R', u'', u'attach(tongue)\n\ntongue.surv <- Surv(time[type==1], delta[type==1])\n\ntongue.surv')


# - The plus-signs identify observations that are right-censored.

# # Estimating survival with Kaplan-Meier
# 
# ### Using R

# The simplest fit estimates a survival object against an intercept. However, the `survfit()` function has several optional arguments. For example, we can change the confidence interval using `conf.int` and `conf.type`. 
# 
# See `help(survfit.formula)` for the comprehensive documentation.

# In[8]:

get_ipython().run_cell_magic(u'R', u'', u'surv.fit <- survfit(tongue.surv~1)\nsurv.fit')


# It is often helpful to call the `summary()` and `plot()` functions on this object.

# In[9]:

get_ipython().run_cell_magic(u'R', u'', u'summary(surv.fit)')


# In[10]:

get_ipython().run_cell_magic(u'R', u'-h 400', u"plot(surv.fit, main='Kaplan-Meier estimate with 95% confidence bounds',\n    xlab='time', ylab='survival function')")


# Let's convert this plot into an interactive plotly object using [plotly](https://plot.ly) and [ggplot2](http://ggplot2.org). 
# 
# First, we will use a helper ggplot function written by [Edwin Thoen](http://www.r-statistics.com/2013/07/creating-good-looking-survival-curves-the-ggsurv-function/) to plot pretty survival distributions in R. 

# In[11]:

get_ipython().run_cell_magic(u'R', u'', u"\nggsurv <- function(s, CI = 'def', plot.cens = T, surv.col = 'gg.def',\n                   cens.col = 'red', lty.est = 1, lty.ci = 2,\n                   cens.shape = 3, back.white = F, xlab = 'Time',\n                   ylab = 'Survival', main = ''){\n \n  library(ggplot2)\n  strata <- ifelse(is.null(s$strata) ==T, 1, length(s$strata))\n  stopifnot(length(surv.col) == 1 | length(surv.col) == strata)\n  stopifnot(length(lty.est) == 1 | length(lty.est) == strata)\n \n  ggsurv.s <- function(s, CI = 'def', plot.cens = T, surv.col = 'gg.def',\n                       cens.col = 'red', lty.est = 1, lty.ci = 2,\n                       cens.shape = 3, back.white = F, xlab = 'Time',\n                       ylab = 'Survival', main = ''){\n \n    dat <- data.frame(time = c(0, s$time),\n                      surv = c(1, s$surv),\n                      up = c(1, s$upper),\n                      low = c(1, s$lower),\n                      cens = c(0, s$n.censor))\n    dat.cens <- subset(dat, cens != 0)\n \n    col <- ifelse(surv.col == 'gg.def', 'black', surv.col)\n \n    pl <- ggplot(dat, aes(x = time, y = surv)) +\n      xlab(xlab) + ylab(ylab) + ggtitle(main) +\n      geom_step(col = col, lty = lty.est)\n \n    pl <- if(CI == T | CI == 'def') {\n      pl + geom_step(aes(y = up), color = col, lty = lty.ci) +\n        geom_step(aes(y = low), color = col, lty = lty.ci)\n    } else (pl)\n \n    pl <- if(plot.cens == T & length(dat.cens) > 0){\n      pl + geom_point(data = dat.cens, aes(y = surv), shape = cens.shape,\n                       col = cens.col)\n    } else if (plot.cens == T & length(dat.cens) == 0){\n      stop ('There are no censored observations')\n    } else(pl)\n \n    pl <- if(back.white == T) {pl + theme_bw()\n    } else (pl)\n    pl\n  }\n \n  ggsurv.m <- function(s, CI = 'def', plot.cens = T, surv.col = 'gg.def',\n                       cens.col = 'red', lty.est = 1, lty.ci = 2,\n                       cens.shape = 3, back.white = F, xlab = 'Time',\n                       ylab = 'Survival', main = '') {\n    n <- s$strata\n \n    groups <- factor(unlist(strsplit(names\n                                     (s$strata), '='))[seq(2, 2*strata, by = 2)])\n    gr.name <-  unlist(strsplit(names(s$strata), '='))[1]\n    gr.df <- vector('list', strata)\n    ind <- vector('list', strata)\n    n.ind <- c(0,n); n.ind <- cumsum(n.ind)\n    for(i in 1:strata) ind[[i]] <- (n.ind[i]+1):n.ind[i+1]\n \n    for(i in 1:strata){\n      gr.df[[i]] <- data.frame(\n        time = c(0, s$time[ ind[[i]] ]),\n        surv = c(1, s$surv[ ind[[i]] ]),\n        up = c(1, s$upper[ ind[[i]] ]),\n        low = c(1, s$lower[ ind[[i]] ]),\n        cens = c(0, s$n.censor[ ind[[i]] ]),\n        group = rep(groups[i], n[i] + 1))\n    }\n \n    dat <- do.call(rbind, gr.df)\n    dat.cens <- subset(dat, cens != 0)\n \n    pl <- ggplot(dat, aes(x = time, y = surv, group = group)) +\n      xlab(xlab) + ylab(ylab) + ggtitle(main) +\n      geom_step(aes(col = group, lty = group))\n \n    col <- if(length(surv.col == 1)){\n      scale_colour_manual(name = gr.name, values = rep(surv.col, strata))\n    } else{\n      scale_colour_manual(name = gr.name, values = surv.col)\n    }\n \n    pl <- if(surv.col[1] != 'gg.def'){\n      pl + col\n    } else {pl + scale_colour_discrete(name = gr.name)}\n \n    line <- if(length(lty.est) == 1){\n      scale_linetype_manual(name = gr.name, values = rep(lty.est, strata))\n    } else {scale_linetype_manual(name = gr.name, values = lty.est)}\n \n    pl <- pl + line\n \n    pl <- if(CI == T) {\n      if(length(surv.col) > 1 && length(lty.est) > 1){\n        stop('Either surv.col or lty.est should be of length 1 in order\n             to plot 95% CI with multiple strata')\n      }else if((length(surv.col) > 1 | surv.col == 'gg.def')[1]){\n        pl + geom_step(aes(y = up, color = group), lty = lty.ci) +\n          geom_step(aes(y = low, color = group), lty = lty.ci)\n      } else{pl +  geom_step(aes(y = up, lty = group), col = surv.col) +\n               geom_step(aes(y = low,lty = group), col = surv.col)}\n    } else {pl}\n \n \n    pl <- if(plot.cens == T & length(dat.cens) > 0){\n      pl + geom_point(data = dat.cens, aes(y = surv), shape = cens.shape,\n                      col = cens.col)\n    } else if (plot.cens == T & length(dat.cens) == 0){\n      stop ('There are no censored observations')\n    } else(pl)\n \n    pl <- if(back.white == T) {pl + theme_bw()\n    } else (pl)\n    pl\n  }\n  pl <- if(strata == 1) {ggsurv.s(s, CI , plot.cens, surv.col ,\n                                  cens.col, lty.est, lty.ci,\n                                  cens.shape, back.white, xlab,\n                                  ylab, main)\n  } else {ggsurv.m(s, CI, plot.cens, surv.col ,\n                   cens.col, lty.est, lty.ci,\n                   cens.shape, back.white, xlab,\n                   ylab, main)}\n  pl\n}")


# Voila!

# In[12]:

get_ipython().run_cell_magic(u'R', u'-h 400', u'p <- ggsurv(surv.fit) + theme_bw()\np')


# We have to use a workaround to render an interactive plotly object by using an iframe in the ipython kernel. This is a bit easier if you are working in an R kernel.

# In[13]:

get_ipython().run_cell_magic(u'R', u'', u'# Create the iframe HTML\nplot.ly <- function(url) {\n    # Set width and height from options or default square\n    w <- "750"\n    h <- "600"\n    html <- paste("<center><iframe height=\\"", h, "\\" id=\\"igraph\\" scrolling=\\"no\\" seamless=\\"seamless\\"\\n\\t\\t\\t\\tsrc=\\"", \n        url, "\\" width=\\"", w, "\\" frameBorder=\\"0\\"></iframe></center>", sep="")\n    return(html)\n}')


# In[14]:

get_ipython().magic(u'R p <- plot.ly("https://plot.ly/~rmdk/111/survival-vs-time/")')
# pass object to python kernel
get_ipython().magic(u'R -o p')

# Render HTML
HTML(p[0])


# The `y axis` represents the probability a patient is still alive at time $t$ weeks. We see a steep drop off within the first 100 weeks, and then observe the curve flattening. The dotted lines represent the 95% confidence intervals.

# ### Using Python

# We will now replicate the above steps using python. Above, we have already specified a variable `tongues` that holds the data in a pandas dataframe.

# In[15]:

from lifelines.estimation import KaplanMeierFitter
kmf = KaplanMeierFitter()


# The method takes the same parameters as it's R counterpart, a time vector and a vector indicating which observations are observed or censored. The model fitting sequence is similar to the [scikit-learn](http://scikit-learn.org/stable/) api.

# In[16]:

f = tongue.type==1
T = tongue[f]['time']
C = tongue[f]['delta']

kmf.fit(T, event_observed=C)


# To get a plot with the confidence intervals, we simply can call `plot()` on our `kmf` object.

# In[17]:

kmf.plot(title='Tumor DNA Profile 1')


# Now we can convert this plot to an interactive [Plotly](https://plot.ly) object. However, we will have to augment the legend and filled area manually. Once we create a helper function, the process is simple.
# 
# Please see the Plotly Python [user guide](https://plot.ly/python/overview/#in-%5B37%5D) for more insight on how to update plot parameters. 
# 
# > Don't forget you can also easily edit the chart properties using the Plotly GUI interface by clicking the "Play with this data!" link below the chart.

# In[19]:

p = kmf.plot(ci_force_lines=True, title='Tumor DNA Profile 1 (95% CI)')

# Collect the plot object
kmf1 = plt.gcf() 

def pyplot(fig, ci=True, legend=True):
    # Convert mpl fig obj to plotly fig obj, resize to plotly's default
    py_fig = tls.mpl_to_plotly(fig, resize=True)
    
    # Add fill property to lower limit line
    if ci == True:
        style1 = dict(fill='tonexty')
        # apply style
        py_fig['data'][2].update(style1)
        
        # Change color scheme to black
        py_fig['data'].update(dict(line=Line(color='black')))
    
    # change the default line type to 'step'
    py_fig['data'].update(dict(line=Line(shape='hv')))
    # Delete misplaced legend annotations 
    py_fig['layout'].pop('annotations', None)
    
    if legend == True:
        # Add legend, place it at the top right corner of the plot
        py_fig['layout'].update(
            showlegend=True,
            legend=Legend(
                x=1.05,
                y=1
            )
        )
        
    # Send updated figure object to Plotly, show result in notebook
    return py.iplot(py_fig)

pyplot(kmf1, legend=False)


# <hr>
# # Multiple Types
# 
# ### Using R

# Many times there are different groups contained in a single dataset. These may represent categories such as treatment groups, different species, or different manufacturing techniques. The `type` variable in the `tongues` dataset describes a patients DNA profile. Below we define a Kaplan-Meier estimate for each of these groups in R and Python.

# In[19]:

get_ipython().run_cell_magic(u'R', u'', u"\nsurv.fit2 <- survfit( Surv(time, delta) ~ type)\n\np <- ggsurv(surv.fit2) + \n        ggtitle('Lifespans of different tumor DNA profile') + theme_bw()\np")


# Convert to a Plotly object.

# In[20]:

#%R py$ggplotly(plt)

get_ipython().magic(u'R p <- plot.ly("https://plot.ly/~rmdk/173/lifespans-of-different-tumor-dna-profile/")')
# pass object to python kernel
get_ipython().magic(u'R -o p')

# Render HTML
HTML(p[0])


# ### Using Python

# In[21]:

f2 = tongue.type==2
T2 = tongue[f2]['time']
C2 = tongue[f2]['delta']

ax = plt.subplot(111)

kmf.fit(T, event_observed=C, label=['Type 1 DNA'])
kmf.survival_function_.plot(ax=ax)
kmf.fit(T2, event_observed=C2, label=['Type 2 DNA'])
kmf.survival_function_.plot(ax=ax)

plt.title('Lifespans of different tumor DNA profile')

kmf2 = plt.gcf()


# Convert to a Plotly object.

# In[25]:

pyplot(kmf2, ci=False)


# <hr>
# # Testing for Difference

# It looks like DNA Type 2 is potentially more deadly, or more difficult to treat compared to Type 1. However, the difference between these survival curves still does not seem dramatic. It will be useful to perform a statistical test on the different DNA profiles to see if their survival rates are significantly different.
# 
# Python's *lifelines* contains methods in `lifelines.statistics`, and the R package `survival` uses a function `survdiff()`. Both functions return a p-value from a chi-squared distribution.
# 
# It turns out these two DNA types do not have significantly different survival rates.

# ### Using R

# In[31]:

get_ipython().run_cell_magic(u'R', u'', u'survdiff(Surv(time, delta) ~ type)')


# ### Using Python

# In[32]:

from lifelines.statistics import logrank_test
summary_= logrank_test(T, T2, C, C2, alpha=99)

print summary_


# <hr>
# # Estimating Hazard Rates
# 
# ### Using R

# To estimate the hazard function, we compute the cumulative hazard function using the [Nelson-Aalen estimator](), defined as:
# 
# $$\hat{\Lambda} (t) = \sum_{t_i \leq t} \frac{d_i}{n_i}$$
# 
# where $d_i$ is the number of deaths at time $t_i$ and $n_i$ is the number of susceptible individuals. Both R and Python modules use the same estimator. However, in R we will use the `-log` of the Fleming and Harrington estimator, which is equivalent to the Nelson-Aalen.

# In[33]:

get_ipython().run_cell_magic(u'R', u' ', u"\nhaz <- Surv(time[type==1], delta[type==1])\nhaz.fit  <- summary(survfit(haz ~ 1), type='fh')\n\nx <- c(haz.fit$time, 250)\ny <- c(-log(haz.fit$surv), 1.474)\ncum.haz <- data.frame(time=x, cumulative.hazard=y)\n\np <- ggplot(cum.haz, aes(time, cumulative.hazard)) + geom_step() + theme_bw() + \n        ggtitle('Nelson-Aalen Estimate')\np")


# In[23]:

get_ipython().magic(u'R p <- plot.ly("https://plot.ly/~rmdk/185/cumulativehazard-vs-time/")')
# pass object to python kernel
get_ipython().magic(u'R -o p')

# Render HTML
HTML(p[0])


# ### Using Python

# In[26]:

from lifelines.estimation import NelsonAalenFitter

naf = NelsonAalenFitter()
naf.fit(T, event_observed=C)

naf.plot(title='Nelson-Aalen Estimate')


# In[27]:

naf.plot(ci_force_lines=True, title='Nelson-Aalen Estimate')
py_p = plt.gcf()

pyplot(py_p, legend=False)

