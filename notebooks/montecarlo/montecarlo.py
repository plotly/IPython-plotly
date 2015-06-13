
# coding: utf-8

# # Computational Methods in Bayesian Analysis

# ###Introduction
# 
# For most problems of interest, Bayesian analysis requires integration over multiple parameters, making the calculation of a [posterior](https://en.wikipedia.org/wiki/Posterior_probability) intractable whether via analytic methods or standard methods of numerical integration.
# 
# However, it is often possible to *approximate* these integrals by drawing samples
# from posterior distributions. For example, consider the expected value (mean) of a vector-valued random variable $\mathbf{x}$:
# 
# $$
# E[\mathbf{x}] = \int \mathbf{x} f(\mathbf{x}) \mathrm{d}\mathbf{x}\,, \quad
# \mathbf{x} = \{x_1, \ldots, x_k\}
# $$
# 
# where $k$ (dimension of vector $\mathbf{x}$) is perhaps very large.

# If we can produce a reasonable number of random vectors $\{{\bf x_i}\}$, we can use these values to approximate the unknown integral. This process is known as [**Monte Carlo integration**](https://en.wikipedia.org/wiki/Monte_Carlo_integration). In general, Monte Carlo integration allows integrals against probability density functions
# 
# $$
# I = \int h(\mathbf{x}) f(\mathbf{x}) \mathrm{d}\mathbf{x}
# $$
# 
# to be estimated by finite sums
# 
# $$
# \hat{I} = \frac{1}{n}\sum_{i=1}^n h(\mathbf{x}_i),
# $$
# 
# where $\mathbf{x}_i$ is a sample from $f$. This estimate is valid and useful because:
# 
# - $\hat{I} \rightarrow I$ with probability $1$ by the [strong law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers#Strong_law);
# 
# - simulation error can be measured and controlled.

# ### Example (Negative Binomial Distribution)
# 
# We can use this kind of simulation to estimate the expected value of a random variable that is negative binomial-distributed. The [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution) applies to discrete positive random variables. It can be used to model the number of Bernoulli trials that one can expect to conduct until $r$ failures occur.

# The [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) reads
# 
# $$
# f(k \mid p, r) = {k + r - 1 \choose k} (1 - p)^k p^r\,,
# $$
# 
# where $k \in \{0, 1, 2, \ldots \}$ is the value taken by our non-negative discrete random variable and
# $p$ is the probability of success ($0 < p < 1$).
# 
# 
# ![negative binomial (courtesy Wikipedia)](http://upload.wikimedia.org/wikipedia/commons/8/83/Negbinomial.gif)

# Most frequently, this distribution is used to model *overdispersed counts*, that is, counts that have variance larger
# than the mean (i.e., what would be predicted under a
# [Poisson distribution](http://en.wikipedia.org/wiki/Poisson_distribution)).
# 
# In fact, the negative binomial can be expressed as a continuous mixture of Poisson distributions,
# where a [gamma distributions](http://en.wikipedia.org/wiki/Gamma_distribution) act as mixing weights:
# 
# $$
# f(k \mid p, r) = \int_0^{\infty} \text{Poisson}(k \mid \lambda) \,
# \text{Gamma}_{(r, (1 - p)/p)}(\lambda) \, \mathrm{d}\lambda,
# $$
# 
# where the parameters of the gamma distribution are denoted as (shape parameter, inverse scale parameter).
# 
# Let's resort to simulation to estimate the mean of a negative binomial distribution with $p = 0.7$ and $r = 3$:

# In[1]:

import numpy as np

r = 3
p = 0.7


# In[2]:

# Simulate Gamma means (r: shape parameter; p / (1 - p): scale parameter).
lam = np.random.gamma(r, p / (1 - p), size=100)


# In[3]:

# Simulate sample Poisson conditional on lambda.
sim_vals = np.random.poisson(lam)


# In[4]:

sim_vals.mean()


# The actual expected value of the negative binomial distribution is $r p / (1 - p)$, which in this case is 7. That's pretty close, though we can do better if we draw more samples:

# In[5]:

lam = np.random.gamma(r, p / (1 - p), size=100000)
sim_vals = np.random.poisson(lam)
sim_vals.mean()


# This approach of drawing repeated random samples in order to obtain a desired numerical result is generally known as **Monte Carlo simulation**.
# 
# Clearly, this is a convenient, simplistic example that did not require simuation to obtain an answer. For most problems, it is simply not possible to draw independent random samples from the posterior distribution because they will generally be (1) multivariate and (2) not of a known functional form for which there is a pre-existing random number generator.
# 
# However, we are not going to give up on simulation. Though we cannot generally draw independent samples for our model, we can usually generate *dependent* samples, and it turns out that if we do this in a particular way, we can obtain samples from almost any posterior distribution.

# ## Markov Chains
# 
# A Markov chain is a special type of *stochastic process*. The standard definition of a stochastic process is an ordered collection of random variables:
# 
# $$\begin{gathered}
# \begin{split}\{X_t:t \in T\}\end{split}\notag\\\begin{split}\end{split}\notag
# \end{gathered}$$
# 
# where $t$ is frequently (but not necessarily) a time index. If we think of $X_t$ as a state $X$ at time $t$, and invoke the following dependence condition on each state:
# 
# \\[\begin{aligned}
# &Pr(X_{t+1}=x_{t+1} | X_t=x_t, X_{t-1}=x_{t-1},\ldots,X_0=x_0) \cr
# &= Pr(X_{t+1}=x_{t+1} | X_t=x_t)
# \end{aligned}\\]
# 
# then the stochastic process is known as a Markov chain. This conditioning specifies that the future depends on the current state, but not past states. Thus, the Markov chain wanders about the state space,
# remembering only where it has just been in the last time step. 
# 
# The collection of transition probabilities is sometimes called a *transition matrix* when dealing with discrete states, or more generally, a *transition kernel*.
# 
# It is useful to think of the Markovian property as **mild non-independence**. 
# 
# If we use Monte Carlo simulation to generate a Markov chain, this is called **Markov chain Monte Carlo**, or MCMC. If the resulting Markov chain obeys some important properties, then it allows us to indirectly generate independent samples from a particular posterior distribution.
# 
# 
# > ### Why MCMC Works: Reversible Markov Chains
# > 
# > Markov chain Monte Carlo simulates a Markov chain for which some function of interest
# > (*e.g.* the joint distribution of the parameters of some model) is the unique, invariant limiting distribution. An invariant distribution with respect to some Markov chain with transition kernel $Pr(y \mid x)$ implies that:
# > 
# > $$\int_x Pr(y \mid x) \pi(x) dx = \pi(y).$$
# > 
# > Invariance is guaranteed for any *reversible* Markov chain. Consider a Markov chain in reverse sequence:
# > $\{\theta^{(n)},\theta^{(n-1)},...,\theta^{(0)}\}$. This sequence is still Markovian, because:
# > 
# > $$Pr(\theta^{(k)}=y \mid \theta^{(k+1)}=x,\theta^{(k+2)}=x_1,\ldots ) = Pr(\theta^{(k)}=y \mid \theta^{(k+1)}=x)$$
# > 
# > Forward and reverse transition probabilities may be related through Bayes theorem:
# > 
# > $$\frac{Pr(\theta^{(k+1)}=x \mid \theta^{(k)}=y) \pi^{(k)}(y)}{\pi^{(k+1)}(x)}$$
# > 
# > Though not homogeneous in general, $\pi$ becomes homogeneous if:
# > 
# > -   $n \rightarrow \infty$
# > 
# > -   $\pi^{(i)}=\pi$ for some $i < k$
# > 
# > If this chain is homogeneous it is called reversible, because it satisfies the ***detailed balance equation***:
# > 
# > $$\pi(x)Pr(y \mid x) = \pi(y) Pr(x \mid y)$$
# > 
# > Reversibility is important because it has the effect of balancing movement through the entire state space. When a Markov chain is reversible, $\pi$ is the unique, invariant, stationary distribution of that chain. Hence, if $\pi$ is of interest, we need only find the reversible Markov chain for which $\pi$ is the limiting distribution.
# > This is what MCMC does!

# ## Gibbs Sampling
# 
# The Gibbs sampler is the simplest and most prevalent MCMC algorithm. If a posterior has $k$ parameters to be estimated, we may condition each parameter on current values of the other $k-1$ parameters, and sample from the resultant distributional form (usually easier), and repeat this operation on the other parameters in turn. This procedure generates samples from the posterior distribution. Note that we have now combined Markov chains (conditional independence) and Monte Carlo techniques (estimation by simulation) to yield Markov chain Monte Carlo.
# 
# Here is a stereotypical Gibbs sampling algorithm:
# 
# 1.  Choose starting values for states (parameters):
#     ${\bf \theta} = [\theta_1^{(0)},\theta_2^{(0)},\ldots,\theta_k^{(0)}]$
# 
# 2.  Initialize counter $j=1$
# 
# 3.  Draw the following values from each of the $k$ conditional
#     distributions:
# 
#     $$\begin{aligned}
#     \theta_1^{(j)} &\sim& \pi(\theta_1 | \theta_2^{(j-1)},\theta_3^{(j-1)},\ldots,\theta_{k-1}^{(j-1)},\theta_k^{(j-1)}) \\
#     \theta_2^{(j)} &\sim& \pi(\theta_2 | \theta_1^{(j)},\theta_3^{(j-1)},\ldots,\theta_{k-1}^{(j-1)},\theta_k^{(j-1)}) \\
#     \theta_3^{(j)} &\sim& \pi(\theta_3 | \theta_1^{(j)},\theta_2^{(j)},\ldots,\theta_{k-1}^{(j-1)},\theta_k^{(j-1)}) \\
#     \vdots \\
#     \theta_{k-1}^{(j)} &\sim& \pi(\theta_{k-1} | \theta_1^{(j)},\theta_2^{(j)},\ldots,\theta_{k-2}^{(j)},\theta_k^{(j-1)}) \\
#     \theta_k^{(j)} &\sim& \pi(\theta_k | \theta_1^{(j)},\theta_2^{(j)},\theta_4^{(j)},\ldots,\theta_{k-2}^{(j)},\theta_{k-1}^{(j)})\end{aligned}$$
# 
# 4.  Increment $j$ and repeat until convergence occurs.
# 
# As we can see from the algorithm, each distribution is conditioned on the last iteration of its chain values, constituting a Markov chain as advertised. The Gibbs sampler has all of the important properties outlined in the previous section: it is aperiodic, homogeneous and ergodic. Once the sampler converges, all subsequent samples are from the target distribution. This convergence occurs at a geometric rate.

# ## Example: Inferring patterns in UK coal mining disasters
# 
# Let's try to model a more interesting example, a time series of recorded coal mining 
# disasters in the UK from 1851 to 1962.
# 
# Occurrences of disasters in the time series is thought to be derived from a 
# Poisson process with a large rate parameter in the early part of the time 
# series, and from one with a smaller rate in the later part. We are interested 
# in locating the change point in the series, which perhaps is related to changes 
# in mining safety regulations.

# In[6]:

disasters_array = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

n_count_data = len(disasters_array)


# In[7]:

import plotly.plotly as py
import plotly.graph_objs as pgo


# In[8]:

data = pgo.Data([
        pgo.Scatter(
            x=[str(year) + '-01-01' for year in np.arange(1851, 1962)],
            y=disasters_array,
            mode='lines+markers'
    )
])


# In[9]:

layout = pgo.Layout(
    title='UK coal mining disasters (per year), 1851--1962',
    xaxis=pgo.XAxis(title='Year', type='date', range=['1851-01-01', '1962-01-01']),
    yaxis=pgo.YAxis(title='Disaster count')
)


# In[10]:

fig = pgo.Figure(data=data, layout=layout)


# In[11]:

py.iplot(fig, filename='coal_mining_disasters')


# We are going to use Poisson random variables for this type of count data. Denoting year $i$'s accident count by $y_i$, 
# 
# $$y_i \sim \text{Poisson}(\lambda).$$
# 
# For those unfamiliar, Poisson random variables look like this:

# In[12]:

data2 = pgo.Data([
        pgo.Histogram(
            x=np.random.poisson(l, 1000),
            opacity=0.75,
            name=u'λ=%i' % l
        ) for l in [1, 5, 12, 25]
])


# In[13]:

layout_grey_bg = pgo.Layout(
    xaxis=pgo.XAxis(zeroline=False, showgrid=True, gridcolor='rgb(255, 255, 255)'),
    yaxis=pgo.YAxis(zeroline=False, showgrid=True, gridcolor='rgb(255, 255, 255)'),
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgba(204, 204, 204, 0.5)'
)


# In[14]:

layout2 = layout_grey_bg.copy()


# In[15]:

layout2.update(
    barmode='overlay',
    title='Poisson Means',
    xaxis=pgo.XAxis(range=[0, 50]),
    yaxis=pgo.YAxis(range=[0, 400])
)


# In[16]:

fig2 = pgo.Figure(data=data2, layout=layout2)


# In[17]:

py.iplot(fig2, filename='poisson_means')


# The modeling problem is about estimating the values of the $\lambda$ parameters. Looking at the time series above, it appears that the rate declines over time.
# 
# A **changepoint model** identifies a point (here, a year) after which the parameter $\lambda$ drops to a lower value. Let us call this point in time $\tau$. So we are estimating two $\lambda$ parameters:
# 
# $$
# \lambda = 
# \begin{cases}
# \lambda_1  & \text{if } t < \tau, \cr
# \lambda_2 & \text{if } t \geq \tau.
# \end{cases}
# $$
# 
# We need to assign prior probabilities to both $\{\lambda_1, \lambda_2\}$. The gamma distribution not only provides a continuous density function for positive numbers, but it is also *conjugate* with the Poisson sampling distribution. 

# In[18]:

lambda1_lambda2 = [(0.1, 100), (1, 100), (1, 10), (10, 10)]


# In[19]:

data3 = pgo.Data([
        pgo.Histogram(
            x=np.random.gamma(*p, size=1000),
            opacity=0.75,
            name=u'α=%i, β=%i' % (p[0], p[1]))
        for p in lambda1_lambda2
])


# In[20]:

layout3 = layout_grey_bg.copy()
layout3.update(
    barmode='overlay',
    xaxis=pgo.XAxis(range=[0, 300])
)


# In[21]:

fig3 = pgo.Figure(data=data3, layout=layout3)


# In[22]:

py.iplot(fig3, filename='gamma_distributions')


# We will specify suitably vague hyperparameters $\alpha$ and $\beta$ for both priors:
# 
# \begin{align}
# \lambda_1 &\sim \text{Gamma}(1, 10), \\
# \lambda_2 &\sim \text{Gamma}(1, 10).
# \end{align}
# 
# Since we do not have any intuition about the location of the changepoint (unless we visualize the data), we will assign a discrete uniform prior over the entire observation period [1851, 1962]:
# 
# \begin{align}
# &\tau \sim \text{DiscreteUniform(1851, 1962)}\\
# &\Rightarrow P(\tau = k) = \frac{1}{111}.
# \end{align}

# ### Implementing Gibbs sampling
# 
# We are interested in estimating the joint posterior of $\lambda_1, \lambda_2$ and $\tau$ given the array of annnual disaster counts $\mathbf{y}$. This gives:
# 
# $$
#  P( \lambda_1, \lambda_2, \tau | \mathbf{y} ) \propto P(\mathbf{y} | \lambda_1, \lambda_2, \tau ) P(\lambda_1, \lambda_2, \tau) 
# $$
# 
# To employ Gibbs sampling, we need to factor the joint posterior into the product of conditional expressions:
# 
# $$
#  P( \lambda_1, \lambda_2, \tau | \mathbf{y} ) \propto P(y_{t<\tau} | \lambda_1, \tau) P(y_{t\ge \tau} | \lambda_2, \tau) P(\lambda_1) P(\lambda_2) P(\tau)
# $$
# 
# which we have specified as:
# 
# $$\begin{aligned}
# P( \lambda_1, \lambda_2, \tau | \mathbf{y} ) &\propto \left[\prod_{t=1851}^{\tau} \text{Poi}(y_t|\lambda_1) \prod_{t=\tau+1}^{1962} \text{Poi}(y_t|\lambda_2) \right] \text{Gamma}(\lambda_1|\alpha,\beta) \text{Gamma}(\lambda_2|\alpha, \beta) \frac{1}{111} \\
# &\propto \left[\prod_{t=1851}^{\tau} e^{-\lambda_1}\lambda_1^{y_t} \prod_{t=\tau+1}^{1962} e^{-\lambda_2} \lambda_2^{y_t} \right] \lambda_1^{\alpha-1} e^{-\beta\lambda_1} \lambda_2^{\alpha-1} e^{-\beta\lambda_2} \\
# &\propto \lambda_1^{\sum_{t=1851}^{\tau} y_t +\alpha-1} e^{-(\beta+\tau)\lambda_1} \lambda_2^{\sum_{t=\tau+1}^{1962} y_i + \alpha-1} e^{-\beta\lambda_2}
# \end{aligned}$$
# 
# So, the full conditionals are known, and critically for Gibbs, can easily be sampled from.
# 
# $$\lambda_1 \sim \text{Gamma}(\sum_{t=1851}^{\tau} y_t +\alpha, \tau+\beta)$$
# $$\lambda_2 \sim \text{Gamma}(\sum_{t=\tau+1}^{1962} y_i + \alpha, 1962-\tau+\beta)$$
# $$\tau \sim \text{Categorical}\left( \frac{\lambda_1^{\sum_{t=1851}^{\tau} y_t +\alpha-1} e^{-(\beta+\tau)\lambda_1} \lambda_2^{\sum_{t=\tau+1}^{1962} y_i + \alpha-1} e^{-\beta\lambda_2}}{\sum_{k=1851}^{1962} \lambda_1^{\sum_{t=1851}^{\tau} y_t +\alpha-1} e^{-(\beta+\tau)\lambda_1} \lambda_2^{\sum_{t=\tau+1}^{1962} y_i + \alpha-1} e^{-\beta\lambda_2}} \right)$$
# 
# Implementing this in Python requires random number generators for both the gamma and discrete uniform distributions. We can leverage NumPy for this:

# In[23]:

# Function to draw random gamma variate
rgamma = np.random.gamma

def rcategorical(probs, n=None):
    # Function to draw random categorical variate
    return np.array(probs).cumsum().searchsorted(np.random.sample(n))


# Next, in order to generate probabilities for the conditional posterior of $\tau$, we need the kernel of the gamma density:
# 
# \\[\lambda^{\alpha-1} e^{-\beta \lambda}\\]

# In[24]:

dgamma = lambda lam, a, b: lam**(a - 1) * np.exp(-b * lam)


# Diffuse hyperpriors for the gamma priors on $\{\lambda_1, \lambda_2\}$:

# In[25]:

alpha, beta = 1., 10


# For computational efficiency, it is best to pre-allocate memory to store the sampled values. We need 3 arrays, each with length equal to the number of iterations we plan to run:

# In[26]:

# Specify number of iterations
n_iterations = 1000

# Initialize trace of samples
lambda1, lambda2, tau = np.empty((3, n_iterations + 1))


# The penultimate step initializes the model paramters to arbitrary values:

# In[27]:

lambda1[0] = 6
lambda2[0] = 2
tau[0] = 50


# Now we can run the Gibbs sampler.

# In[28]:

# Sample from conditionals
for i in range(n_iterations):
    
    # Sample early mean
    lambda1[i + 1] = rgamma(disasters_array[:tau[i]].sum() + alpha, 1./(tau[i] + beta))
    
    # Sample late mean
    lambda2[i + 1] = rgamma(disasters_array[tau[i]:].sum() + alpha,
                            1./(n_count_data - tau[i] + beta))
    
    # Sample changepoint: first calculate probabilities (conditional)
    p = np.array([dgamma(lambda1[i + 1], disasters_array[:t].sum() + alpha, t + beta) *
                  dgamma(lambda2[i + 1], disasters_array[t:].sum() + alpha, n_count_data - t + beta)
                  for t in range(n_count_data)])
    
    # ... then draw sample
    tau[i + 1] = rcategorical(p/p.sum())


# Plotting the trace and histogram of the samples reveals the marginal posteriors of each parameter in the model.

# In[29]:

color = '#3182bd'


# In[30]:

trace1 = pgo.Scatter(
    y=lambda1,
    xaxis='x1',
    yaxis='y1',
    line=pgo.Line(width=1),
    marker=pgo.Marker(color=color)
)

trace2 = pgo.Histogram(
    x=lambda1,
    xaxis='x2',
    yaxis='y2',
    line=pgo.Line(width=0.5),
    marker=pgo.Marker(color=color)
)

trace3 = pgo.Scatter(
    y=lambda2,
    xaxis='x3',
    yaxis='y3',
    line=pgo.Line(width=1),
    marker=pgo.Marker(color=color)
)

trace4 = pgo.Histogram(
    x=lambda2,
    xaxis='x4',
    yaxis='y4',
    marker=pgo.Marker(color=color)
)

trace5 = pgo.Scatter(
    y=tau,
    xaxis='x5',
    yaxis='y5',
    line=pgo.Line(width=1),
    marker=pgo.Marker(color=color)
)

trace6 = pgo.Histogram(
    x=tau,
    xaxis='x6',
    yaxis='y6',
    marker=pgo.Marker(color=color)
)


# In[31]:

data4 = pgo.Data([trace1, trace2, trace3, trace4, trace5, trace6])


# In[32]:

import plotly.tools as tls


# In[33]:

fig4 = tls.make_subplots(3, 2)


# In[34]:

fig4['data'] += data4


# In[35]:

def add_style(fig):
    for i in fig['layout'].keys():
        fig['layout'][i]['zeroline'] = False
        fig['layout'][i]['showgrid'] = True
        fig['layout'][i]['gridcolor'] = 'rgb(255, 255, 255)'
    fig['layout']['paper_bgcolor'] = 'rgb(255, 255, 255)'
    fig['layout']['plot_bgcolor'] = 'rgba(204, 204, 204, 0.5)'
    fig['layout']['showlegend']=False


# In[36]:

add_style(fig4)


# In[37]:

fig4['layout'].update(
    yaxis1=pgo.YAxis(title=r'$\lambda_1$'),
    yaxis3=pgo.YAxis(title=r'$\lambda_2$'),
    yaxis5=pgo.YAxis(title=r'$\tau$'))


# In[38]:

py.iplot(fig4, filename='modelling_params')


# ## The Metropolis-Hastings Algorithm
# 
# The key to success in applying the Gibbs sampler to the estimation of Bayesian posteriors is being able to specify the form of the complete conditionals of
# ${\bf \theta}$, because the algorithm cannot be implemented without them. In practice, the posterior conditionals cannot always be neatly specified. 
# 
# 
# Taking a different approach, the Metropolis-Hastings algorithm generates ***candidate***  state transitions from an alternate distribution, and *accepts* or *rejects* each candidate probabilistically.
# 
# Let us first consider a simple Metropolis-Hastings algorithm for a single parameter, $\theta$. We will use a standard sampling distribution, referred to as the *proposal distribution*, to produce candidate variables $q_t(\theta^{\prime} | \theta)$. That is, the generated value, $\theta^{\prime}$, is a *possible* next value for
# $\theta$ at step $t+1$. We also need to be able to calculate the probability of moving back to the original value from the candidate, or
# $q_t(\theta | \theta^{\prime})$. These probabilistic ingredients are used to define an *acceptance ratio*:
# 
# $$\begin{gathered}
# \begin{split}a(\theta^{\prime},\theta) = \frac{q_t(\theta^{\prime} | \theta) \pi(\theta^{\prime})}{q_t(\theta | \theta^{\prime}) \pi(\theta)}\end{split}\notag\\\begin{split}\end{split}\notag\end{gathered}$$
# 
# The value of $\theta^{(t+1)}$ is then determined by:
# 
# $$\theta^{(t+1)} = \left\{\begin{array}{l@{\quad \mbox{with prob.} \quad}l}\theta^{\prime} & \text{with probability } \min(a(\theta^{\prime},\theta^{(t)}),1) \\ \theta^{(t)} & \text{with probability } 1 - \min(a(\theta^{\prime},\theta^{(t)}),1) \end{array}\right.$$
# 
# This transition kernel implies that movement is not guaranteed at every step. It only occurs if the suggested transition is likely based on the acceptance ratio.
# 
# A single iteration of the Metropolis-Hastings algorithm proceeds as follows:
# 
# 1.  Sample $\theta^{\prime}$ from $q(\theta^{\prime} | \theta^{(t)})$.
# 
# 2.  Generate a Uniform[0,1] random variate $u$.
# 
# 3.  If $a(\theta^{\prime},\theta) > u$ then
#     $\theta^{(t+1)} = \theta^{\prime}$, otherwise
#     $\theta^{(t+1)} = \theta^{(t)}$.
# 
# The original form of the algorithm specified by Metropolis required that
# $q_t(\theta^{\prime} | \theta) = q_t(\theta | \theta^{\prime})$, which reduces $a(\theta^{\prime},\theta)$ to
# $\pi(\theta^{\prime})/\pi(\theta)$, but this is not necessary. In either case, the state moves to high-density points in the distribution with high probability, and to low-density points with low probability. After convergence, the Metropolis-Hastings algorithm describes the full target posterior density, so all points are recurrent.
# 
# ### Random-walk Metropolis-Hastings
# 
# A practical implementation of the Metropolis-Hastings algorithm makes use of a random-walk proposal.
# Recall that a random walk is a Markov chain that evolves according to:
# 
# $$
# \theta^{(t+1)} = \theta^{(t)} + \epsilon_t \\
# \epsilon_t \sim f(\phi)
# $$
# 
# As applied to the MCMC sampling, the random walk is used as a proposal distribution, whereby dependent proposals are generated according to:
# 
# $$\begin{gathered}
# \begin{split}q(\theta^{\prime} | \theta^{(t)}) = f(\theta^{\prime} - \theta^{(t)}) = \theta^{(t)} + \epsilon_t\end{split}\notag\\\begin{split}\end{split}\notag\end{gathered}$$
# 
# Generally, the density generating $\epsilon_t$ is symmetric about zero,
# resulting in a symmetric chain. Chain symmetry implies that
# $q(\theta^{\prime} | \theta^{(t)}) = q(\theta^{(t)} | \theta^{\prime})$,
# which reduces the Metropolis-Hastings acceptance ratio to:
# 
# $$\begin{gathered}
# \begin{split}a(\theta^{\prime},\theta) = \frac{\pi(\theta^{\prime})}{\pi(\theta)}\end{split}\notag\\\begin{split}\end{split}\notag\end{gathered}$$
# 
# The choice of the random walk distribution for $\epsilon_t$ is frequently a normal or Student’s $t$ density, but it may be any distribution that generates an irreducible proposal chain.
# 
# An important consideration is the specification of the **scale parameter** for the random walk error distribution. Large values produce random walk steps that are highly exploratory, but tend to produce proposal values in the tails of the target distribution, potentially resulting in very small acceptance rates. Conversely, small values tend to be accepted more frequently, since they tend to produce proposals close to the current parameter value, but may result in chains that ***mix*** very slowly.
# 
# Some simulation studies suggest optimal acceptance rates in the range of 20-50%. It is often worthwhile to optimize the proposal variance by iteratively adjusting its value, according to observed acceptance rates early in the MCMC simulation .

# ## Example: Linear model estimation
# 
# This very simple dataset is a selection of real estate prices \\(p\\), with the associated age \\(a\\) of each house. We wish to estimate a simple linear relationship between the two variables, using the Metropolis-Hastings algorithm.
# 
# **Linear model**:
# 
# $$\mu_i = \beta_0 + \beta_1 a_i$$
# 
# **Sampling distribution**:
# 
# $$p_i \sim N(\mu_i, \tau)$$
# 
# **Prior distributions**:
# 
# $$\begin{aligned}
# & \beta_i \sim N(0, 10000) \cr
# & \tau \sim \text{Gamma}(0.001, 0.001)
# \end{aligned}$$

# In[39]:

age = np.array([13, 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11, 
                15, 11, 7, 13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 
                14, 10, 14, 11, 13, 14, 10])

price = np.array([2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800, 
                  1990, 3500, 5100, 3900, 2900, 4950, 2000, 3400, 8999, 4000, 
                  2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500, 
                  2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700])/1000.


# To avoid numerical underflow issues, we typically work with log-transformed likelihoods, so the joint posterior can be calculated as sums of log-probabilities and log-likelihoods.
# 
# This function calculates the joint log-posterior, conditional on values for each parameter:

# In[40]:

from scipy.stats import distributions
dgamma = distributions.gamma.logpdf
dnorm = distributions.norm.logpdf

def calc_posterior(a, b, t, y=price, x=age):
    # Calculate joint posterior, given values for a, b and t

    # Priors on a,b
    logp = dnorm(a, 0, 10000) + dnorm(b, 0, 10000)
    # Prior on t
    logp += dgamma(t, 0.001, 0.001)
    # Calculate mu
    mu = a + b*x
    # Data likelihood
    logp += sum(dnorm(y, mu, t**-2))
    
    return logp


# The `metropolis` function implements a simple random-walk Metropolis-Hastings sampler for this problem. It accepts as arguments:
# 
# - the number of iterations to run
# - initial values for the unknown parameters
# - the variance parameter of the proposal distribution (normal)

# In[41]:

rnorm = np.random.normal
runif = np.random.rand

def metropolis(n_iterations, initial_values, prop_var=1):

    n_params = len(initial_values)
            
    # Initial proposal standard deviations
    prop_sd = [prop_var]*n_params
    
    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))
    
    # Set initial values
    trace[0] = initial_values
        
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    # Initialize acceptance counts
    accepted = [0]*n_params
    
    for i in range(n_iterations):
    
        if not i%1000: print('Iteration %i' % i)
    
        # Grab current parameter values
        current_params = trace[i]
    
        for j in range(n_params):
    
            # Get current value for parameter j
            p = trace[i].copy()
    
            # Propose new value
            if j==2:
                # Ensure tau is positive
                theta = np.exp(rnorm(np.log(current_params[j]), prop_sd[j]))
            else:
                theta = rnorm(current_params[j], prop_sd[j])
            
            # Insert new value 
            p[j] = theta
    
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*p)
    
            # Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob
    
            # Sample a uniform random variate
            u = runif()
    
            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                accepted[j] += 1
            else:
                # Reject
                trace[i+1,j] = trace[i,j]
                
    return trace, accepted


# Let's run the MH algorithm with a very small proposal variance:

# In[42]:

n_iter = 10000
trace, acc = metropolis(n_iter, initial_values=(1,0,1), prop_var=0.001)


# We can see that the acceptance rate is way too high:

# In[43]:

np.array(acc, float)/n_iter


# In[44]:

trace1 = pgo.Scatter(
    y=trace.T[0],
    xaxis='x1',
    yaxis='y1',
    marker=pgo.Marker(color=color)
)

trace2 = pgo.Histogram(
    x=trace.T[0],
    xaxis='x2',
    yaxis='y2',
    marker=pgo.Marker(color=color)
)

trace3 = pgo.Scatter(
    y=trace.T[1],
    xaxis='x3',
    yaxis='y3',
    marker=pgo.Marker(color=color)
)

trace4 = pgo.Histogram(
    x=trace.T[1],
    xaxis='x4',
    yaxis='y4',
    marker=pgo.Marker(color=color)
)

trace5 = pgo.Scatter(
    y=trace.T[2],
    xaxis='x5',
    yaxis='y5',
    marker=pgo.Marker(color=color)
)

trace6 = pgo.Histogram(
    x=trace.T[2],
    xaxis='x6',
    yaxis='y6',
    marker=pgo.Marker(color=color)
)


# In[45]:

data5 = pgo.Data([trace1, trace2, trace3, trace4, trace5, trace6])


# In[46]:

fig5 = tls.make_subplots(3, 2)


# In[47]:

fig5['data'] += data5


# In[48]:

add_style(fig5)


# In[49]:

fig5['layout'].update(showlegend=False,
                     yaxis1=pgo.YAxis(title='intercept'),
                     yaxis3=pgo.YAxis(title='slope'),
                     yaxis5=pgo.YAxis(title='precision')
)


# In[50]:

py.iplot(fig5, filename='MH algorithm small proposal variance')


# Now, with a very large proposal variance:

# In[51]:

trace_hivar, acc = metropolis(n_iter, initial_values=(1,0,1), prop_var=100)


# In[52]:

np.array(acc, float)/n_iter


# In[53]:

trace1 = pgo.Scatter(
    y=trace_hivar.T[0],
    xaxis='x1',
    yaxis='y1',
    marker=pgo.Marker(color=color)
)

trace2 = pgo.Histogram(
    x=trace_hivar.T[0],
    xaxis='x2',
    yaxis='y2',
    marker=pgo.Marker(color=color)
)

trace3 = pgo.Scatter(
    y=trace_hivar.T[1],
    xaxis='x3',
    yaxis='y3',
    marker=pgo.Marker(color=color)
)

trace4 = pgo.Histogram(
    x=trace_hivar.T[1],
    xaxis='x4',
    yaxis='y4',
    marker=pgo.Marker(color=color)
)

trace5 = pgo.Scatter(
    y=trace_hivar.T[2],
    xaxis='x5',
    yaxis='y5',
    marker=pgo.Marker(color=color)
)

trace6 = pgo.Histogram(
    x=trace_hivar.T[2],
    xaxis='x6',
    yaxis='y6',
    marker=pgo.Marker(color=color)
)


# In[54]:

data6 = pgo.Data([trace1, trace2, trace3, trace4, trace5, trace6])


# In[55]:

fig6 = tls.make_subplots(3, 2)


# In[56]:

fig6['data'] += data6


# In[57]:

add_style(fig6)


# In[58]:

fig6['layout'].update(
    yaxis1=pgo.YAxis(title='intercept'),
    yaxis3=pgo.YAxis(title='slope'),
    yaxis5=pgo.YAxis(title='precision')
)


# In[59]:

py.iplot(fig6, filename='MH algorithm large proposal variance')


# ### Adaptive Metropolis
# 
# In order to avoid having to set the proposal variance by trial-and-error, we can add some tuning logic to the algorithm. The following implementation of Metropolis-Hastings reduces proposal variances  by 10% when the acceptance rate is low, and increases it by 10% when the acceptance rate is high.

# In[60]:

def metropolis_tuned(n_iterations, initial_values, f=calc_posterior, prop_var=1, 
                     tune_for=None, tune_interval=100):
    
    n_params = len(initial_values)
            
    # Initial proposal standard deviations
    prop_sd = [prop_var] * n_params
    
    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))
    
    # Set initial values
    trace[0] = initial_values
    # Initialize acceptance counts
    accepted = [0]*n_params
    
    # Calculate joint posterior for initial values
    current_log_prob = f(*trace[0])
    
    if tune_for is None:
        tune_for = n_iterations/2

    for i in range(n_iterations):
    
        if not i%1000: print('Iteration %i' % i)
    
        # Grab current parameter values
        current_params = trace[i]
    
        for j in range(n_params):
    
            # Get current value for parameter j
            p = trace[i].copy()
    
            # Propose new value
            if j==2:
                # Ensure tau is positive
                theta = np.exp(rnorm(np.log(current_params[j]), prop_sd[j]))
            else:
                theta = rnorm(current_params[j], prop_sd[j])
            
            # Insert new value 
            p[j] = theta
    
            # Calculate log posterior with proposed value
            proposed_log_prob = f(*p)
    
            # Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob
    
            # Sample a uniform random variate
            u = runif()
    
            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                accepted[j] += 1
            else:
                # Reject
                trace[i+1,j] = trace[i,j]
                
            # Tune every 100 iterations
            if (not (i+1) % tune_interval) and (i < tune_for):
        
                # Calculate aceptance rate
                acceptance_rate = (1.*accepted[j])/tune_interval
                if acceptance_rate<0.1:
                    prop_sd[j] *= 0.9
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.95
                if acceptance_rate>0.4:
                    prop_sd[j] *= 1.05
                elif acceptance_rate>0.6:
                    prop_sd[j] *= 1.1
        
                accepted[j] = 0
                
    return trace[tune_for:], accepted


# In[61]:

trace_tuned, acc = metropolis_tuned(n_iter*2, initial_values=(1,0,1), prop_var=5, tune_interval=25, tune_for=n_iter)


# In[62]:

np.array(acc, float)/(n_iter)


# In[63]:

trace1 = pgo.Scatter(
    y=trace_tuned.T[0],
    xaxis='x1',
    yaxis='y1',
    line=pgo.Line(width=1),
    marker=pgo.Marker(color=color)
)

trace2 = pgo.Histogram(
    x=trace_tuned.T[0],
    xaxis='x2',
    yaxis='y2',
    marker=pgo.Marker(color=color)
)

trace3 = pgo.Scatter(
    y=trace_tuned.T[1],
    xaxis='x3',
    yaxis='y3',
    line=pgo.Line(width=1),
    marker=pgo.Marker(color=color)
)

trace4 = pgo.Histogram(
    x=trace_tuned.T[1],
    xaxis='x4',
    yaxis='y4',
    marker=pgo.Marker(color=color)
)

trace5 = pgo.Scatter(
    y=trace_tuned.T[2],
    xaxis='x5',
    yaxis='y5',
    line=pgo.Line(width=0.5),
    marker=pgo.Marker(color=color)
)

trace6 = pgo.Histogram(
    x=trace_tuned.T[2],
    xaxis='x6',
    yaxis='y6',
    marker=pgo.Marker(color=color)
)


# In[64]:

data7 = pgo.Data([trace1, trace2, trace3, trace4, trace5, trace6])


# In[65]:

fig7 = tls.make_subplots(3, 2)


# In[66]:

fig7['data'] += data7


# In[67]:

add_style(fig7)


# In[68]:

fig7['layout'].update(
    yaxis1=pgo.YAxis(title='intercept'),
    yaxis3=pgo.YAxis(title='slope'),
    yaxis5=pgo.YAxis(title='precision')
)


# In[69]:

py.iplot(fig7, filename='adaptive-metropolis')


# 50 random regression lines drawn from the posterior:

# In[70]:

# Data points
points = pgo.Scatter(
    x=age,
    y=price,
    mode='markers'
)

# Sample models from posterior
xvals = np.linspace(age.min(), age.max())
line_data = [np.column_stack([np.ones(50), xvals]).dot(trace_tuned[np.random.randint(0, 1000), :2]) for i in range(50)]

# Generate Scatter obejcts
lines = [pgo.Scatter(x=xvals, y=line, opacity=0.5, marker=pgo.Marker(color='#e34a33'),
                     line=pgo.Line(width=0.5)) for line in line_data]

data8 = pgo.Data([points] + lines)

layout8 = layout_grey_bg.copy()
layout8.update(
    showlegend=False,
    hovermode='closest',
    xaxis=pgo.XAxis(title='Age', showgrid=False, zeroline=False),
    yaxis=pgo.YAxis(title='Price', showline=False, zeroline=False)
)

fig8 = pgo.Figure(data=data8, layout=layout8)
py.iplot(fig8, filename='regression_lines')


# ## Exercise: Bioassay analysis
# 
# Gelman et al. (2003) present an example of an acute toxicity test, commonly performed on animals to estimate the toxicity of various compounds.
# 
# In this dataset `log_dose` includes 4 levels of dosage, on the log scale, each administered to 5 rats during the experiment. The response variable is `death`, the number of positive responses to the dosage.
# 
# The number of deaths can be modeled as a binomial response, with the probability of death being a linear function of dose:
# 
# <div style="font-size: 150%;">  
# $$\begin{aligned}
# y_i &\sim \text{Bin}(n_i, p_i) \\
# \text{logit}(p_i) &= a + b x_i
# \end{aligned}$$
# </div>
# 
# The common statistic of interest in such experiments is the **LD50**, the dosage at which the probability of death is 50%.
# 
# Use Metropolis-Hastings sampling to fit a Bayesian model to analyze this bioassay data, and to estimate LD50.

# In[71]:

# Log dose in each group
log_dose = [-.86, -.3, -.05, .73]

# Sample size in each group
n = 5

# Outcomes
deaths = [0, 1, 3, 5]


# In[72]:

from scipy.stats import distributions
dbin = distributions.binom.logpmf
dnorm = distributions.norm.logpdf

invlogit = lambda x: 1./(1 + np.exp(-x))

def calc_posterior(a, b, y=deaths, x=log_dose):

    # Priors on a,b
    logp = dnorm(a, 0, 10000) + dnorm(b, 0, 10000)
    # Calculate p
    p = invlogit(a + b*np.array(x))
    # Data likelihood
    logp += sum([dbin(yi, n, pi) for yi,pi in zip(y,p)])
    
    return logp


# In[73]:

bioassay_trace, acc = metropolis_tuned(n_iter, f=calc_posterior, initial_values=(1,0), prop_var=5, tune_for=9000)


# In[74]:

trace1 = pgo.Scatter(
    y=bioassay_trace.T[0],
    xaxis='x1',
    yaxis='y1',
    marker=pgo.Marker(color=color)
)

trace2 = pgo.Histogram(
    x=bioassay_trace.T[0],
    xaxis='x2',
    yaxis='y2',
    marker=pgo.Marker(color=color)
)

trace3 = pgo.Scatter(
    y=bioassay_trace.T[1],
    xaxis='x3',
    yaxis='y3',
    marker=pgo.Marker(color=color)
)

trace4 = pgo.Histogram(
    x=bioassay_trace.T[1],
    xaxis='x4',
    yaxis='y4',
    marker=pgo.Marker(color=color)
)


# In[75]:

data9 = pgo.Data([trace1, trace2, trace3, trace4])


# In[76]:

fig9 = tls.make_subplots(2, 2)


# In[77]:

fig9['data'] += data9


# In[78]:

add_style(fig9)


# In[79]:

fig9['layout'].update(
    yaxis1=pgo.YAxis(title='intercept'),
    yaxis3=pgo.YAxis(title='slope')
)


# In[80]:

py.iplot(fig9, filename='bioassay')

