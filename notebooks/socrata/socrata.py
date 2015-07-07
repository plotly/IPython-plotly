
# coding: utf-8

# # Plotly and Socrata

# ## Awesome datasets and graphs coming together

# Taken from both companies' Wikipedia pages:
# 
# > Plotly is an online analytics and data visualization tool. Plotly provides online graphing, analytics, a Python command line, and stats tools for individuals and collaboration, as well as scientific graphing libraries for Python, R, MATLAB, Perl, Julia, Arduino, and REST.
# 
# > Socrata is a company that provides social data discovery services for opening government data. Socrata targets non-technical Internet users who want to view and share government, healthcare, energy, education, or environment data. Its products are issued under a proprietary, closed, exclusive license.
# 
# Simply put, the two are meant to work together and this IPython notebook will you how you can turn a dataset like <a href="https://opendata.socrata.com/Fun/Top-1-000-Songs-To-Hear-Before-You-Die/ed74-c6ni" target="_blank">this one</a> and into a plot like <a href="https://plot.ly/~etpinard/270/number-of-songs-listed-in-the-guardians-top-1000-songs-to-hear-before-you-die-pe/" target="_blank">that one</a>.

# ### 1. Get a Socrata application token

# You need an application token to communicate with Socrata from a Socrata Open Data API (soda for short).
# 
# Register to Socrata and get your application token <a href="http://dev.socrata.com/register" target="_blank">here</a>.

# ### 2. Install the Soda Ruby wrapper

# Unfortunately, there are no Soda Python wrapper available at this moment in time. But, fortunately, IPython allows us to use mutliple programming language inside the same environment (called an IPython notebook). So, here we will use Ruby and the `soda-ruby` <a href="http://socrata.github.io/soda-ruby/" target="_blank">gem</a> to comminicate with Socrata.
# 
# With Ruby and gem installed on your machine, run in a terminal/command prompt:
# 
# * `$ gem install soda-ruby`
# 
# Add `sudo` in front of the above for a system-wide install on Unix-like machines. Information about local gem install can be found <a href="http://stackoverflow.com/questions/220176/how-can-i-install-a-local-gem" target="_blank">here</a>.
# 
# Then, add the line:
# 
#     gem 'soda-ruby', :require => 'soda'
# 
# to a file named `Gemfile` placed either in the current directory or in folder part of the gems path found of your machine (more <a href="http://gilesbowkett.blogspot.ca/2009/06/find-your-ruby-gems-path.html" target="_blank">here</a>).

# ### 3. Get dataset from Socrata with Ruby and transfer it to IPython

# Head to <a href="https://opendata.socrata.com/" target="_blank">opendata.socrata.com</a>, browse or search for a dataset that you like and click on its link. I chose a list of the Guardian's "Top 1,000 Songs to Hear Before You Die" which can be viewed <a href="https://opendata.socrata.com/Fun/Top-1-000-Songs-To-Hear-Before-You-Die/ed74-c6ni" target="_blank">here</a>. Here is a screenshot of the web page in question:
# 
# <img src="http://i.imgur.com/l1U7Ytn.png" style="padding-top:1em;padding-bottom:1em;">
# 
# Then, 
# 
# 1. Click on `Export`, a blue button on the upper right side of the page.
# 
# 2. Click on `Soda API`, the upper-most tab under `Export`.
# 
# 3. Copy the `API Access Endpoint`, under the `Soda API` tab.
# 
# In our case the API Access Endpoint is:
# 
#     http://opendata.socrata.com/resource/ed74-c6ni.json
#     
# The API Access Endpoint represent the link between the dataset hosted on Socrata and the API, in our case soda-ruby. It contains two pieces of important information: the domain name and the dataset identifier. From the Socrata <a href="http://dev.socrata.com/docs/endpoints.html" target="_blank">offical docs</a>, take note that the API Access Endpoint corresponds to:
# 
#     http://$domain/resource/$dataset_identifier
# 
# So, in our case the domain name is `opendata.socrata.com` and the dataset identifier is `ed74-c6ni`. Note that `.json` is just the file extension, not needed to access the dataset).
# 
# Now, call the `%%ruby` IPython inline magic to turn on Ruby inside the cell below:

# In[1]:

get_ipython().run_cell_magic(u'ruby', u'--out socrata_data', u'\n# with --out, data written to the stdout in this ruby cell \n# will be mapped to a Python variable (socrata_data) after execution.\n\nrequire \'soda/client\'\nrequire \'json\'      \n\n# Set up client object with domain and application token\nclient = SODA::Client.new({:domain => "opendata.socrata.com", \n                           :app_token => "eqZC5q2iEmFXdIu2qEbtZkWgP"})\n\n# Get data with dataset identifier\nresponse = client.get("ed74-c6ni")\n\n# Print dataset to stdout as a JSON \nputs response.to_json')


# And there you go, the Socrata dataset in now inside our IPython namespace!
# 
# Next, we will handle the dataset inside IPython using the popular `pandas` module, so

# In[2]:

import pandas as pd

# Read the retrieved JSON dataset (df stands for dataframe)
df = pd.read_json(socrata_data)


# In[3]:

df.head()  # print the first 5 lines of the dataframe


# In[4]:

df.shape  # print the dataframe's size


# ### 4. Get relevent data and plot it using Plotly!

# ##### Get relevent data

# Let's make a Plotly bar chart with the following features:
# 
# * Artists (on x-axis) vs number of songs in the *1,000 Songs* list (on y-axis),
# * Plot only artists with 4+ or more songs in the *1,000 Songs* list,
# * Plot artists in descending order starting from the artist with the most songs in *1,000 Songs* list.

# So, let's first make a dictionary pairing the artist's name to the their number of tracks in the *1,000 songs* list:

# In[5]:

song_by_artist = df.groupby('artist').size().to_dict()

song_by_artist


# Now, loop through that dictionary and select the key-value pairs corresponding to the artists with 4 or more songs in the *1,000 songs* list:

# In[6]:

song_by_artist_4plus = {k:v for k,v in song_by_artist.items() if v>=4}

song_by_artist_4plus


# Next, as Python dictionaries cannot be sorted, make separate lists of keys and values from the `song_by_artist_4plus` dictionary and sort them in descending order:

# In[7]:

import numpy as np  

# Lists of keys and values
my_keys = song_by_artist_4plus.keys()
my_vals = song_by_artist_4plus.values()

# Find indices of sorted values (first converted to a numpy array)
i_sorted = np.argsort(np.array(my_vals))[::-1]

# Sort both the keys and value list 
my_keys_sorted = [my_keys[i] for i in i_sorted]
my_vals_sorted = [my_vals[i] for i in i_sorted]


# ##### Plot it using Plotly!

# If have a plotly account as well as a credentials file set up on your machine, singing in to Plotly's servers is done automatically while importing `plotly.plotly`:

# In[8]:

import plotly.plotly as py  


# For more info on how to sign up or sign in to Plotly, see <a href="http://nbviewer.ipython.org/github/plotly/python-user-guide/blob/master/s00_homepage/s00_homepage.ipynb#Installation-guidelines" target="_blank">Plotly's Python API User Guide</a>.
# 
# Next, import a few graph objects needed to make our Plotly plot:

# In[9]:

from plotly.graph_objs import Figure, Data, Layout
from plotly.graph_objs import Bar
from plotly.graph_objs import XAxis, YAxis, Marker, Font, Margin


# Make an instance of the bar and data object:

# In[10]:

my_bar = Bar(x=my_keys_sorted,  # labels of the x-axis
             y=my_vals_sorted,  # values of the y-axis
             marker= Marker(color='#2ca02c'))  # a nice green color

my_data = Data([my_bar])  # make data object, (Data accepts only list)


# Make an instance of the layout object:

# In[11]:

my_title = 'Number of songs listed in the Guardian\'s<br><em>Top 1,000 Songs to Hear Before You Die</em> per artist with 4 or more songs'
my_ytitle = 'Number of songs per artist'

my_layout = Layout(title=my_title,   # set plot title
                   showlegend=False, # remove legend 
                   font= Font(family='Georgia, serif', # set global font family
                              color='#635F5D'),        #   and color 
                   plot_bgcolor='#EFECEA',   # set plot color to grey
                   xaxis= XAxis(title='',             # no x-axis title
                                tickangle=45,         # tick labels' angle
                                ticks='outside',      # draw ticks outside axes 
                                ticklen=8,            # tick length
                                tickwidth=1.5,),       #   and width, 
                   yaxis= YAxis(title=my_ytitle,      # y-axis title
                                gridcolor='#FFFFFF',  # white grid lines
                                ticks='outside',     
                                ticklen=8,           
                                tickwidth=1.5),
                   autosize=False,  # manual figure size
                   width=700,       
                   height=500,
                   margin= Margin(b=140)  # increase bottom margin, 
                  )                       # to fit long x-axis tick labels


# Make instance of the figure object, send it to Plotly and get a plot in return inside this IPython notebook:

# In[12]:

my_fig = Figure(data=my_data, layout=my_layout)

py.iplot(my_fig, filename='socrata1')


# Not bad, but let's try to improve our plot by making use of Plotly's hover capabilities.
# 
# Next, we add hover text to each of bars so that hovering with cursor over them will show a list of the songs' titles and years of release included in the *1,000 songs* list in chronological order.
# 
# First, we need to trim the original dataframe so that it contains only the artists with 4 or more songs in the *1,000 songs* list:

# In[13]:

# Rows which have 'artist' name in song_by_artist_4plus
i_good = (df['artist'].isin(song_by_artist_4plus))  

df_good = df[i_good]  # a new dataframe


# In[14]:

df_good.shape   # a much smaller dataframe than the original


# Next, loop through the sorted artists names building a text list to be linked the the `'text'` key in the data object.
# 
# Unfortunately, the biggest lists will have to be truncated to fit inside the Plotly figure:

# In[16]:

my_text = []  # init. the hover-text list

# Loop through the sorted artist names, so that my_text
# will have to same ordering as the values linked to 'x' and 'y' in my_data
for k in my_keys_sorted:
    
    # Slice dataframe to artist name and sort songs by year
    i_artist = (df['artist']==k)
    df_tmp = df_good[i_artist].sort(columns='year')
    
    my_text_tmp = ''  # init. string 
    cnt_song = 0                   # song counter for given artist
    N_song = len(df_tmp['title'])  # total number of song for given artist
    
    # Loop through songs
    for i_song, song in df_tmp.iterrows():
        
        # Add to string and counter
        my_text_tmp += song['title']+' ('+str(song['year'])+')<br>'
        cnt_song += 1
        
        # Skip if song list is too long to fit on figure
        if cnt_song>12:
            diff = N_song - cnt_song
            my_text_tmp += ' and '+str(diff)+' more ...'
            break
    
    # Append hover-text list
    my_text += [my_text_tmp]
    
# Update figure object 
my_fig['data'][0].update(text=my_text)


# Finally, add a text annotation citing our data source to our plot:

# In[17]:

from plotly.graph_objs import Annotation

my_anno_text = '<em>Open Data by Socrata</em><br>Hover over the bars to see list of songs'

my_anno = Annotation(text=my_anno_text,  # annotation text
                     x=0.95,  # position's x-coord
                     y=0.95,  #   and y-coord
                     xref='paper',  # use paper coords
                     yref='paper',  #  for both coordinates
                     font= Font(size=14),  # increase font size (default is 12)
                     showarrow=False,      # remove arrow 
                     bgcolor='#FFFFFF',      # white background
                     borderpad=4)       # space bt. border and text (in px)

# Update figure object
my_fig['layout'].update(annotations=[my_anno])


# And now all we have left to do is to send the updated figure object to plotly:

# In[18]:

py.iplot(my_fig, filename='socrata1-hover')


# Spend some time hovering over the bars and admire plotly's interactibility!
# 
# 
# *Great data and beautiful visualization, at your finger tips.*
# 
# <br>
# 
# <hr>

# <div style="float:right; \">
#     <img src="http://i.imgur.com/4vwuxdJ.png" 
#  align=right style="float:right; margin-left: 5px; margin-top: -10px" />
# </div>
# 
# <h4 style="margin-top:80px;">Got Questions or Feedback? </h4>
# 
# About <a href="https://plot.ly" target="_blank">Plotly</a>
# 
# * email: feedback@plot.ly 
# * tweet: 
# <a href="https://twitter.com/plotlygraphs" target="_blank">@plotlygraphs</a>
# 
# <h4 style="margin-top:30px;">Notebook styling ideas</h4>
# 
# Big thanks to
# 
# * <a href="http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Prologue/Prologue.ipynb" target="_blank">Cam Davidson-Pilon</a>
# * <a href="http://lorenabarba.com/blog/announcing-aeropython/#.U1ULXdX1LJ4.google_plusone_share" target="_blank">Lorena A. Barba</a>
# 
# <br>
