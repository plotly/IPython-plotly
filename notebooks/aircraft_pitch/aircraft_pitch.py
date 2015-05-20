
# coding: utf-8

# <h1>Aircraft Pitch: Frequency Domain Methods for Controller Design</h1>
# 
# Key MATLAB commands used in this tutorial are: <a href ="http://www.mathworks.com/help/control/ref/tf.html" class = 'nounderline'>tf</a> , <a href = "http://www.mathworks.com/help/control/ref/step.html" class = 'nounderline'>step</a> , <a href = "http://www.mathworks.com/help/control/ref/feedback.html" class = 'nounderline'>feedback</a> , <a href = "http://www.mathworks.com/help/control/ref/pole.html" class = 'nounderline'>pole</a> , <a href = "http://www.mathworks.com/help/control/ref/margin.html" class = 'nounderline'>margin</a> , <a href = "http://www.mathworks.com/help/control/ref/stepinfo.html" class = 'nounderline'>stepinfo</a>
# 
# Original content from <a href="http://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlFrequency">University of Michigan </a>
# 
# <h2> Contents </h2>
# 
# - <a href = "#olr" class = 'nounderline'>Open-loop response</a>
# 
# - <a href = "#clr" class = 'nounderline'>Closed-loop response</a>
# 
# - <a href = "#lc" class = 'nounderline'>Lead compensator</a>

# From the main problem, the open-loop transfer function for the aircraft pitch dynamics is
# 

#  <p> <span class="eqn_num">(1)</span>$$ P(s) = \frac{\Theta(s)}{\Delta(s)} = \frac {1.151s+0.1774}{s^3+0.739s^2+0.921s}$$</p>

# where the input is elevator deflection angle $\delta$ and the output is the aircraft pitch angle $\theta$.
# 
# For the original problem setup and the derivation of the above transfer function please refer to the Aircraft Pitch: System Modeling page
# 
# For a step reference of 0.2 radians, the design criteria are the following.
# 
# - Overshoot less than 10%
# 
# - Rise time less than 2 seconds
# 
# - Settling time less than 10 seconds
# 
# - Steady-state error less than 2%
# 
# <h2 id = olr>**Open-loop response**</h2>

# Let's first begin by examining the behavior of the open-loop plant. Specifically, create a new m-file, and enter the following commands. Note the scaling of the step response by 0.2 to account for the fact that the input is a step of 0.2 radians (11 degrees). Running this m-file in the MATLAB command window should give you the step response plot shown below.

# In[22]:

get_ipython().magic(u'load_ext pymatbridge')


# In[24]:

get_ipython().run_cell_magic(u'capture', u'', u"%%matlab \n\nf = figure;\nt = [0:0.01:10];\ns = tf('s');\nP_pitch = (1.151*s + 0.1774)/(s^3 + 0.739*s^2 + 0.921*s);\nstep(0.2*P_pitch,t);\naxis([0 10 0 0.8]);\nylabel('pitch angle (rad)');\ntitle('Open-loop Step Response');\ngrid\n\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);")


# In[55]:

show_plot('https://plot.ly/~UMichiganControl/0/')


# Examination of the above plot indicates that the open-loop system is unstable for a step input, that is, its output grows unbounded when given a step input. This is due to the fact that the transfer function has a pole at the origin.
# 
# <br>

# <h2 id=clr>**Closed-loop response**</h2>
# 
# Let's now close the loop on our plant and see if that stabilizes the system. Consider the following unity feedback architecture for our system.

# <img src="http://ctms.engin.umich.edu/CTMS/Content/AircraftPitch/Control/Frequency/figures/feedback_pitch2.png">

# The following code entered in the MATLAB command window generates the closed-loop transfer function assuming the unity-feedback architecture above and a unity-gain controller, C(s) = 1.

# In[25]:

get_ipython().run_cell_magic(u'matlab', u'', u'sys_cl = feedback(P_pitch,1)')


# Examining the poles of this transfer function using the pole command as shown below, it can be seen that this closed-loop system is indeed stable since all of the poles have negative real part.
# 
# 

# In[33]:

get_ipython().run_cell_magic(u'matlab', u'', u'pole(sys_cl)')


# Stability of this closed-loop system can also be determined using the frequency response of the open-loop system. The margin command generates the Bode plot for the given transfer function with annotations for the gain margin and phase margin of the system when the loop is closed as demonstrated below.

# In[34]:

get_ipython().run_cell_magic(u'capture', u'', u'%%matlab\n\nf = figure; \nmargin(P_pitch)\ngrid\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);')


# In[54]:

show_plot('https://plot.ly/~UMichiganControl/1/')


# Examination of the above demonstrates that the closed-loop system is indeed stable since the phase margin and gain margin are both positive. Specifically, the phase margin equals 46.9 degrees and the gain margin is infinite. It is good that this closed-loop system is stable, but does it meet our requirements? Add the following code to your m-file and re-run and you will generate the step response plot shown below.
# 

# In[13]:

get_ipython().run_cell_magic(u'capture', u'', u"%%matlab \n\nf = figure; \nsys_cl = feedback(P_pitch,1);\nstep(0.2*sys_cl), grid\nylabel('pitch angle (rad)');\ntitle('Closed-loop Step Response')\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);")


# In[53]:

show_plot('https://plot.ly/~UMichiganControl/2/')


# Examination of the above demonstrates that the settle time requirement of 10 seconds is not close to being met. One way to address this is to make the system response faster, but then the overshoot shown above will likely become a problem. Therefore, the overshoot must be reduced in conjunction with making the system response faster. We can accomplish these goals by adding a compensator to reshape the Bode plot of the open-loop system. The Bode plot of the open-loop system indicates behavior of the closed-loop system. More specifically,
# 
# - the gain crossover frequency is directly related to the closed-loop system's speed of response, and
# 
# - the phase margin is inversely related to the closed-loop system's overshoot.
# 
# the gain crossover frequency is directly related to the closed-loop system's speed of response, and
# the phase margin is inversely related to the closed-loop system's overshoot.
# Therefore, we need to add a compensator that will increase the gain crossover frequency and increase the phase margin as indicated in the Bode plot of the open-loop system.
# 
# <br>
# 

# <h2 id = "lc"> **Lead compensator**</h2>
# 
# A type of compensator that can accomplish both of our goals is a lead compensator. Referring to the Lead and Lag Compensators page, a lead compensator adds positive phase to the system. Additional positive phase increases the phase margin, thus, increasing the damping. The lead compensator also generally increases the magnitude of the open-loop frequency response at higher frequencies, thereby, increasing the gain crossover frequency and overall speed of the system. Therefore, the settling time should decrease as a result of the addition of a lead compensator. The general form of the transfer function of a lead compensator is the following.

# <p> <span class="eqn_num">(2)</span> $$ C(s)=K \frac{Ts + 1}{\alpha Ts+1} \ \ \ (\alpha < 1) $$ </p>

# We thus need to find $\alpha$, T and K. Typically, the gain K is set to satisfy requirements on steady-state error. Since our system is already type 1 (the plant has an integrator) the steady-state error for a step input will be zero for any value of K. Even though the steady-state error is zero, the slow tail on the response can be attributed to the fact the velocity-error constant is too small. This deficiency can be addressed by employing a value of K that is greater than 1, in other words, a value of K that will shift the magnitude plot upward. Through some trial and error, we will somewhat arbitrarily choose K = 10. Running the following code in the MATLAB window will demonstrate the effect of adding this K.
# 
# 

# In[42]:

get_ipython().run_cell_magic(u'capture', u'', u"%%matlab \n\nf = figure;\nK = 10;\nmargin(K*P_pitch), grid\nsys_cl = feedback(K*P_pitch,1);\nstep(0.2*sys_cl), grid\ntitle('Closed-loop Step Response with K = 10')\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);")


# In[52]:

show_plot('https://plot.ly/~UMichiganControl/3/')


# In[51]:

show_plot('https://plot.ly/~UMichiganControl/4/')


# From examination of the above Bode plot, we have increased the system's magnitude at all frequencies and have pushed the gain crossover frequency higher. The effect of these changes are evident in the closed-loop step response shown above. Unfortunately, the addition of the K has also reduced the system's phase margin as evidenced by the increased overshoot in the system's step response. As mentioned previously, the lead compensator will help add damping to the system in order to reduce the overshoot in the step response.
# 
# Continuing with the design of our compensator, we will next address the parameter $\alpha$ which is defined as the ratio between the zero and pole. The larger the separation between the zero and the pole the greater the bump in phase where the maximum amount of phase that can be added with a single pole-zero pair is 90 degrees. The following equation captures the maximum phase added by a lead compensator as a function of $\alpha$.

# <p><span class="eqn_num">(3)</span>$$ \sin(\phi_m)=\frac{1 - \alpha}{1 + \alpha} $$ </p>

# Relationships between the time response and frequency response of a standard underdamped second-order system can be derived. One such relationship that is a good approximation for damping ratios less than approximately 0.6 or 0.7 is the following

# <p><span class="eqn_num">(4)</span> $$ \zeta \approx \frac{PM (degrees)}{100^{\circ}} $$</p>

# While our system does not have the form of a standard second-order system, we can use the above relationship as a starting point in our design. As we are required to have overshoot less than 10%, we need our damping ratio $\zeta$ to be approximately larger than 0.59 and thus need a phase margin greater than about 59 degrees. Since our current phase margin (with the addition of K) is approximately 10.4 degrees, an additional 50 degrees of phase bump from the lead compensator should be sufficient. Since it is known that the lead compensator will further increase the magnitude of the frequency response, we will need to add more than 50 degrees of phase lead to account for the fact that the gain crossover frequency will increase to a point where the system has more phase lag. We will somewhat arbitrarily add 5 degrees and aim for a total bump in phase of 50+5 = 55 degrees.
# 
# We can then use this number to solve the above relationship for $\alpha$ as shown below.

# <p><span class="eqn_num">(5)</span>$$ \alpha = \frac{1 - \sin(55^{\circ})}{1 + \sin(55^{\circ})} \approx 0.10 $$ </p>

# From the above, we can calculate that $\alpha$ must be less than approximately 0.10. For this value of $\alpha$, the following relationship can be used to determine the amount of magnitude increase that will be supplied by the lead compensator at the location of the maximum bump in phase.
# 
# 

# <p><span class="eqn_num">(6)</span>$$ 20 \log \left( \frac{1}{\sqrt{\alpha}} \right) \approx 20 \log \left( \frac{1}{\sqrt{0.10}} \right) \approx 10 dB $$</p>

# Examining the Bode plot shown above, the magnitude of the uncompensated system equals -10 dB at approximately 6.1 rad/sec. Therefore, the addition of our lead compensator will move the gain crossover frequency from 3.49 rad/sec to approximately 6.1 rad/sec. Using this information, we can then calculate a value of T from the following in order to center the maximum bump in phase at the new gain crossover frequency in order to maximize the system's resulting phase margin.

# <p><span class="eqn_num">(7)</span> $$ \omega_m = \frac{1}{T \sqrt{\alpha}} \Rightarrow T = \frac{1}{6.1\sqrt{.10}} \approx 0.52 $$ </p>

# With the values K = 10, $\alpha$ = 0.10, and T = 0.52 calculated above, we now have a first attempt at our lead compensator. Adding the following lines to your m-file and running at the command line will generate the plot shown below demonstrating the effect of your lead compensator on the system's frequency response.
# 
# 

# In[61]:

get_ipython().run_cell_magic(u'capture', u'', u'%%matlab \n\nf = figure;\nK = 10;\nalpha = 0.10; \nT = 0.52;\nC_lead = K*(T*s + 1) / (alpha*T*s + 1);\nmargin(C_lead*P_pitch), grid\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);')


# In[50]:

show_plot('https://plot.ly/~UMichiganControl/5/')


# Examination of the above demonstrates that the lead compensator increased the system's phase margin and gain crossover frequency as desired. We now need to look at the actual closed-loop step response in order to determine if we are close to meeting our requirements. Replace the step response code in your m-file with the following and re-run in the MATLAB command window.

# In[64]:

get_ipython().run_cell_magic(u'capture', u'', u"%%matlab \n\nf = figure; \nsys_cl = feedback(C_lead*P_pitch,1);\nstep(0.2*sys_cl), grid\ntitle('Closed-loop Step Response with K = 10, alpha = 0.10, and T = 0.52')\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);")


# In[71]:

show_plot('https://plot.ly/~UMichiganControl/6/')


# Examination of the above demonstrates that we are close to meeting our requirements. Using the MATLAB command stepinfo as shown below we can see precisely the characteristics of the closed-loop step response.
# 
# 

# In[80]:

get_ipython().run_cell_magic(u'matlab', u'', u'\nstepinfo(0.2*sys_cl)')


# From the above, all of our requirements are met except for the overshoot which is a bit larger than the requirement of 10%. Iterating on the above design process, we arrive at the parameters K = 10, $\alpha$ = 0.04, and T = 0.55. The performance achieved with this controller can then be verified by modifying the code in your m-file as follows.
# 
# 

# In[50]:

get_ipython().run_cell_magic(u'capture', u'', u"%%matlab \n\nf = figure; \nK = 10;\nalpha = 0.04;\nT = 0.55;\nC_lead = K*(T*s + 1) / (alpha*T*s + 1);\nsys_cl = feedback(C_lead*P_pitch,1);\nstep(0.2*sys_cl), grid\ntitle('Closed-loop Step Response with K = 10, alpha = 0.04, and T = 0.55')\n\n%%%%%%%%%%%%%%%%%%%\n%     PLOTLY      % \n%%%%%%%%%%%%%%%%%%%\n\nfig2plotly(f);")


# In[66]:

show_plot('https://plot.ly/~UMichiganControl/7/')


# Examination of the above step response demonstrates that the requirements are now met. Using the stepinfo command again more clearly demonstrates that the requirements are met.

# In[72]:

get_ipython().run_cell_magic(u'matlab', u'', u'\nstepinfo(0.2*sys_cl)')


# Therefore, the following lead compensator is able to satisfy all of our design requirements.
# 
# 

# <p> <span class="eqn_num">(8)</span> $$C(s)=10\frac{0.55s + 1 }{ 0.022s+1}$$</p>

# In[5]:

# CSS styling within IPython notebook
from IPython.core.display import HTML
def css_styling():
    styles = open("./css/style_notebook_umich.css", "r").read()
    return HTML(styles)

css_styling()

