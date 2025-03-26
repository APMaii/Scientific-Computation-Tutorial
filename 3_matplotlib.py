#from teh book and my tutorials

https://github.com/ageron/handson-ml2/blob/master/tools_matplotlib.ipynb




"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                     MATPLOTLIB

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


#----------------------
'''
For Creation of plots
'''
#----------------------

import matplotlib as mpl
import matplotlib.pyplot as plt


#---adjust the style or change that as you need that
plt.style.use('classic')
plt.style.use('seaborn-whitegrid') #this is for scatter


#for showing i ipython consule or in command open new window --> plt.show()
# ------- file: myplot.py ------
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()



#in notebook or anything we musyt first use this before imporintg
#%matplotlib
#to force an update, use plt.draw(). Using plt.show() in Matplotlib mode is not required.






#------first you must build a figure and save that plt.figure() in one fig and then 
#Matplotlib automatically creates a figure in the background if you don't explicitly create one.
#This is why plt.plot() works without calling plt.figure().

'''
When you need to create multiple figures.
When you want to set the figure size (figsize).
When you want to manually manage subplots.

'''

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');
#then plt.show()


#so first you must plt.figure() in fig then you can do anything that you want like saving
fig.savefig('my_figure.png')




#for displaying one image
from IPython.display import Image
Image('my_figure.png')

#or with from PIL import image
#then from image we have plt.imshow



'''
fig.canvas.get_supported_filetypes()
Out[8]: {'eps': 'Encapsulated Postscript',
'jpeg': 'Joint Photographic Experts Group',
'jpg': 'Joint Photographic Experts Group',
'pdf': 'Portable Document Format',
'pgf': 'PGF code for LaTeX',
'png': 'Portable Network Graphics',
'ps': 'Postscript',
'raw': 'Raw RGBA bitmap',
'rgba': 'Raw RGBA bitmap',
'svg': 'Scalable Vector Graphics',
'svgz': 'Scalable Vector Graphics',
'tif': 'Tagged Image File Format',
'tiff': 'Tagged Image File Format'}
'''






#---ALSO YOIU can build multiple subplots
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 6))  # Store the figure object
ax = figure.add_subplot(1, 1, 1)  # Create a subplot inside the figure
ax.plot([1, 2, 3], [4, 5, 6])  # Use ax instead of plt
plt.show()



#---what is axis ?
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [10, 20, 30])
plt.axis([0, 5, 5, 35])  # [xmin, xmax, ymin, ymax]
plt.show()





#----or
plt.axis('off')  # Hides the axis completely
plt.axis('equal')  # Makes x and y units the same size
plt.axis('tight')  # Removes extra white space around the plot


#---so finally we can have figure and each subplto is axis---->
import matplotlib.pyplot as plt

# Create a figure with a specific size
figure = plt.figure(figsize=(8, 6))

# Create a subplot inside the figure
ax = figure.add_subplot(1, 1, 1)  # 1 row, 1 column, 1st plot
ax.plot([1, 2, 3], [10, 20, 30], label="Line Plot")

# Adjust axis
plt.axis([0, 4, 5, 35])  # Set x and y range
plt.legend()  # Show legend
plt.show()









#---------------------------------------------------------
#---------------------------------------------------------
#--------Two Interfaces for the Price of One-------------
#---------------------------------------------------------
#---------------------------------------------------------
#---MATLAB-style interface
plt.figure() # create a plot figure
# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))





#or......
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))




#or for each one
import matplotlib.pyplot as plt

# Create a figure with a specific size
figure = plt.figure(figsize=(8, 6))
# Create a subplot inside the figure
ax = figure.add_subplot(1, 1, 1)  # 1 row, 1 column, 1st plot
ax.plot([1, 2, 3], [10, 20, 30], label="Line Plot")
# Adjust axis
plt.axis([0, 4, 5, 35])  # Set x and y range
plt.legend()  # Show legend
plt.show()




fig = plt.figure()
# Add subplots
ax1 = fig.add_subplot(2, 1, 1)  # (rows, columns, index)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot([1, 2, 3], [4, 5, 6])
ax2.plot([1, 2, 3], [6, 5, 4])
plt.show()



'''
In Matplotlib, the figure (an instance of the class plt.Figure) can be thought of as a
single container that contains all the objects representing axes, graphics, text, and
labels. The axes (an instance of the class plt.Axes) is what we see above: a bounding
box with ticks and labels, which will eventually contain the plot elements that make
up our visualization. Throughout this book, we’ll commonly use the variable name
fig to refer to a figure instance, and ax to refer to an axes instance or group of axes
instances.
'''





#--------------------------
#--------------------------
'''          PLOT       '''
#--------------------------
#--------------------------
#WHEN YOU have the x and y --> for ploting






plt.plot(x, np.sin(x - 0), color='blue') # specify color by name
plt.plot(x, np.sin(x - 1), color='g') # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75') # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44') # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse') # all HTML color names supported




plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')
# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-') # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':') # dotted




#------easy-------
plt.plot(x, x + 0, '-g') # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r') # dotted red




plt.plot(x, y, '-p', color='gray',
markersize=15, linewidth=4,
markerfacecolor='white',
markeredgecolor='gray',
markeredgewidth=2)
plt.ylim(-1.2, 1.2)



#--------------------------
#--------------------------
'''        SCATTER      '''
#--------------------------
#--------------------------
'''
plt.plot can be
noticeably more efficient than plt.scatter. The reason is that plt.scatter has the
capability to render a different size and/or color for each point, so the renderer must
do the extra work of constructing each point individually. In plt.plot, on the other
hand, the points are always essentially clones of each other, so the work of determining
the appearance of the points is done only once for the entire set of data. For large
datasets, the difference between these two can lead to vastly different performance,
and for this reason, plt.plot should be preferred over plt.scatter for large
datasets.
'''

x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')

#instead you can use this --->


plt.scatter(x, y, marker='o')



rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
cmap='viridis')
plt.colorbar() # show color scale



#--------------
plt.errorbar(x, y, yerr=dy, fmt='.k')

#also another things is this
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
color='gray', alpha=0.2)





#--------------------------
#--------------------------
''' Density and Contour Plots  '''
#--------------------------
#--------------------------
#imagien we have one function
def f(x, y):
  return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
'''
grid of x values, a grid of y values, and a grid of z values. The x and y values
represent positions on the plot, and the z values will be represented by the contour
levels.

'''

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, colors='black')

#or with colour
plt.contour(X, Y, Z, 20, cmap='RdGy')

#for knowing the colourmap
#plt.cm.<TAB>

#also for seeing the colorbar
#for showing when we hvae the cmap
plt.colorbar()


#laos witrh imageshow for whoing the label
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')
#it only need the extents, also it can show the figures




#---also you can use the both combination
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
cmap='RdGy', alpha=0.5)
plt.colorbar()
plt.clim(-1, 1)





#--------------------------
#--------------------------
'''       HISTOGRAMS      '''
#--------------------------
#--------------------------

data = np.random.randn(1000)
plt.hist(data)



#IT HAS A LOT OF OPTIONS
plt.hist(data, bins=30, normed=True, alpha=0.5,
histtype='stepfilled', color='steelblue',
edgecolor='none');



#---also you can use this for all of them----
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);






#Two-Dimensional Histograms and Binnings------
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T




plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')


#also something like hexagonal
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')










#--------------------------
#--------------------------
'''   PIE CHART         '''
#--------------------------
#--------------------------



#--------------------------
#--------------------------
'''            '''
#--------------------------
#--------------------------






#--------------------------
#--------------------------
'''    Three-Dimensional Plotting        '''
#--------------------------
#--------------------------
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
#An empty three-dimensional axes


#first we must have ax and then we must have the ax.plot or ax.scatter
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')


# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');




#hree-Dimensional Contour Plots-------
def f(x, y):
return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');



#change the camera---
ax.view_init(60, 35)







#--------------------------
#--------------------------
'''Wireframes and Surface Plots'''
#--------------------------
#--------------------------
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe');



ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
cmap='viridis', edgecolor='none')
ax.set_title('surface');



ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
cmap='viridis', edgecolor='none');








#--------------------------
#--------------------------
'''     Thicks       '''
#--------------------------
#--------------------------













#-------------------------------------------------------
#----------------------FOR ALL PLOTS--------------------
#-------------------------------------------------------
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);



#[xmin, xmax, ymin,ymax]
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);

plt.plot(x, np.sin(x))
plt.axis('tight');

plt.plot(x, np.sin(x))
plt.axis('equal');

#or hide
plt.axis('off')



#---title----
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")



#legen and labeling
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend();

#ax or plt
ax.legend(loc='upper left', frameon=False)
ax.legend(frameon=False, loc='lower center', ncol=2)
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
#also labelspacing=1, title='City Area'









'''
for changing from MATLAB to object oriented
• plt.xlabel() → ax.set_xlabel()
• plt.ylabel() → ax.set_ylabel()
• plt.xlim() → ax.set_xlim()
• plt.ylim() → ax.set_ylim()
• plt.title() → ax.set_title()

'''



ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
xlabel='x', ylabel='sin(x)',
title='A Simple Plot');


#---some times we used cmap
plt.colorbar()
plt.clim(-1, 1)



plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)




#-----two plots inside each other-------
ax1 = plt.axes() # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])




#--------------------------------
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));





#---------------------------
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])






#----------------------------------
#-------Text and Annotation---------
# Add labels to the plot
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)





#-----arrows-----
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
arrowprops=dict(arrowstyle="->",
connectionstyle="angle3,angleA=0,angleB=-90"))





#================================
#Changing the Defaults: rcParams
#================================


IPython_default = plt.rcParams.copy()

from matplotlib import cycler
colors = cycler('color',
['#EE6666', '#3388BB', '#9988DD',
'#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)





"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                     SEA BORN

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""














