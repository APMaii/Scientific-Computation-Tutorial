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


#**********************************
#**********************************
'''
---------------------------------------------
Actually for very quick drawing graphs and plot you can use the plt 

but if you want to go more advanced and more controll you can
go for object orientetd graphs with figure ands fisure and ax
but instead of figure.xlable or anything you can go for set



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



#---also you can go very easy
x = np.linspace(-1.4, 1.4, 30)
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot = top left
plt.plot(x, x)
plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot = top right
plt.plot(x, x**2)
plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd subplot = bottow left
plt.plot(x, x**3)
plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot = bottom right
plt.plot(x, x**4)
plt.show()





#----with size----
plt.subplot2grid((3,3), (0, 0), rowspan=2, colspan=2)
plt.plot(x, x**2)
plt.subplot2grid((3,3), (0, 2))
plt.plot(x, x**3)
plt.subplot2grid((3,3), (1, 2), rowspan=2)
plt.plot(x, x**4)
plt.subplot2grid((3,3), (2, 0), colspan=2)
plt.plot(x, x**5)
plt.show()






#--------------------------
#--------------------------
'''          PLOT       '''
#--------------------------
#--------------------------
#WHEN YOU have the x and y --> for ploting

x=np.array([1,2])
y=np.array([5,10])
plt.plot(x,y)



#even it can create itself ...
#x=[0,1,2,3,4,5,6]
plt.plot(y)


#=================
#---Just points----
plt.plot(x,y,'o')
plt.plot(x,y,'*')
plt.plot(x,y,'x')
plt.plot(x,y,'H')




#=================
#---Point + Lines---
plt.plot(x,y,marker='o')
plt.plot(x,y,marker='*')


#ms-->marker size
plt.plot(x,y,marker='*',ms=30)
plt.plot(x,y,marker='*',ms=50)
plt.plot(x,y,marker='*',ms=2)

#default--> Blue
#around
plt.plot(x,y,marker='o',ms=50,mec='r')

#inside
plt.plot(x,y,marker='o',ms=50,mfc='r')


#change both
plt.plot(x,y,marker='o',ms=50,mec='r',mfc='r')




#=================
#----Just line---
plt.plot(x,y)

#line style-->ls
plt.plot(x,y,ls='-')
plt.plot(x,y,ls='--')
plt.plot(x,y,ls='-.')
plt.plot(x,y,ls=':')


#line color
lt.plot(x,y,color='r')
plt.plot(x,y,c='r')



plt.plot(x,y,linewidth=20)


#---- Shekle marker ha / marker style-----
'''
'o'	Circle	
'*'	Star	
'.'	Point	
','	Pixel	
'x'	X	
'X'	X (filled)	
'+'	Plus	
'P'	Plus (filled)	
's'	Square	
'D'	Diamond	
'd'	Diamond (thin)	
'p'	Pentagon	
'H'	Hexagon	
'h'	Hexagon	
'v'	Triangle Down	
'^'	Triangle Up	
'<'	Triangle Left	
'>'	Triangle Right	
'1'	Tri Down	
'2'	Tri Up	
'3'	Tri Left	
'4'	Tri Right	
'|'	Vline	
'_'	Hline


'''


#----ranh ha / color-------
'''
'r' - Red
'g' - Green
'b' - Blue
'c' - Cyan
'm' - Magenta
'y' - Yellow
'k' - Black
'w' - White
'''

'''
'solid' (default)	'-'	
'dotted'	':'	
'dashed'	'--'	
'dashdot'	'-.'

'''



#all together
plt.plot(x,y,marker='*',ms=20,mfc='r',linewidth=10,c='g') 



#https://matplotlib.org/stable/gallery/color/colormap_reference.html
#https://matplotlib.org/stable/gallery/color/named_colors.html






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





#----also two thing with each other---
plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-")
plt.plot( [0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
plt.axis([-10, 110, -10, 140])
plt.show()




plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-", [0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
plt.plot( [0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
plt.axis([-10, 110, -10, 140])
plt.show()




x = np.linspace(-1.4, 1.4, 30)
plt.plot(x, x, 'g--', x, x**2, 'r:', x, x**3, 'b^')
plt.show()





x = np.linspace(-1.4, 1.4, 30)
line1, line2, line3 = plt.plot(x, x, 'g--', x, x**2, 'r:', x, x**3, 'b^')
line1.set_linewidth(3.0)
line1.set_dash_capstyle("round")
line3.set_alpha(0.2)
plt.show()




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



plt.scatter(x,y)

plt.scatter(x, y, marker='o')

plt.scatter(x1,y1,color='r')

#------------------------------
liste_colors=['r','r','g','g']
#bejaye 'r' list ro bzae
plt.scatter(x1,y1,color=liste_colors)


#or
colour_inten=[10,20,30,40]
plt.scatter(x1,y1,c=colour_inten,cmap='viridis')



colour_inten=[10,20,30,40]
plt.scatter(x1,y1,c=colour_inten,cmap='inferno')
plt.colorbar() 



#--size--
plt.scatter(x1,y1,color='r',s=60)


liste_size=[10,20,30,70]
#bejaye 60
plt.scatter(x1,y1,color='r',s=liste_size)




plt.scatter(x1,y1,color='r',s=liste_size,alpha=0.2)

alpha_list=[0.1,1,0.5,0.6]
plt.scatter(x1,y1,color='r',s=liste_size,alpha=alpha_list)





#so it has color, s and alpha --> they can get the only one option
#or list



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
''' Pie Chart '''
#--------------------------
#--------------------------
lab=['apple','banana','cher','daa']
size=[50,30,10,10]
plt.pie(size,labels=lab)
plt.show()



lab=['apple','banana','cher','daa']
size=np.array([50,30,10,10])
myc=['green','red','blue','black']
plt.pie(size,labels=lab,colors=myc)
plt.show()




lab=['apple','banana','cher','daa']
size=np.array([50,30,10,10])
myc=['green','red','blue','black']
ex=[0,0,0.2,0]
plt.pie(size,labels=lab,colors=myc,explode=ex)
plt.show()








#--------------------------
#--------------------------
'''       HISTOGRAMS      '''
#--------------------------
#--------------------------
#----BAR------
x=['A','B','C','D'] #LIST, ARRAY
y=np.array([3,8,1,10])
plt.bar(x,y)
plt.show()


x=['A','B','C','D'] #LIST, ARRAY
y=np.array([3,8,1,10])
plt.bar(x,y)
plt.title('NEMODARE ZIBAYE MAN',fontdict=title_font)
plt.xlabel('case ha')
plt.ylabel('jamiat')
plt.show()


#COLOUR
x=['A','B','C','D'] #LIST, ARRAY
y=np.array([3,8,1,10])
plt.bar(x,y,color='r')
plt.show()


#andaze
x=['A','B','C','D'] #LIST, ARRAY
y=np.array([3,8,1,10])
plt.bar(x,y,color='r',width=0.1)
plt.show()








#----HISTOGRAM----
data = np.random.randn(1000)
plt.hist(data)



x = np.random.normal(170, 10, 250)
plt.hist(x)





data1 = np.random.randn(400)
data2 = np.random.randn(500) + 3
data3 = np.random.randn(450) + 6
data4a = np.random.randn(200) + 9
data4b = np.random.randn(100) + 10

plt.hist(data1, bins=5, color='g', alpha=0.75, label='bar hist') # default histtype='bar'
plt.hist(data2, color='b', alpha=0.65, histtype='stepfilled', label='stepfilled hist')
plt.hist(data3, color='r', histtype='step', label='step hist')
plt.hist((data4a, data4b), color=('r','m'), alpha=0.55, histtype='barstacked', label=('barstacked a', 'barstacked b'))

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()







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
'''       IMAGE      '''
#--------------------------
#--------------------------
import matplotlib.image as mpimg

img = mpimg.imread('my_square_function.png')
print(img.shape, img.dtype)
plt.imshow(img)
plt.show()




plt.imshow(img)
plt.axis('off')
plt.show()




#--------------------------
#--------------------------
'''       Animation      '''
#--------------------------
#--------------------------
x = np.linspace(-1, 1, 100)
y = np.sin(x**2*25)
data = np.array([x, y])

fig = plt.figure()
line, = plt.plot([], [], "r-") # start with an empty plot
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.plot([-0.5, 0.5], [0, 0], "b-", [0, 0], [-0.5, 0.5], "b-", 0, 0, "ro")
plt.grid(True)
plt.title("Marvelous animation")

# this function will be called at every iteration
def update_line(num, data, line):
    line.set_data(data[..., :num] + np.random.rand(2, num) / 25)  # we only plot the first `num` data points.
    return line,

line_ani = animation.FuncAnimation(fig, update_line, frames=50, fargs=(data, line), interval=100)
plt.close() # call close() to avoid displaying the static plot


#saving animation----
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
line_ani.save('my_wiggly_animation.mp4', writer=writer)


















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
'''
The axes have little marks called "ticks". To be precise, "ticks" are the locations of
the marks (eg. (-1, 0, 1)), "tick lines" are the small lines drawn at those locations,
"tick labels" are the labels drawn next to the tick lines, and "tickers" are objects 
that are capable of deciding where to place ticks. The default tickers typically do
a pretty good job at placing ~5 to 8 ticks at a reasonable distance from one another.

But sometimes you need more control (eg. there are too many tick labels on the logit
graph above). Fortunately, matplotlib gives you full control over ticks. You can
even activate minor ticks.

'''

x = np.linspace(-2, 2, 100)

plt.figure(1, figsize=(15,10))
plt.subplot(131)
plt.plot(x, x**3)
plt.grid(True)
plt.title("Default ticks")

ax = plt.subplot(132)
plt.plot(x, x**3)
ax.xaxis.set_ticks(np.arange(-2, 2, 1))
plt.grid(True)
plt.title("Manual ticks on the x-axis")

ax = plt.subplot(133)
plt.plot(x, x**3)
plt.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.xaxis.set_ticks([-2, 0, 1, 2])
ax.yaxis.set_ticks(np.arange(-5, 5, 1))
ax.yaxis.set_ticklabels(["min", -4, -3, -2, -1, 0, 1, 2, 3, "max"])
plt.title("Manual ticks and tick labels\n(plus minor ticks) on the y-axis")


plt.grid(True)

plt.show()
















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



#or with fonts----
title_font={ 'family': 'serif'   ,
        'color':  'red'  ,
        'size':  20   }

xy_font={ 'family': 'serif'   ,
        'color':  'green'  ,
        'size':  10  }



plt.plot(x,y,marker='*',mec='r',mfc='g',ms=20,ls='-.',linewidth=13,c='y')
plt.title('nemoodare man',fontdict=title_font)
plt.xlabel('salam paeen',fontdict=xy_font)
plt.ylabel('salam chap',fontdict=xy_font)
plt.show()




plt.title('Nemoodaram',fontdict=font_title ,loc='left',pad=40)


#---yscale---
plt.figure(3)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

plt.figure(4)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.05)
plt.title('symlog')
plt.grid(True)

plt.show()




#---grid---
plt.grid()
#more options
plt.grid(color='green',linestyle='--',linewidth=0.5)





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

x = np.linspace(-1.5, 1.5, 30)
px = 0.8
py = px**2
plt.plot(x, x**2, "b-", px, py, "ro")
plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='blue', horizontalalignment="center")
plt.text(px - 0.08, py, "Beautiful point", ha="right", weight="heavy")
plt.text(px, py, "x = %0.2f\ny = %0.2f"%(px, py), rotation=50, color='gray')
plt.show()



#--another--
plt.plot(x, x**2, px, py, "ro")
plt.annotate("Beautiful point", xy=(px, py), xytext=(px-1.3,py+0.5),
                           color="green", weight="heavy", fontsize=14,
                           arrowprops={"facecolor": "lightgreen"})
plt.show()



#----or-----
plt.plot(x, x**2, px, py, "ro")

bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
plt.text(px-0.2, py, "Beautiful point", bbox=bbox_props, ha="right")

bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)

plt.show()




#-----another example
# Add labels to the plot
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)





#----hline and vline-----
from numpy.random import randn

def plot_line(axis, slope, intercept, **kargs):
    xmin, xmax = axis.get_xlim()
    plt.plot([xmin, xmax], [xmin*slope+intercept, xmax*slope+intercept], **kargs)

x = randn(1000)
y = 0.5*x + 5 + randn(1000)*2
plt.axis([-2.5, 2.5, -5, 15])
plt.scatter(x, y, alpha=0.2)
plt.plot(1, 0, "ro")
plt.vlines(1, -5, 0, color="red")
plt.hlines(0, -2.5, 1, color="red")
plot_line(axis=plt.gca(), slope=0.5, intercept=5, color="magenta")
plt.grid(True)
plt.show()








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

rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)
#before using searborn
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')



#----you can set seaborn and use the matplotlib
import seaborn as sns
sns.set()

# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')




#---histograms----
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
for col in 'xy':
  plt.hist(data[col], normed=True, alpha=0.5)




for col in 'xy':
  sns.kdeplot(data[col], shade=True)


#also the pprobability and others
sns.distplot(data['x'])
sns.distplot(data['y'])




#also it can be the pixels and histogram in axises
with sns.axes_style('white'):
  sns.jointplot("x", "y", data, kind='kde')


#if you want real pixel
with sns.axes_style('white'):
  sns.jointplot("x", "y", data, kind='hex')


#or you can import data directly
tips = sns.load_dataset('tips')
iris = sns.load_dataset("iris")



#pair plots------
sns.pairplot(iris, hue='species', size=2.5)




















