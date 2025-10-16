'''
**using statistics and also the notes from class

This is for statistics and we can use before Traditional ML
'''


'''

Statistics is the study of data. It helps us analyze and interpret numbers. There are two main types:

1. Descriptive Statistics – Summarizing data (mean, median, mode, etc.).
2. Inferential Statistics – Making predictions from data (hypothesis testing, regression, etc.).

# Install if you haven't
# !pip install numpy pandas scipy

import numpy as np
import pandas as pd
import scipy.stats as stats



'''

#-----------------------------------------
#Basic Descriptive Statistics

mean_value = np.mean(data)
print("Mean:", mean_value)


median_value = np.median(data)
print("Median:", median_value)


mode_value = stats.mode(data)
print("Mode:", mode_value.mode[0])

range_value = max(data) - min(data)
print("Range:", range_value)

variance_value = np.var(data, ddof=1)  # Use ddof=1 for sample variance
print("Variance:", variance_value)

std_dev = np.std(data, ddof=1)
print("Standard Deviation:", std_dev)


import matplotlib.pyplot as plt

# Histogram (shows frequency of data points)
plt.hist(data, bins=5, color='blue', edgecolor='black')
plt.title("Histogram of Data")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()





'''
Probability


Probability tells us how likely an event is to happen. It ranges from 0 (impossible) to 1 (certain).

Basic Probability Concepts
Experiment: Rolling a dice 🎲
Sample Space (S): All possible outcomes {1, 2, 3, 4, 5, 6}
Event (E): Getting an even number {2, 4, 6}
Probability of Event (P(E))

P(E) : Favorable Outcomes / all events


''''

dice_rolls = [random.randint(1, 6) for _ in range(10)]
print("Dice Rolls:", dice_rolls)

outcomes = ["Heads", "Tails"]
tosses = [random.choice(outcomes) for _ in range(10)]
print("Coin Tosses:", tosses)



#-----Unifrom distribution-----
import numpy as np
import matplotlib.pyplot as plt

# Simulating 10,000 rolls of a fair die
rolls = np.random.randint(1, 7, size=10000)

# Plot histogram
plt.hist(rolls, bins=np.arange(0.5, 7.5, 1), edgecolor='black', density=True)
plt.xticks([1, 2, 3, 4, 5, 6])
plt.title("Uniform Distribution - Dice Rolls")
plt.xlabel("Dice Value")
plt.ylabel("Probability")
plt.show()



#----Normal (Gaussian) Distribution----
#Most values cluster around the mean (e.g., people's height, IQ scores).
#The bell curve shape is defined by:

#Mean (μ) → Center of the distribution
#Standard Deviation (σ) → Spread of the data

mu, sigma = 0, 1  # Mean = 0, Standard deviation = 1
normal_data = np.random.normal(mu, sigma, 1000)

# Plot normal distribution
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='g')

# Plot theoretical bell curve
x = np.linspace(-4, 4, 100)
plt.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2), color='red')

plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()



#---Binomial Distribution-------
#Used when there are two possible outcomes (success/failure), like flipping a coin.
#P(k) = ((n k )) p **k  (1-p)**n-k
#n = number of trials
#k = number of successes
#p = probability of success

n, p = 10, 0.5  # 10 trials, 50% success chance
binom_data = np.random.binomial(n, p, 1000)

# Plot binomial distribution
plt.hist(binom_data, bins=10, edgecolor='black', density=True)
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.show()



#--Poisson Distribution----
#Models rare events (e.g., earthquakes, customer arrivals).
#The λ (lambda) parameter represents the average number of occurrences in a given time.
lambda_ = 3  # Average 3 events per time period
poisson_data = np.random.poisson(lambda_, 1000)

# Plot Poisson distribution
plt.hist(poisson_data, bins=15, edgecolor='black', density=True)
plt.title("Poisson Distribution (λ=3)")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.show()





#======================================
#-------Pre Processing the Data-------
#=====================================


#--------------------------------
#------- Outlier Detection---------
#--------------------------------


'''
Outliers are extreme values that can skew results.

Approaches to Detect Outliers:
1️⃣ Z-Score Method
2️⃣ Interquartile Range (IQR) Method
3️⃣ Box Plot
4️⃣ Visual Inspection (Histogram, KDE)

''''


#---Z-Score Method----
#f ∣Z∣>3, the value is an outlier.
#thios is work well for normal data

z_scores = (data - np.mean(data)) / np.std(data)
outliers = data[np.abs(z_scores) > 3]
print("Outliers:", outliers)


#IQR (Interquartile Range) Method
#for skwed data
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data < lower_bound) | (data > upper_bound)]
print("Outliers:", outliers)




#Box Plot Outlier Detection
plt.boxplot(data, vert=False)
plt.title("Box Plot (Outlier Detection)")
plt.show()


#Using Machine Learning for Outlier Detection
#Isolation Forest
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05)  # 5% outliers
outliers = iso.fit_predict(data.reshape(-1,1))
print("Outliers:", data[outliers == -1])


#--dbscan
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2, min_samples=5)
outliers = dbscan.fit_predict(data.reshape(-1,1))
print("Outliers:", data[outliers == -1])



#--------------------------------
#-------Normality Test------------
#--------------------------------

'''
Many statistical methods assume normality (e.g., Z-tests, T-tests, ANOVA, regression). 
If data isn’t normal, we might need non-parametric tests or transformations.


Ways to Check Normality:

Qualitative/Visual Methods
Histogram
Q-Q Plot
Box Plot
KDE (Kernel Density Estimation)



Quantitative/Statistical Tests
Shapiro-Wilk Test
Kolmogorov-Smirnov Test
Anderson-Darling Test
D’Agostino’s K² Test

''''


#----visual----

#If the histogram looks bell-shaped, the data is likely normal.---

import numpy as np
import matplotlib.pyplot as plt

# Generate normal data
data = np.random.normal(50, 10, 1000)

# Plot histogram
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Histogram of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()






# A symmetric box plot suggests normality.
plt.boxplot(data, vert=False)
plt.title("Box Plot")
plt.show()


#Compares quantiles of data with a normal distribution.
# If the points fall on a straight line, the data is normal.
import scipy.stats as stats
import matplotlib.pyplot as plt

# Q-Q plot
stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()





#KDE (Kernel Density Estimation)
#A smoother version of a histogram.
# bell-shaped curve indicates normality.
import seaborn as sns

sns.kdeplot(data, shade=True, color="blue")
plt.title("KDE Plot")
plt.show()







#----Statistical Normality Tests-------


#Shapiro-Wilk Test---------------
#Best for small datasets (n<5000).

from scipy.stats import shapiro

stat, p = shapiro(data)
print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p}")

if p > 0.05:
    print("Data is normally distributed (Fail to reject H₀)")
else:
    print("Data is NOT normally distributed (Reject H₀)")

# If p>0.05, the data is normal.


#***** in nromality test teh hypothesis is we  are nromal




#kolmogorov-Smirnov (K-S) Test---------------
from scipy.stats import kstest

stat, p = kstest(data, 'norm')
print(f"K-S Test: Statistic={stat}, p-value={p}")





#andersn darling---------------
from scipy.stats import anderson

result = anderson(data)
print(f"Anderson-Darling Test Statistic: {result.statistic}")

for i in range(len(result.critical_values)):
    sig_level = result.significance_level[i]
    crit_val = result.critical_values[i]
    if result.statistic < crit_val:
        print(f"Accept normality at {sig_level}% level")
    else:
        print(f"Reject normality at {sig_level}% level")


# D’Agostino’s K² Test-------
#hecks for skewness & kurtosis.
from scipy.stats import normaltest

stat, p = normaltest(data)
print(f"D’Agostino’s K² Test: Statistic={stat}, p-value={p}")

''''
Shapiro-Wilk → Best for small data
K-S Test → General normality check
Anderson-Darling → Multi-level decision
D’Agostino’s K² → Skewness & kurtosis

''''
  



#======================================
#----WHAT CAN WE DO IF TEH DATA IS NOT NORMALL????
#======================================
#If the data is skewed (not normal), we can apply transformations to make it normal.


#---transfrom to Log-------
import numpy as np
data_log = np.log(data)  # Apply log transformation



#-------Square Root Transformation-------
data_sqrt = np.sqrt(data)


#-------Box-Cox Transformation-------
from scipy.stats import boxcox
data_boxcox, lambda_value = boxcox(data + 1)  # Adding 1 to avoid zero issues
print("Lambda:", lambda_value)  # Determines best transformation




#-------Yeo-Johnson Transformation-------
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
data_yeo = pt.fit_transform(data.reshape(-1, 1))







#======================================
#-------inferential Statistics--------
#=====================================


'''
A Z-score (or standard score) tells us how many standard deviations a value is away from the mean.


Z = X -  miu / sigma
X= observed
miu = mean
sigma = standard deviation


Interpreting Z-Scores:
Z = 0 → The value is exactly at the mean.
Z = 1 → The value is 1 standard deviation above the mean.
Z = -1 → The value is 1 standard deviation below the mean.
Z = 2 → The value is 2 standard deviations above the mean.


'''
import scipy.stats as stats

# Given values
X = 80   # Observed value
mu = 70  # Mean
sigma = 10  # Standard deviation

# Calculate Z-score
Z = (X - mu) / sigma
print("Z-Score:", Z)

# Probability of getting a value less than X
probability = stats.norm.cdf(Z)
print("Probability of getting a value less than X:", probability)



#If Z = 1.0, the probability of a value being less than X is 84.1%.
#If Z = -1.0, the probability is 15.9%.





'''
Central Limit Theorem (CLT)

The distribution of sample means approaches a normal distribution regardless of the population distribution if the sample
size is large enough (n ≥ 30)


Key Points:
Even if the population is skewed, the sample mean follows a normal distribution.
The larger the sample size, the closer the distribution is to normal.
The standard deviation of the sample mean is: sigma X | = sigma / radical n
sigma X =  standard error of the mean
σ = population standard deviation
n = sample size






'''


import numpy as np
import matplotlib.pyplot as plt

# Generate a skewed population (Exponential Distribution)
population = np.random.exponential(scale=2, size=10000)

# Take sample means of size 30, 50, 100
sample_means_30 = [np.mean(np.random.choice(population, 30)) for _ in range(1000)]
sample_means_50 = [np.mean(np.random.choice(population, 50)) for _ in range(1000)]
sample_means_100 = [np.mean(np.random.choice(population, 100)) for _ in range(1000)]

# Plot histograms
plt.figure(figsize=(10,6))
plt.hist(sample_means_30, bins=30, alpha=0.5, label="Sample Size 30")
plt.hist(sample_means_50, bins=30, alpha=0.5, label="Sample Size 50")
plt.hist(sample_means_100, bins=30, alpha=0.5, label="Sample Size 100")
plt.legend()
plt.title("Central Limit Theorem - Sample Mean Distributions")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()

#As sample size increases, the distribution of sample means becomes more normal.


#-------------------------------------
#--------Hypothesis Testing-----------

#HYPOTHESIS BEFORE ANYONE--------

#Basic Steps in Hypothesis Testing:

'''

State the null and alternative hypotheses:

Null Hypothesis (H₀): No effect/difference.

Alternative Hypothesis (H₁): There is an effect/difference.

Choose a significance level (α) (commonly 0.05).

Compute the test statistic (Z or T).

Compare p-value with α:
                  If  p<0.05, reject Null
                  If p>0.05, fail to reject Null
​	
'''



#=======================
#--------Z TEST----------
#=======================

#A Z-test is used when:
#The population standard deviation (σ) is known.
#The sample size is large (n≥30).
'''
Assumptions before that:
Normality: The population follows a normal distribution.
Independence: Observations must be independent.
Known Population Variance 
Random Sampling: The sample must be randomly selected.


#if sampel is less than 30 or unknown --> t test

'''



#One-Sample Z-Test

from statsmodels.stats.weightstats import ztest

sample_data = [68, 72, 74, 70, 73, 69, 75, 76, 72, 74]
population_mean = 70

z_stat, p_value = ztest(sample_data, value=population_mean)
print("Z-Statistic:", z_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject the null hypothesis: The sample mean is significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")




# Two-Sample Z-Test
sample1 = np.random.normal(70, 5, 30)
sample2 = np.random.normal(74, 5, 30)

z_stat, p_value = ztest(sample1, sample2)
print("Z-Statistic:", z_stat)
print("P-Value:", p_value)






#=======================
#------T test------------
#=======================

#The population standard deviation (σ) is unknown.
#The sample size is small (n<30).
'''
assumption
Normality: The data should be approximately normal (for small samples).
independence: Observations must be independent.
Unknown Population Variance
Equal Variances (for two-sample t-tests):

'''





#one sample
from scipy.stats import ttest_1samp

sample = [22, 23, 19, 21, 25, 30, 24, 22, 27, 26]
t_stat, p_value = ttest_1samp(sample, 24)
print("T-Statistic:", t_stat)
print("P-Value:", p_value)



from scipy.stats import ttest_ind

group1 = np.random.normal(50, 10, 20)
group2 = np.random.normal(55, 10, 20)

t_stat, p_value = ttest_ind(group1, group2)
print("T-Statistic:", t_stat)
print("P-Value:", p_value)






#----------------- Comparison variances------
#To compare two variances, we use an F-Test
# f = SIGMA1 **2 / SIGMA2 **2

from scipy.stats import f

var1 = np.var(group1, ddof=1)
var2 = np.var(group2, ddof=1)

F_stat = var1 / var2
p_value = 1 - f.cdf(F_stat, len(group1)-1, len(group2)-1)

print("F-Statistic:", F_stat)
print("P-Value:", p_value)

#If p < 0.05, reject  H0(variances are significantly different).



#----------------------------------

'''
Checking Equal Variance (Homogeneity of Variance)

Levene’s Test (Less sensitive to non-normality)
Bartlett’s Test (More powerful, but requires normality)

'''

#Levene’s Test------
from scipy.stats import levene

group1 = np.random.normal(50, 10, 100)  # Mean=50, Std=10, n=100
group2 = np.random.normal(55, 15, 100)  # Mean=55, Std=15, n=100

stat, p_value = levene(group1, group2)
print(f"Levene’s Test: Statistic={stat}, P-Value={p_value}")

#If p>0.05 → Equal variances (assumption holds).
#if p<0.05 → Unequal variances (use Welch’s t-test instead).




#Bartlett’s Test (If Data is Normal)
from scipy.stats import bartlett

stat, p_value = bartlett(group1, group2)
print(f"Bartlett’s Test: Statistic={stat}, P-Value={p_value}")

#If p>0.05 → Equal variances (assumption holds).
#if p<0.05 → Unequal variances (use Welch’s t-test instead).




#Welch’s t-test (does not assume equal variance).-------
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
print(f"Welch’s t-test: T-Statistic={t_stat}, P-Value={p_value}")



#for z or increase the data size or non parametric tests

#----------- if 1 sample Z test not nromal ---> non parametrik
#on-parametric test (Wilcoxon signed-rank test)


#------2 sampel Z test------>non parametric
#non-parametric test (Mann-Whitney U test)



#-----1 t test-------
#Use a non-parametric test (Wilcoxon signed-rank test)


from scipy.stats import wilcoxon

data = [50, 52, 48, 47, 53, 51, 49]  # Example data
stat, p_value = wilcoxon(data - 50)  # Test against population mean 50
print(f"Wilcoxon Test Statistic: {stat}, P-Value: {p_value}")
#If p<0.05, reject H0 (data is significantly different from 50).


#----- 2  t test------
#Use Mann-Whitney U test or Permutation test
from scipy.stats import mannwhitneyu

group1 = [5, 7, 6, 9, 8, 10]
group2 = [11, 15, 14, 13, 16, 12]

stat, p_value = mannwhitneyu(group1, group2)
print(f"Mann-Whitney U Test: Statistic={stat}, P-Value={p_value}")


#so even if non-parametrci is nto enough you can go for boostraping resampling ***



#==================================
#==================================
#==================================
#==================================
'''     CORRELATION TEST      '''
#==================================
#==================================
#==================================
#==================================

'''
When analyzing the relationship between two variables,
we use correlation tests. The most common one is Pearson correlation,
but if the relationship is non-linear, we need alternative methods like Spearman or Kendall tests.
'''

#Linearity: The relationship must be linear (Check using a scatter plot).
# Normality: Both variables should be normally distributed (Check using Shapiro-Wilk test).
#Homoscedasticity: The variance of  Y should be similar across all X values (Check using Levene’s test).
#Independence: Observations must be independent of each other.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
x = np.random.normal(50, 10, 100)  # Normally distributed X
y = 2*x + np.random.normal(0, 5, 100)  # Linear relationship with noise

# 1. Check Linearity
plt.scatter(x, y)
plt.title("Scatter Plot (Check Linearity)")
plt.show()

# 2. Check Normality
print("Shapiro-Wilk test for X:", stats.shapiro(x).pvalue)
print("Shapiro-Wilk test for Y:", stats.shapiro(y).pvalue)

# 3. Check Homoscedasticity
residuals = y - (2*x)
print("Levene’s test for Homoscedasticity:", stats.levene(x, residuals).pvalue)

# 4. Compute Pearson Correlation
r, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation: {r}, p-value: {p_value}")
# If p<0.05, reject  h0→ Correlation is statistically significant.

#----ALSO WE HAVE PEARSON -----


'''
If the data is not linearly related, Pearson’s correlation won’t work. Instead, use Spearman or Kendall correlation.

'''




#---Spearman Correlation (Rank-Based)------
#Monotonicity: The relationship should be increasing or decreasing but not necessarily linear.
# No strict normality assumption (It works for skewed data).
# Suitable for both continuous & ordinal data.


# Generate non-linear data
x = np.random.rand(100) * 10
y = np.sin(x)  # Non-linear relationship

# Compute Spearman correlation
rho, p_value = stats.spearmanr(x, y)
print(f"Spearman correlation: {rho}, p-value: {p_value}")
# If p<0.05, reject  h0→ Correlation is statistically significant.



#---Kendall’s Tau (Non-Parametric)------
# If sample size is small (n<30).
#If data contains many tied ranks.
#When data is ordinal (e.g., survey ratings, Likert scale).

tau, p_value = stats.kendalltau(x, y)
print(f"Kendall’s Tau: {tau}, p-value: {p_value}")
# If p<0.05, reject  h0→ Correlation is statistically significant.


#-------THESE ARE ONLY FOR ONE - ONE---
#-----FOR MULTIPEL AND DRAWING GRAPHS---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Generate random dataset (10 variables, 100 samples)
np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 10), columns=[f'Var{i+1}' for i in range(10)])
# Compute Pearson correlation matrix
corr_matrix = data.corr(method='pearson')
# Print correlation matrix
print(corr_matrix)



plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Heatmap")
plt.show()


#--non linear------
# Compute Spearman correlation matrix
spearman_corr = data.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation Heatmap (Non-Linear)")
plt.show()



#----Pairwise Scatter Plots (Non-Linear Check)----
sns.pairplot(data)
plt.show()




#For detecting non-linear dependencies beyond Spearman/Kendall, use Mutual Information (MI).
from sklearn.feature_selection import mutual_info_regression

# Compute mutual information scores
mi_scores = mutual_info_regression(data.iloc[:, :-1], data.iloc[:, -1])

# Plot MI scores
plt.figure(figsize=(8, 5))
sns.barplot(x=data.columns[:-1], y=mi_scores, palette="viridis")
plt.title("Mutual Information Scores (Non-Linear Dependency)")
plt.xlabel("Features")
plt.ylabel("MI Score")
plt.show()


#Higher MI scores mean stronger non-linear dependency.





#===================================
#===================================
#===================================
'''            ANOVA             '''
#===================================
#===================================
#===================================

'''

ANOVA is a statistical test used to compare the means of three 
or more independent groups to determine if there are significant differences between them.


One-Way ANOVA → Compares means of one independent variable across multiple groups.
Two-Way ANOVA → Compares means across two independent variables (factors).
Repeated Measures ANOVA → Used when the same subjects are tested multiple times.



-----Assumptions of ANOVA-----

 Normality (Residuals follow normal distribution)    / Use Kruskal-Wallis Test (non-parametric ANOVA alternative)
 Homogeneity of Variances (Homoskedasticity) (Equal variance across groups) / Use Welch’s ANOVA (handles unequal variances)
 Independence (Observations are independent)
No Outliers (Outliers can distort results) / remove outliers


  


'''





#-----One-Way ANOVA------
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data for 3 groups
np.random.seed(42)
group1 = np.random.normal(50, 10, 30)
group2 = np.random.normal(55, 10, 30)
group3 = np.random.normal(60, 10, 30)

# Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpretation
if p_value < 0.05:
    print("Reject Null Hypothesis: At least one group is significantly different.")
else:
    print("Fail to Reject Null: No significant difference between groups.")

#if p<0.5 --> reject null --> no difference---> so it means it has effect


#one-way ---> stats.f_oneway()
#Welch ---> stats.ttest_ind(equal_var=False)
#Krusicall ---> stats.kruskal()
#statsmodels.stats.anova_lm()












#---Non-Parametric Alternative (Kruskal-Wallis)-----
stat, p = stats.kruskal(group1, group2, group3)
print(f"Kruskal-Wallis Test: p-value = {p:.3f}")

# If p < 0.05, at least one group is significantly different.



import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Sample Data
np.random.seed(42)
df = pd.DataFrame({
    'Fertilizer': np.repeat(['Organic', 'Chemical'], 15),
    'Sunlight': np.tile(['Low', 'Medium', 'High'], 10),
    'Growth': np.random.normal(20, 5, 30) + 
              (np.repeat([2, 5], 15)) +  # Fertilizer effect
              (np.tile([1, 3, 6], 10))   # Sunlight effect
})

# Display First 5 Rows
print(df.head())



# Fit Two-Way ANOVA Model
model = ols('Growth ~ C(Fertilizer) + C(Sunlight) + C(Fertilizer):C(Sunlight)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Show ANOVA Table
print(anova_table)







#----Friedman Test (Non-Parametric Two-Way ANOVA)----
stat, p = stats.friedmanchisquare(
    df[df['Sunlight'] == 'Low']['Growth'],
    df[df['Sunlight'] == 'Medium']['Growth'],
    df[df['Sunlight'] == 'High']['Growth']
)
print(f"Friedman Test p-value: {p:.3f}")




#---normal but not equal vairanc e--> Welch’s Two-Way ANOVA
stats.ttest_ind(equal_var=False)


#--nomnormal---- fridman test
stats.friedmanchisquare()

















#-------ADD FROM IBM---------
#=========================================================================
#=========================================================================
#=========================================================================
#Estimation


'''
Estimation: application of an algorithm. mean from sample data

X bar = zigam xi/n

inference: underetsand distribution of population (SDTDEV)
zigma (xi-xbar)**2/n-1 ]*1/2
                                             
Machine learning and statsisticalk inference are silmilar.
in both case we using sample data in order to infer qualitites  of actual population.



ML focus on underestanding parameterrs and individual effects involve 
more toolls from statistciak inference


example: Customer churn (rizesh)
feature: length of time, type of purcheste, other charcteristci(age)
churn prediction-->score of individual estimate how mcuh can costumer go

estimation of factor each factor in predicting churnm

Inference--> determining wheter these measured impacts are satistcially significant?


IBM datasets-->
account tyope,customer characteristaate , estimate fo customer lifetime value


'''

sns.barplot(y='churn_value',x='payement',data=df_phone,ci=None)
#churn rate -->y axis  x axis-->features 

#----
#for see ing the some continous value to some categoirial
sns.barplot(y='churn_value',x=pd.cut(df_phone.months,bins=5),data=df_phone,ci=None)


#---
pairplot=df_phone[['month','gb_mon',...]]
sns.pairplot(pairplot,hue='churn_value')

#different linearship betwenn them



#---
#density
sns.joinplot(x=df['months'],y=df['month'],kind='hex')


'''

statsitcial inference-->finding data generating pattern
model-->possible distribution even regression

parametric model-->statistcial model -->constrainr finit nmumebr of
parametrr and made about distribution

non-parametric mdoel--> estimations is not rely on mnay assumptions, 
distribtuion-free inference



#start form non-parametric inference --> creat dsitribuituion  of dtaa using histogram
#CFD

parametric--> set of distributions o r regressions , but they have 
a finit nuymebr pof parmaeters .having some assumptions
particuar-->normal distribution
there is equation-->depend on set parmaetrs (mean, standard devaiation)

assumption we used here -->normal



back to customer life value--->estimate of costumer value of company

the data contian: expected length of time , expected amount spend over time
to estimate lifetimes value-->we mnake assumption for instance
we said that lienar increae or decrease
or nan parametric --> no assumption


In parmateric modeling -->Maximum likelihood etsitmation (MLE).
is related to probability and function of parametr of the model.

L(teta)-->what is likelihood of eman sn stdev
so we chosoe value of teta(parameter) maximize the likelihood function



common distribution------------------------------------
UNIFROM-->equal chance any value in our range

Gussian/Normal-->around mean there is 3sigma spread
lowe stdev is meaningh the mean

Cental limit theory: 
    if you tak averga evalue from bucnh of smaple value,
    distribution of those averages is normal curve if you have enough value
    
Log nromal: log of varibale -->you have normal
take skwerd and tehn with log transform-->normal
**smaller stdev here -->more like mmean


Exponential : landa -->when you and someone time that you watched


Poisson: number of even happend during certain amount of time
we have landa and varianc value
how many people watch this vdieo in the next 10 minutes




Frequentitistic , bayesian----------
Frequentistis repeaded observation in the limit. 
bussines example
how work regard with querying. stdy of working line and how many
match the size of that.
number of sevrs or respond to all web responses.

estimate probability in the limit like bayesian.
think about poisson distribution.



procees has true frequencies in true. but we wnt tomodel on many repeats.
so we mst rely on poission and infrense

Frequente approach: 
    1-fix value of porbabilit. drive porbabilities property of procedure directly
    from data with no external influence. how likely our size->more data we have
    we are more confident
    2-apply derived vale to observed data
    
bayesian-->
parametrs themsev cn be probability distribution. more data ewe have
tighter aroutn aprameter estiamte.
before seing->prior distribution(based on experiemneterrs belief) is formulated.
prior distribution is then updated after seingd ata. posterior dirstribution 


we use same math.same formulas in both frequentis and bayesian.
the element that differes is interpretation.




'''

#==============================================
#==============================================
#==============================================
#==============================================

'''
before only we infrence but here we go for some comaprison


hypothesis test-->statement baout population parameter.

we creat two hypothesis
-th e null hyprthssi
-altrnative hypothesis.



given a data:
    - we can accept H0 or reject H0
    
**you can rject H0 but nevev accept H1-->doesnt matter

in bayesian interpretation: we dont get a decision boudnary
instead we get updated probabilities


EXAMPLE:
    -coin 1 70% coming up
    -coin 2 has 50% coming up

pick one coin without looking
toss coin 10 times and record number.
which one did you toss more?


we can have table to counting that.
we can calculate ratio


suppose we saw 3 heads --

probility of sseing coin 1 is 0.117/ coin2  is 0.009

so coin 1 was 13 rtimes more likely to guve us 3 heads in coin 2
this is likelihood and is something like null hypothesis .



another example--> P(h1=1/2) p(h2)=1/2
so we dont have see data
we can imagine we puloing coin on public:Z


priors P(h1)=1/2=p(h2)=1/2
UPDATING OPRIORIS AFTER SEEN THE DTA 3 HEAD ( BAYES RULES):
p(H1/X)= P(X|h1)P(H1)/P(x)




P(H1|x)/ P(H2|x) = p(H1)*p(x|H1)/ p(h2)*p(x|h2)

this is the ratio we saw

this is how prior go to come.


prior are multilplide by likelihood ratio. which idoesn ot dpeend on the priors


likelihood aratio tells us how we should updat


........
TYP1 TYP2 Error


neyman-pearson paradigm is non-bayesian

Terminolegy

true H0 accept h0 --> CORRECT
TRUTH h1 reject H0 -->correct


Truth H0-->reject H0 -_>Type 1 error
truth H`1 -->accept h0--> type 2

power of test= 1 - P(type || error)
....
The likelihood ratio is called a test statistic : we use it to decide whete
 to accept/reject h0
 
test statistic-->calculated from data

reejection region-->set of value that lead to reject H0

acceptance -->acceptance of h0

null distribution: test statict distribution the null is true.

.....
Significandce level and P-Values-->
we know distribution of null hypothesis.
to get rejection region-->we have to caluclate test staistic.

we will choose before testing that -->we must choose
significance level->ow it is improtant to avoid type 1 and 2e errpr

lower alpha--> extremely low


alpha--> 0.01 or 0.05

0.01 for medication


P-VALUE:smalelst  probability of null hypothesis would be rejcted
confident interval-->value of statistic we accept the null
.....
F-statsitic
H0-->data can be modelet by setting all betas to zero
adding featur ecna not help us-->imagine


reject null if the p-value is small enough
...

Power and samplesize-->

5% significna tets looking for isgnifant resul--> the chances of makinga t 
least one type 1 error increase

probability of at least one type1 error is approximately = 1- (1-0.05)**test


fewer 10 --> 0.05 x (test)
if you get 10 test--> you approximately 50% chance at least one type 1 error accicdentely
the null that it was true.



we use bonferroni correction-->choose p threhsould--> just prethsould
accorind to test numbe

p treshould = 0.05/test


boneforruni correction allows the porbability of a Type 1 error to be controlled bu at 
 cost of power.
 
 effects need to be larger.
 
 
'''
from scipy import stats
import math

#conduct 100 coint and how can i say 75% of that

#null--> i am not special. true raye is predition is 0.5

#n=100 p is 0.5
from scipy.stats import binom
prob=1-binom.cdf(56,100,0.5)

print(str(round(prob*100,1)) + '%')

#9.7 %


#=============================
print(binom.ppf(0.95,100,0.5)+1)


alpha=0.05
t_val2, p_value2 = stats.ttest_ind(smoker_char, nonsmoker_char)
p_value_onetail=p_value2/2
print("t_value = {} , p_value ={} , p_value_onetail = {}".format(t_val2, p_value2, p_value_onetail))















