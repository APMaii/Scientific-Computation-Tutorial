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

#A Z-test is used when:
#The population standard deviation (σ) is known.
#The sample size is large (n≥30).


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




#------T test------------
#The population standard deviation (σ) is unknown.
#The sample size is small (n<30).

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





























