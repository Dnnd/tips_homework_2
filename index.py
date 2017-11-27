# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
from  scipy.stats import ks_2samp
from scipy.stats import kstwobign
 
def theor_cdf(x, rel_scale):
    return [(1 - np.exp(-f*rel_scale)) for f in x]

def theor_pdf(x, rel_scale):
    return [rel_scale*np.exp(-rel_scale * f) for f in x ]

def read_unix_csv(filename):
    data = []
    with open('9_data1.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, 'unix')
        for row in reader:
            for number in row:
                data.append(float(number))
    return data

def d_stat(data1, data2):
    ecdf_1 = ECDF(data1)
    x = sorted(data2)
    d_min = []
    d_max = []
    data_len = np.size(data2)
    for i in range(data_len):
        y = ecdf_1(x[i])
        d_min.append((i+1)/data_len - y)
        d_max.append(y - i/data_len)
    return np.max(d_min), np.max(d_max)

        
data1 = np.array(read_unix_csv('9_data1.csv'))          
plt.figure(50)
mean = data1.mean()
std = data1.std(ddof=1)
rel_scale_coeff = 1 / mean;
print('mean: {}, s: {}, l: {}'.format(mean, std, rel_scale_coeff))

hist, bin_edges = np.histogram(data1, 'sturges')
midpoints = (bin_edges[1:] + bin_edges[:-1])/2
x_tdf = np.arange(0.0, np.max(bin_edges), 0.001)

fig0 = plt.figure(0)
plt.title('Histogram')
plt.xticks(bin_edges)
plt.hist(data1, bins='sturges',color='gray', edgecolor='black')
bins_width = (data1.max() - data1.min()) / (np.size(bin_edges) - 1)
print('bins width: {}'.format(bins_width))

plt.grid(b=True, which='both',axis='y')
#plt.savefig('hist.png', dpi=900)

plt.figure(10)
plt.title('PDF Histogram')
plt.xticks(bin_edges)
hist_norm, _ = np.histogram(data1, 'sturges', density=True)
plt.hist(data1, bins='sturges',color='gray', edgecolor='black', density=True)
plt.plot(midpoints, hist_norm, color='black')
tpdf =  theor_pdf(x_tdf, rel_scale_coeff)



plt.plot(x_tdf, tpdf, ls='dashed', color='r')
plt.title('PDF histogram')
plt.grid(b=True, which='both',axis='y')
#plt.savefig('pdf_hist.png', dpi=900)

probs = [x/np.sum(hist) for x in hist]
cdf = np.cumsum(probs, dtype=float)
plt.figure(20)
plt.step(midpoints, cdf, where='post', color='black')
plt.grid(b=True, which='both',axis='both')
plt.xticks(midpoints)
plt.title('ECDF')

tcdf = theor_cdf(x_tdf, rel_scale_coeff)
plt.plot(x_tdf, tcdf, ls='dashed', color='r')
#plt.savefig('ecdf.png', dpi = 900)
ct = np.array([ x *(np.sum(hist)*bins_width) for x in theor_pdf(midpoints, rel_scale_coeff)]) 

chi2_emp = np.sum(((hist - ct)**2)/ct)

chi2_crit = st.chi2.ppf(0.99, np.size(bin_edges) - 2)
check_hip = chi2_crit > chi2_emp
print('chi2 crit: {}, chi2 emp: {}'.format(chi2_crit, chi2_emp))
data2 = np.array(read_unix_csv('9_data2.csv'))

dmin, dmax = d_stat(data1, data2)    
dstat_emp = np.max([dmin, dmax])
ks_crit = kstwobign.ppf(0.99)

n, m = np.size(data1), np.size(data2)
print('Dm,n : {}, Kcrit: {}'.format(dstat_emp, ks_crit))
print('D * sqrt [mn / (m + n)] : {}'.format(dstat_emp * np.sqrt(m * n / (m + n))))

dstat, pval = ks_2samp(data1, data2)
print(dstat)
print('d1 and d2 is same with prob: ', pval)
