from src.smear_function import smear_pipeline
import argparse
import yaml
from src.loader import Loader
import os

# need to rename files to be able to run the code, do not see other options here,
# names are too random to define a pattern, follow order of files in excel sheet

def arguments_parser():
    """
    Parse arguments from config file
    """

    parser = argparse.ArgumentParser('Tuberculosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser

parser = arguments_parser()
pars_arg = parser.parse_args()


with open(pars_arg.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


periods = ['2141', '2142', '2143', '2144', '2145', '2151','2152', '2153','2154','2156', '2162', '2163', '2164', '2165', '2166']
type = ['_MTB', '_M.kansasii', '_M.intracellulare', '_M.chelonae', '_M.carpae', '_M.malmoense', '']
repeat =['_wdh','']

num_bacilli_0 = []
order_0 = []
num_bacilli_1 = []
order_1 = []
num_bacilli_2 = []
order_2 = []
num_bacilli_3 = []
order_3 = []
for period in periods:
    for number in range(1, 177):
        for wdh in repeat:
            for i in range(0,4):
                for t in type:
                    path = 'C:/Users/matteo/Downloads/extern_Synlab_' + period + '_' + str(number) + wdh + '_' + str(i) + t + '.czi'
                    #check if file exists

                    if os.path.isfile(path):



                        ld = Loader(path, 'None')
                        ld.load()
                        img = ld.data_array
                        if i == 0:
                            num_bacilli_0.append(smear_pipeline(config, img, ld))
                            order_0.append(period + '_' + str(number) + wdh + '_' + str(i) + t)
                        if i == 1:
                            num_bacilli_1.append(smear_pipeline(config, img, ld))
                            order_1.append(period + '_' + str(number) + wdh + '_' + str(i) + t)

                        if i == 2:
                            num_bacilli_2.append(smear_pipeline(config, img, ld))
                            order_2.append(period + '_' + str(number) + wdh + '_' + str(i) + t)
                        if i == 3:
                            num_bacilli_3.append(smear_pipeline(config, img, ld))
                            order_3.append(period + '_' + str(number) + wdh + '_' + str(i) + t)


import matplotlib.pyplot as plt
import numpy as np

# plot 4  boxplots
plt.figure(figsize=(10, 10))
plt.boxplot([num_bacilli_0, num_bacilli_1, num_bacilli_2, num_bacilli_3], labels=['0', '1', '2', '3'])
# add a line for the mean
plt.axhline(y=np.mean(num_bacilli_0), color='b', linestyle='-')
plt.axhline(y=np.mean(num_bacilli_1), color='r', linestyle='-')
plt.axhline(y=np.mean(num_bacilli_2), color='g', linestyle='-')
plt.axhline(y=np.mean(num_bacilli_3), color='y', linestyle='-')
#add the scattered data points for each boxplot
plt.scatter([1+ np.random.normal(0,0.1,len(num_bacilli_0)) ], num_bacilli_0, color='b')
plt.scatter([2+ np.random.normal(0,0.1,len(num_bacilli_0))], num_bacilli_1, color='r')
plt.scatter([3+ np.random.normal(0,0.1,len(num_bacilli_0))], num_bacilli_2, color='g')
plt.scatter([4+ np.random.normal(0,0.1,len(num_bacilli_0))], num_bacilli_3, color='y')

plt.title('Number of bacilli per smear')
plt.xlabel('Severness grade')
plt.ylabel('Number of bacilli')
plt.show()
plt.savefig('/boxplot.png')

# save the data in a csv file
import pandas as pd
df = pd.DataFrame({'order_0': order_0, 'num_bacilli_0': num_bacilli_0, 'order_1': order_1, 'num_bacilli_1': num_bacilli_1, 'order_2': order_2, 'num_bacilli_2': num_bacilli_2, 'order_3': order_3, 'num_bacilli_3': num_bacilli_3})
df.to_csv('C:/Users/matteo/Downloads/boxplot.csv')