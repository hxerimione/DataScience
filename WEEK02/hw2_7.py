#scatter plot

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'attendance', 'homework','discussion','midterm','final','score']]
plt.scatter(subset['final'],subset['score'],s=None,c=None)
plt.xlabel('final')
plt.ylabel('score')
plt.show()

