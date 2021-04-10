#boxplot
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_excel('./db_score.xlsx')
subset = df[['sno', 'attendance', 'homework','discussion','midterm','final','score']]

boxplot=df.boxplot(column=['discussion'])
plt.show()
