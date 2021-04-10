#histogram
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'attendance', 'homework','discussion','midterm','final','score','grade']]
subset['grade'].hist(bins=np.arange(6),rwidth=0.8,grid=False)
plt.title('grade')
#plt.xticks([0,1,2,3])
plt.show()

