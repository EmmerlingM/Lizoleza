import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
table = pd.read_csv("loltable.csv")
table = table.iloc[:,6:]
table.iloc[:,3].hist()
nptable = np.array(table)

fig, ax = plt.subplots()
ax.hist(nptable[:,3], bins=200)
plt.xlabel("Gold")
plt.ylabel("Frequency of ocurrence")
ax.set_axisbelow(True)
ax.grid(b=True, which='both', axis='both', color='black', linewidth=0.7, alpha=0.2)

regressor = sk.LinearRegression()