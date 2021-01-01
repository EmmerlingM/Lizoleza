import pandas as pd
import numpy as np

table = pd.read_csv("loltable.csv")
table = table.iloc[:,6:]
np.histogram(table[:,3])