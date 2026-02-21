import pandas as pd

b = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]})

print(b["one"])  # <class 'pandas.core.series.Series'>
print(b[["one"]])  # <class 'pandas.core.frame.DataFrame'>
print(b[["one", "two"]])  # <class 'pandas.core.frame.DataFrame'>
