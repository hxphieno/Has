from collections import Counter

import pandas as pd


def cp_name(s1, s2):
    return s1 == s2 or s1[::-1] == s2

df=pd.read_csv("Jpop-3.csv",header=0)
print(df.head())
df=df.iloc[:,2:]


all_elements = [elem for elem in df.values.flatten() if pd.notna(elem)]
element_counts = Counter(all_elements)

for element, count in element_counts.items():
    print(f"元素：'{element}'，出现了 {count} 次")
print(element_counts)