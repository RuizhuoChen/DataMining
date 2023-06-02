import math
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import association_rules

for num in range (100):
        data = pd.read_csv(r"mammographic_masses.csv", delimiter=",", header=0, usecols=['Age', 'Severity'])

        # preprocessing: change to one hot encoding so as to be able to use apriori from mlxtend
        d = data.values.tolist()
        # removing ? values
        i = 0
        while (True):
            if (d[i][0] == '?' or d[i][1] == '?'):
                del d[i]
                i -= 1
            i += 1
            if (i > len(d) - 1):
                break

        # adding attributes
        for i in range(len(d)):
            for j in range(len(d[i])):
                if j == 0 and int(d[i][j]) < num:
                    d[i][j] = 'Young'
                elif j == 0 and int(d[i][j]) >= num:
                    d[i][j] = 'Old'
                d[i][j] = data.columns[j] + "=" + str(d[i][j])

        te = TransactionEncoder()
        te_ary = te.fit(d).transform(d)

        df = pd.DataFrame(te_ary, columns=te.columns_)
        # computing frequent itemsets and association rules
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

        a = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

        # visualizing association rules results
        print(num)
        print(a[["antecedents", "consequents", "support", "confidence"]])