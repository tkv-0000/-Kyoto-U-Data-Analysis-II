# %%
import pandas as pd

df0 = pd.read_csv( 'Raw Data_Paylater and Non Paylater User.csv' , sep = ';' )

df0.describe()

# Let's extract the relevant columns:

# %%
df = df0[  ['IBB1', 'IBB2', 'IBB3', 'Gender', 'E-Paylater User Status', 'Monthly Income',
            'IBB4', 'P1', 'P2', 'P3', 'P4',
            'SI1', 'SI2', 'SI3', 'SI4', 'SI5', 'SI6',
            'H1', 'H2', 'H3', 'H4',
            'SC1', 'SC2', 'SC3 ', 'SC4 ', 'SC5',
            'NE1', 'NE2', 'NE3', 'NE4', 'NE5']  ]

df.describe()

# Next let's use the `rename` function to give the columns simpler variable names:

# %%
df = df.rename( columns={'E-Paylater User Status':'Status', 'Monthly Income':'Income', 'SC3 ': 'SC3', 'SC4 ':'SC4'} )   # column SC3 and SC4 have redundant spaceback in the dataset

df.describe()


# %%
replacement_value = {1: 0, 2: 1}

df['Gender'] = df['Gender'].replace(replacement_value)
df['Status'] = df['Status'].replace(replacement_value)

df.describe()

# One common phenomenon in surveying is straightlining where respondents select the same answer to the question due to various reasons (Juita et al., 2024, pg 6), which counts their responses as outliers. We will search for this phenomenon inside the dataset and remove it if detected. 

def straightline_detection(file):
    num_col = 34     # number of questions
    straightline_drop = []

    for n in range(len(file)):
        row = file.iloc[n]      # read each row inside the file
        most_common_answer = row.value_counts()      # return the row number and frequencies of the outputs inside
        most_common_answer_counts = row.value_counts().max()     #  return the row number and the output with the highest frequency
        if most_common_answer_counts > 0.7 * num_col:      # the number reponse that accounts more than 70% of the questions
            straightline_drop.append(n)

    file = file.drop(straightline_drop, axis = 0)
    return file    

df = straightline_detection(df)

# %%
# calculate the mean for each variables
df['P'] = df[['P1', 'P2', 'P3', 'P4']].mean(axis = 1)
df['SC'] = df[['SC3', 'SC4', 'SC5']].mean(axis = 1)
df['SI'] = df[['SI1', 'SI2', 'SI3', 'SI4', 'SI5', 'SI6']].mean(axis = 1)
df['NE'] = df[['NE1', 'NE2', 'NE3', 'NE4', 'NE5']].mean(axis = 1)
df['H'] = df[['H1', 'H2', 'H3', 'H4']].mean(axis = 1)
df['IBB'] = df[['IBB1', 'IBB2', 'IBB3', 'IBB4']].mean(axis = 1)

# %%
# dropping previous indicators

df = df.drop(['P1', 'P2', 'P3', 'P4', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SI1', 'SI2', 'SI3', 'SI4', 'SI5', 'SI6', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5',
              'H1', 'H2', 'H3', 'H4', 'IBB1', 'IBB2', 'IBB3', 'IBB4'], axis = 1)

# %%
df.describe()

# %% [markdown]
# For the final step, we will save the cleaned dataframe to a new file to execute in later steps of the project

# %%
df.to_csv('cleaned data for Paylater.csv', index = False)


