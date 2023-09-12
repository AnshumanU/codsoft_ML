import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_movie=pd.read_csv('/content/Movie_preprocessed.csv',sep="::",engine='python')
df_movie.dropna(inplace=True)
df_movie.head()
df_movie.shape
df_movie.describe()
df_movie.isna().sum()
df_ratings=pd.read_csv("/content/ratings_preprocessed.csv", sep="::", engine="python")
df_ratings.dropna(inplace=True)
df_ratings.head(10)
df_ratings.shape
df_ratings.describe()
df_ratings.isna().sum()
df_users=pd.read_csv("/content/users_preprocessed.csv", sep="::", engine="python")
df_users.dropna(inplace=True)
df_users.head(10)
df_users.shape
df_users.describe()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df_users['Gender']= labelencoder.fit_transform(df_users['Gender'])

df_users.head()
df_users.isna().sum()
df_data=pd.concat([df_movie,df_ratings,df_users],axis=1)
df_data.dropna()
df_data.head(10)
df_data.shape
df2=df_data.drop(["Occupation","Zip-code","Timestamp"],axis=1)
df2.head()
df2.describe()
df2.isna().sum()
df_final=df2.dropna()
df_final.shape
sns.countplot(x=df_final['Gender'],hue=df_final['Ratings'])
df_final.Age.plot.hist(bins=25)
plt.ylabel("MovieIds")
plt.xlabel("Ratings")
df_final['Ratings'].value_counts().plot(kind='bar')
plt.show()
df_final['MovieID'].plot.hist(bins=25)
plt.xlabel("MovieIs")
plt.ylabel("Ratings")
df_final['Age'].plot.hist(bins=10)
# plt.xlabel("Ratings")
# plt.ylabel("Age")
sns.countplot(x=df_final['Age'],hue=df_final['Ratings'])
df_final.head()
input=df_final.drop(['Ratings','MovieName','Genre','MovieIDs'], axis=1)
target=df_final['Ratings']
target.head()
input.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(input)
scaled_df = pd.DataFrame(scaled_data,
                         columns=input.columns)
scaled_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(input,target,test_size=0.3)
print(Y_train)
print(Y_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

X_test = np.array(X_test)
model.predict(X_test)