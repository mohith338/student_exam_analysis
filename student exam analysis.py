import catboost
import pandas as pd
from jedi.inference.utils import to_list
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv(r'C:\Users\User\Downloads\StudentsPerformance.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df.drop_duplicates().sum()

df.columns=df.columns.str.strip().str.lower().str.replace(' ',"_")
df.info()

df['gender']=df['gender'].fillna('unknown')
df['race/ethnicity']=df['race/ethnicity'].fillna('unknown')
df['parental_level_of_education']=df['parental_level_of_education'].fillna('unknown')
df['lunch']=df['lunch'].fillna('unknown')
df['test_preparation_course']=df['test_preparation_course'].fillna('unknown')
df['math_score']=df['math_score'].fillna(df['math_score'].mean())
df['reading_score']=df['reading_score'].fillna(df['reading_score'].mean())
df['writing_score']=df['writing_score'].fillna(df['writing_score'].mean())

total_records=df.shape[0]
print('total_records', total_records)

math_score_analysis=df['math_score'].mean()
print('math_score_analysis', math_score_analysis)

reading_score_analysis=df['reading_score'].mean()
print('reading_score_analysis', reading_score_analysis)

writing_score_analysis=df['writing_score'].mean()
print('writing_score_analysis', writing_score_analysis)

order=df.columns.tolist()
print(order)

top_result=df.sort_values(by=['math_score','writing_score','reading_score'],ascending=[False,False,False]).head(10)
print(top_result[['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']])


lower_result=df.sort_values(by=['math_score','writing_score','reading_score'],ascending=[True,True,True]).head(10)
print(lower_result[['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']])

import matplotlib.pyplot as plt
df.plot(kind='scatter', x='math_score', y='reading_score')
plt.title('analysis between math and reading score')
plt.xlabel('math_score')
plt.ylabel('reading_score')
plt.show()

df.plot(kind='scatter', x='math_score', y='writing_score')
plt.title('analysis between math and writing score')
plt.xlabel('math_score')
plt.ylabel('writing_score')
plt.show()

top_result.plot(kind='bar')
plt.title('top_results')
plt.show()

lower_result.plot(kind='bar')
plt.title('lower_results')
plt.show()

x=df[['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']]
y=df['math_score']


cat_columns=['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']

model=catboost.CatBoostRegressor(iterations=100, learning_rate=0.01, depth=5,verbose=0)
model.fit(x,y,cat_features=cat_columns)

gender=input("enter the gender")
race_ethnicity=input('enter the race value')
parental_level_of_education=input('enter the parental_level_of_education')
lunch=input('enter the lunch type')
test_preparation_course=input('enter the test_preparation_course')
reading_score=input('enter reading_score')
writing_score=input('enter writing_score')

new_data=pd.DataFrame({
    'gender': [gender],
    'race/ethnicity': [race_ethnicity],
    'parental_level_of_education': [parental_level_of_education],
    'lunch': [lunch],
    'test_preparation_course': [test_preparation_course],
     'reading_score': [reading_score],
'writing_score': [writing_score]
})

predicition=model.predict(new_data)
print('predicition', predicition[0])

