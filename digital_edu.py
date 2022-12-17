import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
df = pd.read_csv('train.csv')
df.drop(['id','bdate','has_photo','has_mobile','relation','life_main','career_start','last_seen','people_main','city','career_end','occupation_name'],axis = 1,inplace = True)
#df.info()
print(df['sex'].value_counts())
def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)
df['education_form'].fillna('Full-time',inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'],axis = 1,inplace= True)
df.info()
def edu_status_apply(edu):
    if edu == 'Undergraduate applicant':
        return 0
    if edu == "Student (Master's)" or edu =="Student (Bachelor's)"or edu == "Student (Specialist)":
        return 1
    if edu == "Alumnus (Master's)" or edu =="Alumnus (Bachelor's)"or edu == "Alumnus (Specialist)":
        return 2
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)
def langs_apply(langs):
    if langs.find('Русский') != -1:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(langs_apply)
def ocu_apply(ocu):
    if ocu == 'work':
        return 1
    else:
        return 0
df['occupation_type'].fillna('university',inplace = True)
df['occupation_type'] = df['occupation_type'].apply(ocu_apply)
print(df['occupation_type'].value_counts())
df.info()
x = df.drop('result',axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных покупок:', round(accuracy_score(y_test,y_pred)*100,2))
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
