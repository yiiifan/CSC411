import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# get the features from the file features.txt
features = list()
with open('UCI_HAR_dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]
print('No of Features: {}'.format(len(features)))

# get the data from txt files to pandas dataffame
X_train = pd.read_csv('UCI_HAR_dataset/train/X_train.txt', delim_whitespace=True, header=None, names=features)

# add subject column to the dataframe
X_train['subject'] = pd.read_csv('UCI_HAR_dataset/train/subject_train.txt', header=None, squeeze=True)

y_train = pd.read_csv('UCI_HAR_dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

# put all columns in a single dataframe
train = X_train
train['Activity'] = y_train
train['ActivityName'] = y_train_labels
train.sample()

# get the data from txt files to pandas dataffame
X_test = pd.read_csv('UCI_HAR_dataset/test/X_test.txt', delim_whitespace=True, header=None, names=features)

# add subject column to the dataframe
X_test['subject'] = pd.read_csv('UCI_HAR_dataset/test/subject_test.txt', header=None, squeeze=True)

# get y labels from the txt file
y_test = pd.read_csv('UCI_HAR_dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})


# put all columns in a single dataframe
test = X_test
test['Activity'] = y_test
test['ActivityName'] = y_test_labels
test.sample()

# list of feaatures that we think
features = ['tBodyAccMagmean','tBodyAccMagstd','tBodyAccJerkMagmean','tBodyAccJerkMagstd','tBodyGyroMagmean',
     'tBodyGyroMagstd','tBodyGyroJerkMagmean','tBodyGyroJerkMagstd','fBodyAccMagmean','fBodyAccMagstd',
     'fBodyBodyAccJerkMagmean','fBodyBodyAccJerkMagstd','fBodyBodyGyroMagmean','fBodyBodyGyroMagstd',
     'fBodyBodyGyroJerkMagmean','fBodyBodyGyroJerkMagstd','fBodyBodyGyroMagmeanFreq','fBodyBodyGyroJerkMagmeanFreq',
    'fBodyAccMagmeanFreq','fBodyBodyAccJerkMagmeanFreq','fBodyAccMagskewness','fBodyAccMagkurtosis',
    'fBodyBodyAccJerkMagskewness', 'fBodyBodyAccJerkMagkurtosis','fBodyBodyGyroMagskewness','fBodyBodyGyroMagkurtosis',
    'fBodyBodyGyroJerkMagskewness','fBodyBodyGyroJerkMagkurtosis','angletBodyAccJerkMeangravityMean','angletBodyAccMeangravity',
     'angletBodyGyroJerkMeangravityMean','angletBodyGyroMeangravityMean','angleXgravityMean',
     'angleYgravityMean','angleZgravityMean']

# train logistic regression (one vs rest)
log_reg = LogisticRegression(C = 30)
log_reg.fit(X_train.drop(['Activity', 'ActivityName', 'subject'],axis=1), y_train_labels)
predicted = log_reg.predict(X_test.drop(['Activity', 'ActivityName', 'subject'],axis=1))

# rows = labels = 6 , columns = features = 561 for weight matrix

# to change the weights to absolute values
absolute_coeff = np.absolute(log_reg.coef_)
print(absolute_coeff[:3])
# get all features into one single array
all_features = X_test.drop(['Activity', 'ActivityName', 'subject'], axis=1).columns

laying_coeff = absolute_coeff[0]

# store features and their weights in a dataframe
df_laying_coeff = pd.DataFrame()
df_laying_coeff['features'] = all_features
df_laying_coeff['weights'] = laying_coeff

# get all features sorted by their weights
imp_features_laying = df_laying_coeff.sort_values(by='weights', ascending=False).features

sitting_coeff = absolute_coeff[1]

# store features and their weights in a dataframe
df_sitting_coeff = pd.DataFrame()
df_sitting_coeff['features'] = all_features
df_sitting_coeff['weights'] = sitting_coeff

# get all features sorted by their weights
imp_features_sitting = df_sitting_coeff.sort_values(by='weights', ascending=False).features

standing_coeff = absolute_coeff[2]

# store features and their weights in a dataframe
df_standing_coeff = pd.DataFrame()
df_standing_coeff['features'] = all_features
df_standing_coeff['weights'] = standing_coeff

# get all features sorted by their weights
imp_features_standing= df_standing_coeff.sort_values(by='weights', ascending=False).features

walking_coeff = absolute_coeff[3]

# store features and their weights in a dataframe
df_walking_coeff = pd.DataFrame()
df_walking_coeff['features'] = all_features
df_walking_coeff['weights'] = walking_coeff

# get all features sorted by their weights
imp_features_walking= df_walking_coeff.sort_values(by='weights', ascending=False).features

walking_down_coeff = absolute_coeff[4]

# store features and their weights in a dataframe
df_walking_down_coeff = pd.DataFrame()
df_walking_down_coeff['features'] = all_features
df_walking_down_coeff['weights'] = walking_down_coeff

# get all features sorted by their weights
imp_features_walking_down= df_walking_down_coeff.sort_values(by='weights', ascending=False).features

walking_up_coeff = absolute_coeff[5]

# store features and their weights in a dataframe
df_walking_up_coeff = pd.DataFrame()
df_walking_up_coeff['features'] = all_features
df_walking_up_coeff['weights'] = walking_up_coeff

# get all features sorted by their weights
imp_features_walking_up= df_walking_up_coeff.sort_values(by='weights', ascending=False).features


from itertools import chain

# create a set of features from all the classe labels
top_features = set(chain( imp_features_laying[:100], imp_features_sitting[:100], imp_features_standing[:100], \
                         imp_features_walking[:100], imp_features_walking_down[:100], imp_features_walking_up[:100] ))

# no of unique features from all class labels
print("\n\nWe got {} unique features from top 100 features of all classes.".format(top_features.__len__()))

print('\n\nWe got {} common features from the reduced feature set and top 100 important features from all classes\n'\
     .format(len(set(top_features).intersection(set(features)))))

# 20 common Features
print('\n\n20 Common features')
print('-------------------------')
for f in set(top_features).intersection(set(features)):
    print('{},'.format(f), end='\t')

print('\n\n-------------------------------------------------------')
print('Features that we think important, but they are not in top (A/c to model)')
print('-------------------------------------------------------')
for f in set(features) - (set(top_features).intersection(set(features))):
    print('{},'.format(f), end='\t')

print('\n\n----------------------------------------------------------')
print('Some of the Features that we missed from important features')
print('---------------------------------------------------------------')
for f in list(set(top_features) - set(features))[:50]:
    print('{},'.format(f), end='\t')

# labels *(by) features matrix
print(absolute_coeff.shape)

# max_weight of each label irrespective of what the feature is
max_weight_of_labels = absolute_coeff.max(axis=1)

no_of_features = absolute_coeff.shape[1]
# combined weights for each label
weights = list()
for i in range(no_of_features):
    weights.append(float(sum(np.divide(absolute_coeff[:,i], max_weight_of_labels))))

# create a dataframe to store the features and new_weights together
df_imp_features = pd.DataFrame()

df_imp_features['features'] = all_features
df_imp_features['weights'] = pd.Series(weights)

# dataframe before sorting
df_imp_features.head()


# dataframe after sorting
df_imp_features.sort_values(by='weights', ascending=False).head()

# get features in their descending order of their weights from this dataframe
imp_features = df_imp_features.sort_values(by='weights', ascending=False).features.values
imp_features[:5]

common_features = set(features).intersection(set(imp_features[:287]))
print('\nNo of common features : {}'.format(len(common_features)))
print('-------------------------')
for f in common_features:
    print('{},'.format(f), end='\t')

print('\n\n---------------------------------------------------------')
print('Features that we think important, but they are not in top')
print('----------------------------------------------------------')
for f in set(features) - (common_features):
    print('{},'.format(f), end='\t')

print('\n\n----------------------------------------------------------')
print('Some of the Features that we missed from important features')
print('---------------------------------------------------------------')
for f in list(set(imp_features) - set(features))[:50]:
    print('{},'.format(f), end='\t')

# important features from first method (one vs rest)
print('\nWe have {} important features from our first method.\n'.format(len(top_features)))
print('we will also consider top 287 features from the second method to compare them.\n')

print('Both methods got {} features in common.'.\
      format(len(set(top_features).intersection(set(imp_features[:287])))))