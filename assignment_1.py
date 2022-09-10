#!/usr/bin/env python3

# ------------------------- #
# << HOMEWORK GROUP :: 3 >> #
# ------------------------- #
# <<   GROUP MEMBERS  >>    #
# ------------------------- #
# 1. DHRUV PATEL
# 2. BHAVANASI APURVA
# 3. SAMUEL HELMRATH
# 4. ABRAR AHMED MOHAMMED
# ------------------------- #


# << imports >>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(44)  # for reproducibility

# << Reading the data >>
# Reading the csv file into a dataframe
df_RAW = pd.read_csv('insurance.csv')

# Copying the Raw dataframe into another one for manipulation
df = df_RAW.copy()

# Removing the Categorical Attributes Columns
# and only keeping the real-valued features
df = df.drop(['sex', 'smoker', 'region'], axis=1)

# << Acquiring the necessary input-output >>

# Input Array : X

# We also need to add bias column at the front
# Otherwise, normal equation would give wrong results
X = df[['age','bmi', 'children']].to_numpy()
print(f'X Array Without Bias Term Added : \n{X}')
bias = np.ones((len(X),1))                      # bias term
X = np.concatenate((bias, X), axis=1)           # adding at column [0]
print(f'\nX Array With Bias Term Added : \n{X}')

# Age Array : age
age = df['age'].to_numpy().reshape((-1,1))

# BMI Array : bmi
bmi = df['bmi'].to_numpy().reshape((-1,1))

# Children Array : children
children = df['children'].to_numpy().reshape((-1,1))

# Output Array : Y
Y = df[['charges']].to_numpy()


# << Getting the train-test split indexes >>
'''
1. train_split : This value defines the size of the train split in fraction
   while the test_split will be (1 - train_split) in fraction
2. idx_train   : It has the indices of randomly chosen data points for training set
3. idx_test    : It has the indices of randomly chosen data points for test set
'''

train_split = 0.8
name_save = int(train_split*100)
print(f'\nTrain Split : {name_save} percent')

# n_data is a list from 0 to length of the dataset (X and/or Y)
n_data = list(range(len(X)))
n_samples = int(len(X)*train_split)

# randomly selecting indices to use for train and test
idx_train = np.random.choice(n_data, n_samples, replace=False)
idx_test = list(set(n_data) - set(idx_train))

# << Getting the train and test : Features and Targets >>
train_X = X[idx_train]
train_Y = Y[idx_train]

test_X = X[idx_test]
test_Y = Y[idx_test]


# << Using Normal Equation for Linear Regression >>

def getTheta(x, y):
    '''
    Args ::
            X :: Input Feature Array containing each instance
            Y :: Output Target Array containing each instance
    Returns ::
            theta :: Hypothesis Parameters (weight vector)
    '''

    theta = (np.linalg.inv(x.T.dot(x))).dot(x.T).dot(y)
    return theta


# << Applying getTheta function on Train X and Y split >>
theta_train = getTheta(train_X, train_Y)
print(f'\ntheta  :: \n{theta_train}')

# << Predicting the output for train and test set using the theta value >>
predicted_train_y = train_X.dot(theta_train)
predicted_test_y = test_X.dot(theta_train)


# << Calculating MSE Loss >>

def getMSE(A, B):
    mse = np.square(np.subtract(A,B)).mean()
    return mse


# << Modeling Power :: Error on Train Set >>
mod_power = getMSE(train_Y, predicted_train_y)

# << Generalization Power :: Error on Test Set>>
gen_power = getMSE(test_Y, predicted_test_y)



# << Plotting >>

# 1) Regression Line as a function of the bmi

y_bmi =  bmi.dot(theta_train[2])
plt.scatter(bmi,Y,s=10,marker='o', label='BMI datapoints')
plt.plot(bmi,y_bmi,c='red', label='Regression Line')
plt.legend(loc='best')
plt.plot()
plt.xlabel("Feature ---> BMI")
plt.ylabel("Target_Variable ---> Insurance Charges")
plt.title('Regression Line as a function of BMI : ' + str(name_save) + '%')
plt.savefig("plots/BMI_"+str(name_save)+"%.png", format="png", dpi=1200)
plt.show()

# 2) Regression line as function of the age

y_age = age.dot(theta_train[1])
plt.scatter(age,Y,s=10,marker='o', label='AGE datapoints')
plt.plot(age,y_age,c='red', label='Regression Line')
plt.legend(loc='best')
plt.plot()
plt.xlabel("Feature ---> AGE")
plt.ylabel("Target_Variable ---> Insurance Charges")
plt.title('Regression Line as a function of age : '+str(name_save)+'%' )
plt.savefig("plots/AGE_"+str(name_save)+"%.png", format="png", dpi=1200)
plt.show()


# 3) Regression line as function of the number of children

y_children = children.dot(theta_train[3])
plt.scatter(children,Y,s=10,marker='o', label='Children datapoints')
plt.plot(children,y_children,c='red', label='Regression Line')
plt.legend(loc='best')
plt.plot()
plt.xlabel("Feature ---> Number of Children")
plt.ylabel("Target_Variable ---> Insurance Charges")
plt.title('Regression Line as a function of number of children : ' + str(name_save) + '%')
plt.savefig("plots/CHILD_"+str(name_save)+"%.png", format="png", dpi=1200)
plt.show()


'''
To get the modeling and generalization error as functions of the training set
we need to run the entire code for each training size and store the errors.
'''

# initializing empty lists for storing
modeling_error = []
generalization_error = []

for i in range(2,9,1):

    train_split = i/10
    test_split = 1 - train_split

    # n_data is a list from 0 to length of X and/or Y
    n_data = list(range(len(X)))
    n_samples = int(len(X)*train_split)

    # randomly selecting indices to use for train and test
    idx_train = np.random.choice(n_data, n_samples, replace=False)
    idx_test = list(set(n_data) - set(idx_train))

    train_X = X[idx_train]    # train features
    train_Y = Y[idx_train]    # train targets

    test_X = X[idx_test]      # test features
    test_Y = Y[idx_test]      # test targets

    theta_train = getTheta(train_X, train_Y)

    # Predicting the output for each instance using the theta value obtained
    predicted_train_y = train_X.dot(theta_train)
    predicted_test_y = test_X.dot(theta_train)

    # Modeling Power :: Error on Train set
    mod_power = getMSE(train_Y, predicted_train_y)

    # Generalization Power :: Error on the Test set
    gen_power = getMSE(test_Y, predicted_test_y)


    # Append values
    modeling_error.append(mod_power)
    generalization_error.append(gen_power)



plot_xx = np.arange(20,90,10)
plt.plot(plot_xx, modeling_error, c='blue', label='Modeling Error')
plt.plot(plot_xx, generalization_error, c='red', label='Generalization Error')
plt.legend(loc='best')
plt.plot()
plt.xlabel("Train Split (in %)")
plt.ylabel("Mean Squared Error Loss")
plt.title('MSE Loss vs Train Split (in %)')
plt.savefig("plots/LOSS.png", format="png", dpi=1200)
plt.show()



