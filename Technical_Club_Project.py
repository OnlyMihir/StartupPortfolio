# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ch=0
while(ch!=5):
    print("Enter 1 to generate Salary vs Experience Graph.")
    print("Enter 2 to generate report to know in which startup to invest money in.")
    print("Enter 3 to generate report of what should be the salary of an employee.")
    ch=int(input("Enter 6 to exit."))
    
    if(ch==1):
        # Simple Linear Regression

        # Importing the dataset
        file=input("Enter Salary and Experience path")
        dataset = pd.read_csv(file)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

        # Feature Scaling
        """from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)"""

        # Fitting Simple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)

        # Visualising the Training set results
        plt.scatter(X_train, y_train, color = 'red')
        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()

        # Visualising the Test set results
        plt.scatter(X_test, y_test, color = 'red')
        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()

    if(ch==2):
        # Multiple Linear Regression

        # Importing the dataset
        file=input("Enter startup data path")
        dataset = pd.read_csv(file)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 4].values

        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder = LabelEncoder()
        X[:, 3] = labelencoder.fit_transform(X[:, 3])
        onehotencoder = OneHotEncoder(categorical_features = [3])
        X = onehotencoder.fit_transform(X).toarray()

        # Avoiding the Dummy Variable Trap
        X = X[:, 1:]

        # Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        """from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)"""

        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)

    if(ch==3):
        # Random Forest Regression

        # Importing the dataset
        file=input("Enter positional salary path")
        dataset = pd.read_csv(file)
        X = dataset.iloc[:, 1:2].values
        y = dataset.iloc[:, 2].values

        # Splitting the dataset into the Training set and Test set
        """from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

        # Feature Scaling
        """from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)"""

        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X, y)

        # Predicting a new result
        y_pred = regressor.predict(6.5)

        # Visualising the Random Forest Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (Random Forest Regression)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()
