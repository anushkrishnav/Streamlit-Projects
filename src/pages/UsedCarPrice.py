import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def write():
    st.title('Used Car Price prediction')
    @st.cache
    def load_data():
        """Reads the data from the csv files , removes the rows with missing value and resets the index"""
        data=pd.read_csv('clean_df.csv')
        data['price'] = pd.to_numeric(data['price'].str.replace('?', ''))
        data['horsepower'] = pd.to_numeric(data['horsepower'].str.replace('?', ''))
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data
    data_load_state=st.text('Loading hold on..........It might take a couple of seconds')
    data=load_data()
    data_load_state.text('')
    if st.checkbox('Show raw data'):
        st.header('raw data')
        st.write(data)
    #----------------------------------------------------------------------------------
    if st.checkbox('Show Meta data'):
        st.write(data.dtypes)
    st.header('Correlation between variables')
    st.write(data.corr())
    #----------------------------------------------------------------------
    st.header('Relationship between price and engine-size')
    sns.regplot(x="engine-size", y="price", data=data)
    plt.ylim(0,)
    st.pyplot()
    #----------------------------------------------------------------------------------

    st.write('Positive Linear relationship')
    st.header('correlation')
    st.write(data[["engine-size", "price"]].corr())
    st.header('Relationship between price and highway-mpg')
    sns.regplot(x="highway-mpg", y="price", data=data)
    st.pyplot()
    #----------------------------------------------------------------------------------

    st.header('correlation')
    st.write(data[["highway-mpg", "price"]].corr())
    st.write('As the highway milage per gallon goes up, the price goes down: this indicates an inverse/negative relationship between these two variables.')
    #----------------------------------------------------------------------------------

    st.header('Categorical variables')
    st.subheader('Price and body-style(category)')
    st.write('These are variables that describe a characteristic of a data unit, and are selected from a small group of categories.')
    sns.boxenplot(x="engine-location", y="price", data=data)
    st.pyplot()
    st.write("Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.")
    #---------------------------------------------------------------------------
    st.subheader('Price and drive-wheels(category)')
    st.write('These are variables that describe a characteristic of a data unit, and are selected from a small group of categories.')
    sns.boxenplot(x="drive-wheels", y="price", data=data)
    st.pyplot()
    st.write("Here we see that the distribution of price between these two drive-wheels categories, front and rear, are distinct enough to take drive-wheels as a potential good predictor of price.")
    st.header("Descriptive Statistical Analysis")
    st.write(data.describe())
    st.write(data.describe(include=['object']))
    #---------------------------------------------------------------------
    #Pipeline
    Input=[('scale',StandardScaler()),  ('model',LinearRegression())]
    pipe=Pipeline(Input)
    Input2=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)) , ('model',LinearRegression())]
    Ploypipe=Pipeline(Input2)
    lr=LinearRegression()
    #---------------------------------------------------------------------
    st.header("Linear Regression")
    st.write("InDependent varaibles - Highway-mpg")
    st.write("Dependent - Price")
    if st.checkbox('Highway-mpg'):
        st.header("InDependent varaibles - Highway-mpg")
        X=data[['highway-mpg']]
        Y=data['price']
        lr.fit(X,Y)
        Yhat=lr.predict(X)
        st.write("Relationship between Price and Highway MPG is")
        st.write("Price = intercept - coef* highway-mpg ")
        #st.write("Price="+str(lr.intercept_)+"+"+str(lr.coef_[0])+"*highway-mpg")
        sns.regplot(x="engine-size", y="price", data=data)
        plt.ylim(0,)
        st.pyplot()
        #-----------------------------------------------------------------------------
        st.header("Residual Plot")
        sns.residplot(X,Y)
        st.pyplot()
        #--------------------------------------------------------------------------------------------
        ax1 = sns.distplot(data['price'], hist=False, color="r", label="Actual Value")
        sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


        plt.title('Actual vs Fitted Values for Price')
        plt.xlabel('Price (in dollars)')
        plt.ylabel('Proportion of Cars')

        st.pyplot()
        plt.close()


    if st.checkbox('Engine size'):
        st.header("InDependent varaibles - Engine size")
        X=data[['engine-size']]
        Y=data['price']
        lr.fit(X,Y)
        Yhat=lr.predict(X)
        st.write("Relationship between Price and engine size is")
        st.write("Price = intercept - coef* engine size ")
        st.write("Price="+str(lr.intercept_)+"+"+str(lr.coef_[0])+"*highway-mpg")
        sns.regplot(x="engine-size", y="price", data=data)
        plt.ylim(0,)
        st.pyplot()
        #-----------------------------------------------------------------------------
        st.header("Residual Plot")
        sns.residplot(X,Y)
        st.pyplot()
        #--------------------------------------------------------------------------------------------
    
    
    if st.checkbox('Horsepower'):
        st.header("InDependent varaibles - horsepower")
        X=data[['horsepower']]
        Y=data['price']
        lr.fit(X,Y)
        '''p_val=lr.predict(np.array(102).reshape(-1,1))
        st.write("30",p_val)
        if st.checkbox('Show highway_mpg and prices'):
            st.header('Compare')
            st.write(data[['horsepower','price']])   ''' 
        Yhat=lr.predict(X)
        st.write("Relationship between Price and horsepower is")
        st.write("Price = intercept - coef* highway-mpg ")
        st.write("Price="+str(lr.intercept_)+"+"+str(lr.coef_[0])+"*highway-mpg")
        sns.regplot(x="engine-size", y="price", data=data)
        plt.ylim(0,)
        st.pyplot()
        #-----------------------------------------------------------------------------
        st.header("Residual Plot")
        sns.residplot(X,Y)
        st.pyplot()
        #--------------------------------------------------------------------------------------------
    





































    

    st.header("Multiple Linear Regression")
    st.write("InDependent varaibles - Highway-mpg horsepower curb-weight engine-size")
    st.write("Dependent - Price")
    Z = data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    lr.fit(Z, data['price'])
    Yhat=lr.predict(Z)
    #st.write("Price="+str(Ploypipe.intercept_)+"+"+str(Ploypipe.coef_[0])+"(horsepower)"+str(Ploypipe.coef_[1])+"(curb-weigh)"+str(Ploypipe.coef_[2])+"(engine-size)"+str(pipe.coef_[3])+"(highway-mpg)")
    #-------------------------------------------------------------------------------

    ax1 = sns.distplot(data['price'], hist=False, color="r", label="Actual Value")
    sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    st.pyplot()
    plt.close()

    #-------------------------------------------------------------------------------
    def PlotPolly(model, independent_variable, dependent_variabble, Name):
        x_new = np.linspace(15, 55, 100)
        y_new = model(x_new)

        plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
        plt.title('Polynomial Fit for Price ~ Length')
        ax = plt.gca()
        ax.set_facecolor((0.898, 0.898, 0.898))
        fig = plt.gcf()
        plt.xlabel(Name)
        plt.ylabel('Price of Cars')
        st.pyplot()
        plt.close()
    x = data['highway-mpg']
    y = data['price']
    f = np.polyfit(x, y, 3)
    p = np.poly1d(f)
    PlotPolly(p, x, y, 'highway-mpg')
    #----------------------------------------------
    st.header("Do predicted value make sense ?")
    st.write("Lets predict Price Using Highway-mpg")
    lr=LinearRegression()
    lr.fit(data[['highway-mpg']],data['price'])
    p_val=lr.predict(np.array(27.0).reshape(-1,1))
    st.write("30",p_val)
    if st.checkbox('Show highway_mpg and prices'):
        st.header('Compare')
        st.write(data[['price','highway-mpg']])    
    
