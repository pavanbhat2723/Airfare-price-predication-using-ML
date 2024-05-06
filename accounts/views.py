from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth

# Create your views here.
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, 'invalid credentials...')
            return redirect('login')
    else:
        return render(request, 'login.html')
        
def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, "username taken...")
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.info(request, "email taken...")
                return redirect('register')
            else:
                user = User.objects.create_user(username=username, password=password1, email=email, first_name=first_name, last_name=last_name)
                user.save()
                messages.info(request, "user created...")
                return redirect('login')
        else:
            messages.info(request, "password not matched...")
            return redirect('register')
        return redirect('/')
    else:
        return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

def contact(request):
    return render(request, 'contact.html')

def about(request):
    return render(request, 'about.html')

def news(request):
    return render(request, 'news.html')

def predict(request):
    return render(request,"predict.html")

    
def data(request):
    if (request.method == 'POST'):
        airline = request.POST['airline']
        flight = request.POST['flight']
        source_city = request.POST['source_city']
        departure_time = request.POST['departure_time']
        stops = request.POST['stops']
        arrival_time = request.POST['arrival_time']
        destination_city = request.POST['destination_city']
        
        duration=float(request.POST['duration'])
        days_left=int(request.POST['days_left'])
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        l.fit_transform([airline,flight, source_city,departure_time,stops,arrival_time,destination_city])

        air = l.fit_transform([airline])
        fl = l.fit_transform([flight])
        src = l.fit_transform([source_city])
        dept = l.fit_transform([departure_time])
        st = l.fit_transform([stops])
        arr = l.fit_transform([arrival_time])
        dest = l.fit_transform([destination_city])
        #dur = l.fit_transform([duration])
        #days = l.fit_transform([days_left])
        import pandas as pd
        df = pd.read_csv("static/Dataset/Clean_Dataset.csv")
        df.dropna(inplace=True)
        df.isnull().sum()
        
        airline1 = l.fit_transform(df["airline"])
        flight1 = l.fit_transform(df["flight"])
        source_city1 = l.fit_transform(df["source_city"])
        departure_time1 = l.fit_transform(df["departure_time"])
        stops1 = l.fit_transform(df["stops"])
        arrival_time1 = l.fit_transform(df["arrival_time"])
        destination_city1 = l.fit_transform(df["destination_city"])
        #duration1 = l.fit_transform(df["duaration"])
        #days_left1 = l.fit_transform(df["days_left"])

        df["airline"]=airline1
        df["flight"]=flight1
        df["source_city"]=source_city1
        df["departure_time"]=departure_time1
        df["stops"]=stops1
        df["arrival_time"]=arrival_time1
        df["destination_city"]=destination_city1
        #df["duration"]=duration1
        #df["days_left"]=days_left1
        from sklearn.tree import DecisionTreeRegressor
        X_train = df[['airline','flight','source_city','departure_time','stops','arrival_time','destination_city','duration','days_left']]
        Y_train = df[['price']]
        tree = DecisionTreeRegressor()
        tree.fit(X_train,Y_train)
        import numpy as np
        pred=np.array([[air,fl,src,dept,st,arr,dest,duration,days_left]],dtype=object)
        prediction = tree.predict(pred)
        print(prediction)
        return render(request,"predict.html",{"air":airline,"fl":flight,"src":source_city,"dept":departure_time,"st":st,
                                              "arr":arrival_time,"dest":destination_city,"duration":duration,"days_left":days_left,"prediction":prediction})

    return render(request,"data.html")



def preprocess(request):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns',None)
    import os
    for dirname, _, filenames in os.walk(''):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    df=pd.read_csv('static/Dataset/Clean_Dataset.csv')
    print(df.head())
    df=df.drop('Unnamed: 0',axis=1)
    print(df.info())


    print(df.describe())
    #print(df.corr().T)
    print(df.shape)
    if df['airline'].dtype == 'object':
        print("Non-numeric values found in 'airline' column. Checking unique values:")
        print(df['airline'].unique())
        
    df1=df.groupby(['flight','airline'],as_index=False).count()
    print(df1.airline.value_counts())
    plt.figure(figsize=(8,5))
    sns.countplot(df1['source_city'],palette='hls')
    plt.title('Flights Count of Different Airlines',fontsize=15)
    plt.xlabel('Airline',fontsize=15)
    plt.ylabel('Count',fontsize=15)
    plt.show()
    df2=df.groupby(['flight','airline','class'],as_index=False).count()
    print(df2['class'].value_counts())
    plt.figure(figsize=(8,6))
    df2['class'].value_counts().plot(kind='pie',textprops={'color':'black'},autopct='%.2f',cmap='cool')
    plt.title('Classes of Different Airlines',fontsize=15)
    plt.legend(['Economy','Business'])
    plt.show()
    plt.figure(figsize=(15,5))
    sns.boxplot(x=df['airline'],y=df['price'],palette='hls')
    plt.title('Airlines Vs Price',fontsize=15)
    plt.xlabel('Airline',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    return render(request,"data.html")

def visualize(request):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns',None)
    import os
    for dirname, _, filenames in os.walk(''):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    df=pd.read_csv('static/Dataset/Clean_Dataset.csv')
    df=df.drop('Unnamed: 0',axis=1)
    plt.figure(figsize=(10,5))
    sns.boxplot(x='class',y='price',data=df,palette='hls')
    plt.title('Class Vs Ticket Price',fontsize=15)
    plt.xlabel('Class',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    plt.figure(figsize=(24,10))
    plt.subplot(1,2,1)
    sns.boxplot(x='departure_time',y='price',data=df)
    plt.title('Departure Time Vs Ticket Price',fontsize=20)
    plt.xlabel('Departure Time',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.subplot(1,2,2)
    sns.boxplot(x='arrival_time',y='price',data=df,palette='hls')
    plt.title('Arrival Time Vs Ticket Price',fontsize=20)
    plt.xlabel('Arrival Time',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    plt.figure(figsize=(24,10))
    plt.subplot(1,2,1)
    sns.boxplot(x='source_city',y='price',data=df)
    plt.title('Source City Vs Ticket Price',fontsize=20)
    plt.xlabel('Source City',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.subplot(1,2,2)
    sns.boxplot(x='destination_city',y='price',data=df,palette='hls')
    plt.title('Destination City Vs Ticket Price',fontsize=20)
    plt.xlabel('Destination City',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    plt.style.use('dark_background')
    plt.figure(figsize=(20,8))
    sns.lineplot(data=df,x='duration',y='price',hue='class',palette='hls')
    plt.title('Ticket Price Versus Flight Duration Based on Class',fontsize=20)
    plt.xlabel('Duration',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    return render(request,"data.html")


def prediction(request):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns',None)
    import os
    for dirname, _, filenames in os.walk(''):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    df=pd.read_csv('static/Dataset/Clean_Dataset.csv')
    df=df.drop('Unnamed: 0',axis=1)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    for col in df.columns:
        if df[col].dtype=='object':
            df[col]=le.fit_transform(df[col])
    x=df.drop(['price'],axis=1)
    y=df['price']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    from sklearn.preprocessing import MinMaxScaler
    mmscaler=MinMaxScaler(feature_range=(0,1))
    x_train=mmscaler.fit_transform(x_train)
    x_test=mmscaler.fit_transform(x_test)
    x_train=pd.DataFrame(x_train)
    x_test=pd.DataFrame(x_test) 
    a={'Model Name':[], 'Mean_Absolute_Error_MAE':[] ,'Adj_R_Square':[] ,'Root_Mean_Squared_Error_RMSE':[] ,'Mean_Absolute_Percentage_Error_MAPE':[] ,'Mean_Squared_Error_MSE':[] ,'Root_Mean_Squared_Log_Error_RMSLE':[] ,'R2_score':[]}
    Results=pd.DataFrame(a)
    print(Results.head())
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn import linear_model
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    import xgboost as xgb
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import GradientBoostingRegressor

# Create objects of Regression / Regressor models with default hyper-parameters

    modelmlg = LinearRegression()
    modeldcr = DecisionTreeRegressor()
    modelbag = BaggingRegressor()
    modelrfr = RandomForestRegressor()
    modelSVR = SVR()
    modelXGR = xgb.XGBRegressor()
    modelKNN = KNeighborsRegressor(n_neighbors=5)
    modelETR = ExtraTreesRegressor()
    modelRE=Ridge()
    modelLO=linear_model.Lasso(alpha=0.1)
    
    
    
    
    
    
    MM = [modelmlg, modeldcr, modelrfr, modelKNN, modelETR, modelXGR, modelbag,modelRE,modelLO]
    for models in MM:
        models.fit(x_train, y_train)
        y_pred = models.predict(x_test)
        print('Model Name: ', models)
        from sklearn import metrics

        print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))  
        print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))  
        print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
        print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))
        print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))
        def MAPE (y_test, y_pred):
            y_test, y_pred = np.array(y_test), np.array(y_pred)
            return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        result = MAPE(y_test, y_pred)
        print('Mean Absolute Percentage Error (MAPE):', round(result, 2), '%')
        r_squared = round(metrics.r2_score(y_test, y_pred),6)
        adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
        print('Adj R Square: ', adjusted_r_squared)
        print('------------------------------------------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------------------
        new_row = {'Model Name' : models,
               'Mean_Absolute_Error_MAE' : metrics.mean_absolute_error(y_test, y_pred),
               'Adj_R_Square' : adjusted_r_squared,
               'Root_Mean_Squared_Error_RMSE' : np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
               'Mean_Absolute_Percentage_Error_MAPE' : result,
               'Mean_Squared_Error_MSE' : metrics.mean_squared_error(y_test, y_pred),
               'Root_Mean_Squared_Log_Error_RMSLE': np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
               'R2_score' : metrics.r2_score(y_test, y_pred)}
        Results = Results.append(new_row, ignore_index=True)
    #------------------------------------------------------------
    print(Results)
    models=['LinearRegression','DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','ExtraTreesRegressor','XGBRegressor','BaggingRegressor','Ridge Regression','Lasso Regression']
    result=pd.DataFrame({'Model_Name':models})
    result['Adj_R_Square']=Results['Adj_R_Square']
    result['Mean_Absolute_Error_MAE']=Results['Mean_Absolute_Error_MAE']
    result['Root_Mean_Squared_Error_RMSE']=Results['Root_Mean_Squared_Error_RMSE']
    result['Mean_Absolute_Percentage_Error_MAPE']=Results['Mean_Absolute_Percentage_Error_MAPE']
    result['Mean_Squared_Error_MSE']=Results['Mean_Squared_Error_MSE']
    result['Root_Mean_Squared_Log_Error_RMSLE']=Results['Root_Mean_Squared_Log_Error_RMSLE']
    result['R2_score']=Results['R2_score']
    result=result.sort_values(by='Adj_R_Square',ascending=False).reset_index(drop=True)
    print(result)
    modelETR.fit(x_train, y_train)
    y_pred = modelETR.predict(x_test)
    df_bk=df.copy()
    out=pd.DataFrame({'Price_actual':y_test,'Price_pred':y_pred})
    result=df_bk.merge(out,left_index=True,right_index=True)
    print(result.sample(10))
    plt.figure(figsize=(20,8))
    sns.lineplot(data=result,x='days_left',y='Price_actual',color='red')
    sns.lineplot(data=result,x='days_left',y='Price_pred',color='blue')
    plt.title('Days Left For Departure Versus Actual Ticket Price and Predicted Ticket Price',fontsize=20)
    plt.legend(labels=['Price actual','Price predicted'],fontsize=19)
    plt.xlabel('Days Left for Departure',fontsize=15)
    plt.ylabel('Actual and Predicted Price',fontsize=15)
    plt.show()
    plt.figure(figsize=(10,5))
    sns.regplot(x='Price_actual',y='Price_pred',data=result,color='cyan')
    plt.title('Actual Price  Vs  Predicted Price ',fontsize=20)
    plt.xlabel('Actual Price',fontsize=15)
    plt.ylabel('Predicted Price',fontsize=15)
    plt.show()
    return render(request,"data.html")


def logout(request):
    auth.logout(request)
    return redirect("/")

from .models import Review


def comment_Review(request):
    print("amal")
    if request.method == 'POST':
        print("hi")
        name = request.POST['name']
        email = request.POST['email']
        review_comment = request.POST['message']
        # Create a new review object
        comment_review = Review.objects.create(name=name, email=email,review_comment=review_comment)
        comment_review.save()

        

        # Display a success message using Django's messages framework
        messages.success(request, "Your review has been successfully submitted.")

        # Redirect back to the review page, passing the reviews to display
        return render(request, 'index.html')
    print("hello")
    return render(request, 'review1.html')

def view_review(request):
    comment_review=Review.objects.all()
    return render(request,'review1.html',{'comment_review':comment_review})



