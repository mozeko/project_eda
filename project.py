import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler



with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'insights','prediction'], 
        icons=['house', 'gear','graph-up'], menu_icon="cast", default_index=1)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("This dashboard is using One of the airline companies dataset from Kaggle for Python Data Analysis & Data Science Diploma.")

if selected == "Home":
    st.title("Flight Fare Price Prediction App ✈")
    st.image("fl.jpg")
    st.write("Airlines play a pivotal role in global transportation, connecting people and goods across vast distances. They operate a complex network of routes, managing everything from aircraft maintenance to customer service Airlines own or lease a fleet of aircraft to transport passengers and cargo. These fleets can range from small regional jets to large intercontinental planes. Airlines factor in various operating costs, including fuel, aircraft maintenance, crew salaries, airport fees, and insurance. Ticket prices fluctuate based on supply and demand dynamics. Popular routes during peak travel seasons often have higher prices. Prices are influenced by competition within the airline industry. Airlines may adjust their fares to remain competitive with other carriers operating on the same routes Airlines generate additional revenue through ancillary services like baggage fees, seat selection, in-flight meals, and entertainment In essence, airline ticket prices are the result of a complex interplay between operational costs, market forces, and consumer behavior. By understanding these factors, travelers can make informed decisions when booking flights.")
    st.sidebar.markdown("Made With :heart_eyes: by Analyst.[Zakaria Mostafa](https://www.linkedin.com/in/zakariamostafa/) ")
    
if selected =="EDA":
    # loadind data
    df = pd.read_csv('new_data.csv')



    #body
    df1=df.groupby(['flight','airline'],as_index=False).count()
    cat_filt = st.sidebar.selectbox("Categorical",[None,'airline','stops','class'])
    # row a
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Count. Indigo",df[df['airline']=='Indigo']['airline'].count())
    a2.metric("Count.GO_FIRST",df[df['airline']=='GO_FIRST']['airline'].count())
    a3.metric("Count.AirAsia",df[df['airline']=='AirAsia']['airline'].count())
    a4.metric("Count.SpiceJet",df[df['airline']=='SpiceJet']['airline'].count())
    a5.write("The most popular companies")
    st.write("")
    # row b
    #fig = px.line(data_frame=df,x='days_left',y='price',color='class')
    #st.plotly_chart(fig, use_container_width=True)
    st.write("Ticket Price Versus Flight Duration Based on Class")
    fig =plt.figure(figsize=(20,10))
    sns.lineplot(data=df,x="duration",y="price",hue="class",palette="hls")
    st.pyplot(fig)

    fig = px.box(data_frame=df,x= 'airline',y='price',title='Airlines Vs Price',color=cat_filt)
    st.plotly_chart(fig,use_container_width=True)

    fig = px.box(data_frame=df,x='stops',y='price',title='Stops Vs Ticket Price',color=cat_filt)
    st.plotly_chart(fig,use_container_width=True)



    # rowc 
    c1 ,c2   =st.columns((5,5))

    with c1:
        st.text("Flights Count of Different Airlines")
        fig = px.bar(data_frame=df,x='airline',y='flight',color=cat_filt)
        st.plotly_chart(fig,use_container_width=True)


    with c2 :
        st.text("Classes of Different Airlines")
        fig = px.pie(data_frame=df,names='class',values='price',color=cat_filt)
        st.plotly_chart(fig,use_container_width=True)


    but = st.button("Show Data")
    if but:
        st.dataframe(df.sample(5))

if selected =="Model":
    st.title("Flight Fare Price Prediction")
    st.write('This app predicts the **flight fare price!**')
    # Loads the Boston House Price Dataset
    df_m = pd.read_csv('data_mlnn.csv')
    #Loads the Boston House Price Dataset
    X = df_m.iloc[:,:-1]
    y = df_m.iloc[:,-1]
    mmscaler=MinMaxScaler(feature_range=(0,1))
    X = mmscaler.fit_transform(X)
    def user_input_features():
        airline = st.sidebar.slider('airline',0,5)
        flight = st.sidebar.slider('flight',0,1500)
        source_city = st.sidebar.slider('source_city',0,5)
        departure_time = st.sidebar.slider('departure_time',0,5)
        stops = st.sidebar.slider('stops', 0,2)
        arrival_time = st.sidebar.slider('arrival_time',0,5)
        destination_city = st.sidebar.slider('destination_city',0,5)
        class_ = st.sidebar.slider('class',0,1)
        duration = st.sidebar.slider('duration', X.duration.min(), X.duration.max(), X.duration.mean())
        days_left = st.sidebar.slider('days_left',0,49)
        data = {'airline': airline,
            'flight': flight,
            'source_city': source_city,
            'departure_time': departure_time,
            'stops': stops,
            'arrival_time': arrival_time,
            'destination_city': destination_city,
            'class_': class_,
            'duration': duration,
            'days_left': days_left,
            
            
            }
        features = pd.DataFrame(X, index=[0])
        return features
    df = user_input_features()
    # Main Panel

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')
    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)

    st.header('Prediction Of Flight Fare Price')
    st.write(prediction)
    st.write('---')

    #################################
    # Splitting the Data into Training set and Testing Set
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)



    # Scaling the values to convert the int values to Machine Languages
    from sklearn.preprocessing import MinMaxScaler
    mmscaler=MinMaxScaler(feature_range=(0,1))

    X_train=mmscaler.fit_transform(X_train)
    X_test=mmscaler.fit_transform(X_test)
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test) 
    a={'Model Name':[], 'Mean_Absolute_Error_MAE':[] ,'Adj_R_Square':[] ,'Root_Mean_Squared_Error_RMSE':[] ,'Mean_Absolute_Percentage_Error_MAPE':[] ,'Mean_Squared_Error_MSE':[] ,'Root_Mean_Squared_Log_Error_RMSLE':[] ,'R2_score':[]}
    Results=pd.DataFrame(a)
    modelrfr = RandomForestRegressor()

    #Trainig the model with
    modelrfr.fit(X_train, y_train)
        
    # Predict the model with test data

    y_pred = modelrfr.predict(X_test)
    out=pd.DataFrame({'Price_actual':y_test,'Price_pred':y_pred})
    result=df_m.merge(out,left_index=True,right_index=True)
    #ax = plt.subplots()
    fig= plt.figure(figsize=(20,10))
    sns.lineplot(data=result,x='days_left',y='Price_actual',color='red')
    sns.lineplot(data=result,x='days_left',y='Price_pred',color='blue')
    plt.title('Days Left For Departure Versus Actual Ticket Price and Predicted Ticket Price',fontsize=20)
    plt.legend(labels=['Price actual','Price predicted'],fontsize=19)
    plt.xlabel('Days Left for Departure',fontsize=15)
    plt.ylabel('Actual and Predicted Price',fontsize=15)
    st.pyplot(fig)
