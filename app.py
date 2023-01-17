import streamlit as st
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def model(arr):
    nparr = np.asarray(arr)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(ts),random_state=int(rs))
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    pred = lr.predict([nparr])
    return pred
if __name__=="__main__":

    st.set_page_config(
        page_title="Ml-LifeCycle",
        page_icon="📚",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': 'https://www.extremelycoolapp.com/help',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"}
    )
    hide_menu_style ="""
            <style>
             #MainMenu {visibility: hidden;}
             </style>
             """

    st.markdown(hide_menu_style, unsafe_allow_html=True)
    no_sidebar_style = """
    <style>
        div[data-testid="collapsedControl"] {display: none;}
    </style>
    """
    st.header('Machine Learning LifeCycle')
    if st.button("Reset"):
        st.session_state['steps'] = '0'

    st.header("Step 1: Choose Dataset")
    option = st.selectbox("Select",
    ('Advertising dataset', 'Placement Dataset', 'Salary Dataset'))
    if option=='Advertising dataset':
        data = "./DataSets/regression/advertising.csv"
    elif option=='Placement Dataset':
        data = "./DataSets/regression/placement.csv"
    elif option==('Salary Dataset'):
        data = "./DataSets/regression/Salary_Data.csv"
    else:
        pass
    
    df = pd.read_csv(data)
    if st.button('Enter',key='1'):
        st.header("DataSet")
        st.dataframe(df)
        st.session_state['steps'] = '1'

    st.header("Step 2: Split Features Data & Labels Data")
    if 'advertising' in data:
        y = df['Sales']
        x = df.drop(columns='Sales',axis = 1)
    elif 'placement' in data:
        y = df['placed']
        x = df.drop(columns='placed',axis = 1)
    elif 'Salary_Data' in data:
        y = df['Salary']
        x = df.drop(columns='Salary',axis = 1)

    st.header("Labels & Features")
    if st.button('Enter',key='2'):
        st.header(f"{st.session_state['steps']}")
        if st.session_state['steps'] =='1':
            col1, col2 = st.columns(2)

            with col1:
                st.header("Labels")
                st.dataframe(x)
            with col2:
                st.header("Features")
                st.dataframe(y)
            st.session_state['steps'] ='2'
        else:
            st.error("you skipped previous steps")

    st.header("Step 3: Train Test Split")
    ts = st.select_slider('Test size',options=['0.1', '0.2', '0.4', '0.4'])
    rs = st.number_input("Enter random number :",min_value=1, max_value=100,step=None)
    if st.button('Enter',key='4'):
        if st.session_state['steps'] =='2':
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(ts),random_state=int(rs))
            cl1, cl2,cl3,cl4 = st.columns(4)
            with cl1:
                st.header("x_train")
                st.dataframe(x_train)
            with cl2:
                st.header("x_test")
                st.dataframe(x_test)
            with cl3:
                st.header("y_train")
                st.dataframe(y_train)
            with cl4:
                st.header("y_test")
                st.dataframe(y_test)
            st.session_state['steps'] ='3'
        else:
            st.error("you skipped previous steps")
    st.header("Step 4:Model Training")
    if st.button('Enter',key='5'):
        if st.session_state['steps'] =='3':
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(ts),random_state=int(rs))
            if 'placement' in data:
                lor = LogisticRegression()
                with st.spinner('Model Training...'):
                    time.sleep(10)
                lor.fit(x_train,y_train)
                st.header(f"Accuracy :{lor.score(x_test,y_test)*100}")
            else:
                lr = LinearRegression()
                lr.fit(x_train,y_train)
                with st.spinner('Model Training...'):
                    time.sleep(1)

                st.info("Model succesfully trained..")
                st.header(f"Accuracy :{lr.score(x_test,y_test)*100}")
            st.session_state['steps'] ='4'
        else:
            st.error("you skipped previous steps")
    st.header("Step 5:Create Interface for your machine learning model")
    if st.button('Enter',key='7'):
        try:
            if st.session_state['steps'] =='4':
                with st.spinner('Creating interface for your model...'):
                    time.sleep(5)
                if 'Salary_Data' in data:
                    YearsExperience = int(st.number_input("Enter Years Experience : "))
                    arr = [YearsExperience]
                    if st.button("Submit"):
                        pred = model(arr)
                        st.write(f"The prediciton is :{pred}")
                elif 'placement' in data:
                    cgpa = int(st.number_input("Enter cgpa :"))
                    placement_exam_marks = int(st.number_input("Enter placement exam marks :"))
                    arr = [cgpa,placement_exam_marks]
                    nparr = np.asarray(arr)
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(ts),random_state=int(rs))
                    lor = LogisticRegression()
                    lor.fit(x_train,y_train)
                    if st.button("Submit"):
                        pred = lor.predict(nparr)
                        st.write(f"The prediciton is :{pred}")
                else:
                    tv = int(st.number_input("Enter tv : "))
                    Radio = int(st.number_input("Enter Radio : "))
                    Newspaper = int(st.number_input("Enter Newspaper :"))
                    arr = [tv,Radio,Newspaper]
                    if st.button("Submit"):
                        pred = model(arr)
                        st.write(f"The prediciton is :{pred}")
            else:
                st.error("you skipped previous steps")
        except Exception:
               st.error("you skipped previous steps")         

