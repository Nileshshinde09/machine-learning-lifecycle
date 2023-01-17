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
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(st.session_state['ts']),random_state=int(st.session_state['rs']))
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
    st.header("Step 5:Create Interface for your machine learning model")
    try:
        # with st.spinner('Creating interface for your model...'):
        #     time.sleep(5)
        data = st.session_state['data']
        df = pd.read_csv(data)
        if 'advertising' in data:
            y = df['Sales']
            x = df.drop(columns='Sales',axis = 1)
        elif 'placement' in data:
            y = df['placed']
            x = df.drop(columns='placed',axis = 1)
        elif 'Salary_Data' in data:
            y = df['Salary']
            x = df.drop(columns='Salary',axis = 1)
        if 'Salary_Data' in data:
            YearsExperience = int(st.number_input("Enter Years Experience : "))
            arr = [YearsExperience]
            if st.button("Submit"):
                pred = model(arr)
                st.header(f"The prediciton is :{pred[0]}")
        elif 'placement' in data:
            cgpa = int(st.number_input("Enter cgpa :"))
            placement_exam_marks = int(st.number_input("Enter placement exam marks :"))
            arr = [cgpa,placement_exam_marks]
            nparr = np.asarray(arr)
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=float(st.session_state['ts']),random_state=int(st.session_state['rs']))
            lor = LogisticRegression()
            lor.fit(x_train,y_train)
            if st.button("Submit"):
                pred = lor.predict([nparr])
                st.header(f"The prediciton is :{pred[0]}")
        else:
            tv = int(st.number_input("Enter tv : "))
            Radio = int(st.number_input("Enter Radio : "))
            Newspaper = int(st.number_input("Enter Newspaper :"))
            arr = [tv,Radio,Newspaper]
            if st.button("Submit"):
                pred = model(arr)
                st.header(f"The prediciton is :{pred[0]}")
    except Exception as e:
            st.error(f"you skipped previous steps{e}")         

