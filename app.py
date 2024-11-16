import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import sklearn as sk
print("streamlit",st.__version__)
print("pandas",pd.__version__)
print("numpy",np.__version__)
print("sklearn",sk.__version__)



st.set_page_config(page_title="Iris Classifier",
                    page_icon="🌷")

col3,col4 = st.columns(2)

with col3:
    st.image("./data/iris-Photoroom.png",width=210)
with col4:
    st.title("classify Iris flowers")
st.markdown("ML model to clasify iris flowers into setosa, versicolor, virginica ")

#sliders here 

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal Characteristics")
    sepal_L = st.slider("sepal length (cm)",1.0, 8.0, 0.5)
    sepal_W = st.slider("sepal width (cm)",2.0, 4.4, 0.5)
with col2:
     st.text("Petal Characteristics")
     petal_L = st.slider("petal length (cm)",1.0, 7.0, 0.5)
     petal_W = st.slider("petal width (cm)",0.1, 2.5, 0.5)

if st.button("Predict type of Iris"):
    prediction = predict(sepal_L, sepal_W, petal_L, petal_W) 
    st.write("🌺Predicted species: ",prediction[0])