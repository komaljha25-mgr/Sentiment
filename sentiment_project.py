import streamlit as st
import joblib

model=joblib.load('sentiment_model.pkl')

st.set_page_config(layout='wide')

st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #4aaaaa, #000000);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 35px;
    font-weight: bold;
    margin-bottom: 25px;
}
</style>

<div class="banner">
    Sentiment Analysis Project
</div>
""", unsafe_allow_html=True)

st.sidebar.image("AI_ML.jpg")

st.sidebar.title("About us")
st.sidebar.text("We are developing ML project based on NLP in LN AI Academy")

st.sidebar.title("About Project")
st.sidebar.text("This project predicts sentiment of given text....")

st.sidebar.title("Contact us")
st.sidebar.text("+91-9958966311")

sample_review=st.selectbox("Sample Review",options=['good food','quality was not good','awesome taste'])
if st.button("Predict",key="b1"):
    pred=model.predict([sample_review])
    prob=model.predict_proba([sample_review] )
    if pred[0]==0:
        st.error(f"Negative {prob[0][0]:.2f}")
    else:
        st.success(f"Positive {prob[0][1]:.2f}")
        st.balloons()
sample_review2=st.text_input("Your Review")
if st.button("Predict",key="b2"):
    pred=model.predict([sample_review2])
    prob=model.predict_proba([sample_review2] )
    if pred[0]==0:
        st.error(f"Negative {prob[0][0]:.2f}")
    else:
        st.success(f"Positive {prob[0][1]:.2f}")
        st.balloons()