import streamlit as st

custom_css = """
<style>
    .stDeployButton{
        visibility: hidden;
    }
    #MainMenu{
        visibility: hidden;
    }
    #fake-twitter-account-detection {
        text-align: center;
        margin-top: -50px; 
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .stButton .st-emotion-cache-15hul6a{
        width: 200px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("FAKE TWITTER ACCOUNT DETECTION")
with st.form("Unknown Sample"):
    col1,col2 = st.columns(2)
    name = col1.text_input("Profile Name")
    frnds = col2.text_input("Friends Count")
    followers = col1.text_input("Followers Count")
    fav = col2.text_input("Favourites Count")
    status = col1.text_input("Status Count")
    loc = col2.text_input("Location")
    pro_img = st.text_input("Profile image URL")
    verified = st.text_input("Is the account verified? (Yes/No)")
    avg_tweets = st.text_input("Average Tweets per Day")
    accnt_age = st.text_input("Account Age")
    a,b,c = st.columns(3)
    submit_state = b.form_submit_button("Submit")
    if submit_state:
        if (name == "" or frnds =="" or followers =="" or fav =="" or loc =="" or status =="" or pro_img =="" or verified =="" or avg_tweets =="" or accnt_age =="") :
            st.warning("Please fill all the fields above !")
        else:
            st.success("Submitted Successfully !")

compare = st.checkbox("Compare Algorithms")
if compare:
    algo = st.multiselect("Choose the Algorithm for prediction: ", options=("Decision Trees","Linear Regression","k-Nearest Neighbours"))
    for a in algo:
        if a == "k-Nearest Neighbours":
            k = st.text_input("Enter the Number of neighbours: ")
else:
    algo = st.selectbox("Choose the Algorithm for prediction: ", options=("Decision Trees","Linear Regression","k-Nearest Neighbours"))
    if algo == "k-Nearest Neighbours":
        k = st.text_input("Enter the Number of neighbours: ")
predict = st.button("Start Prediction")



