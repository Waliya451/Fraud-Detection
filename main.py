import streamlit as st
import converted_script as algorithms

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
empty = 0
unknown_sample = []

st.markdown(custom_css, unsafe_allow_html=True)

st.title("FAKE TWITTER ACCOUNT DETECTION")
with st.form("Unknown Sample"):
    col1, col2 = st.columns(2)
    name = col1.text_input("Profile Name")
    default_profile = bool(col2.text_input("Is the Profile set as default?"))
    frnds = int(col1.text_input("Friends Count"))
    followers = int(col2.text_input("Followers Count"))
    fav = int(col1.text_input("Favourites Count"))
    status = int(col2.text_input("Status Count"))
    default_profile_image = bool(col1.text_input("Is the Profile image set as default?"))
    geo = bool(col2.text_input("Is the Profile set as geo-enabled?"))
    pro_img = st.text_input("Profile image URL")
    verified = bool(st.text_input("Is the account verified? (Yes/No)"))
    avg_tweets = float(st.text_input("Average Tweets per Day"))
    accnt_age = int(st.text_input("Account Age"))
    a, b, c = st.columns(3)
    submit_state = b.form_submit_button("Submit")

if submit_state:
    if (
        name == "" or frnds == "" or followers == "" or fav == "" or default_profile == "" or
        default_profile_image == "" or geo == "" or status == "" or pro_img == "" or
        verified == "" or avg_tweets == "" or accnt_age == ""
    ):
        st.warning("Please fill all the fields above!")
        empty = 1
    else:
        unknown_sample.extend([
            default_profile, default_profile_image, fav, followers, frnds, geo,
            status, verified, avg_tweets, accnt_age
        ])
        st.write("Unknown Sample:", unknown_sample)
        st.success("Submitted Successfully!")

compare = st.checkbox("Compare Algorithms")
if compare:
    algo = st.multiselect("Choose the Algorithm for prediction:", options=("Decision Trees", "Linear Regression", "k-Nearest Neighbours"))
    for a in algo:
        if a == "k-Nearest Neighbours":
            k = st.text_input("Enter the Number of neighbours:")
else:
    algo = st.selectbox("Choose the Algorithm for prediction:", options=("Decision Trees", "Linear Regression", "k-Nearest Neighbours"))
    if algo == "k-Nearest Neighbours":
        k = st.text_input("Enter the Number of neighbours:")

def make_prediction():
    if submit_state and empty == 0:
        st.write("Prediction Algorithm:", algo)
        if compare:
            pass
        else:
            if algo == "Decision Trees":
                prediction = algorithms.dt_model.predict([unknown_sample])
                st.write("Prediction Result:", prediction)
                if prediction[0] == 1:
                    st.markdown("It's a fake account")
                else:
                    st.markdown("It's not fake")

predict = st.button("Start Prediction",on_click=make_prediction)


