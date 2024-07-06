import streamlit as st
import converted_script as algorithms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

# Set option to suppress Matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Custom CSS
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

# Title
st.title("FAKE TWITTER ACCOUNT DETECTION")

# Initialize session state variables if they don't exist
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False
if "unknown_sample" not in st.session_state:
    st.session_state["unknown_sample"] = []

# Form
with st.form("Unknown Sample"):
    col1, col2 = st.columns(2)
    name = col1.text_input("Profile Name")
    default_profile = col2.selectbox("Is the Profile set as default?", ["", "Yes", "No"])
    frnds = col1.text_input("Friends Count")
    followers = col2.text_input("Followers Count")
    fav = col1.text_input("Favourites Count")
    status = col2.text_input("Status Count")
    default_profile_image = col1.selectbox("Is the Profile image set as default?", ["", "Yes", "No"])
    geo = col2.selectbox("Is the Profile set as geo-enabled?", ["", "Yes", "No"])
    pro_img = st.text_input("Profile image URL")
    verified = st.selectbox("Is the account verified?", ["", "Yes", "No"])
    avg_tweets = st.text_input("Average Tweets per Day")
    accnt_age = st.text_input("Account Age")
    a, b, c = st.columns(3)
    submit_state = b.form_submit_button("Submit")

# Handle form submission
if submit_state:
    if (
        name == "" or frnds == "" or followers == "" or fav == "" or default_profile == "" or
        default_profile_image == "" or geo == "" or status == "" or pro_img == "" or
        verified == "" or avg_tweets == "" or accnt_age == ""
    ):
        st.warning("Please fill all the fields above!")
    else:
        try:
            default_profile = default_profile == "Yes"
            default_profile_image = default_profile_image == "Yes"
            geo = geo == "Yes"
            verified = verified == "Yes"

            fav = int(fav)
            followers = int(followers)
            frnds = int(frnds)
            status = int(status)
            avg_tweets = float(avg_tweets)
            accnt_age = float(accnt_age)

            unknown_sample = [
                default_profile, default_profile_image, fav, followers, frnds,
                geo, status, verified, avg_tweets, accnt_age
            ]

            st.session_state["unknown_sample"] = np.array(unknown_sample).reshape(1, -1)
            st.session_state["form_submitted"] = True
            st.success("Submitted Successfully!")
        except ValueError as e:
            st.error(f"Error converting input to numeric values: {e}")

# Algorithm selection
compare = st.checkbox("Compare Algorithms")
if compare:
    algo = st.multiselect("Choose the Algorithm for prediction:", options=("Decision Trees", "Linear Regression", "k-Nearest Neighbours"))
    for a in algo:
        if a == "k-Nearest Neighbours":
            k = st.text_input("Enter the Number of neighbours:")
else:
    algo = st.selectbox("Choose the Algorithm for prediction:", options=("Decision Trees", "Logistic Regression", "k-Nearest Neighbours"))
    if algo == "k-Nearest Neighbours":
        k = st.text_input("Enter the Number of neighbours:")

# Prediction and Visualization
def plot_roc_curve(algo_name):
    y_scores = algo_name.predict_proba(algorithms.X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(algorithms.Y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve using Matplotlib
    plt.figure()
    plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Display the plot using Streamlit's st.pyplot()
    st.pyplot()

def print_result(prediction):
    if prediction[0] == 1:
        st.markdown("<h2>It's a fake account</h2>",unsafe_allow_html=True)
    else:
        st.markdown("<h2>It's not a fake account</h2>",unsafe_allow_html=True)

# Prediction button
if st.button("Start Prediction"):
    # st.write(f"Button pressed. form_submitted: {st.session_state['form_submitted']}")
    if st.session_state["form_submitted"]:
        st.write("Prediction Algorithm:", algo)
        if compare:
            st.write("Comparison mode enabled.")
        else:
            true_labels = [algorithms.Y_test] 
            unknown_sample = st.session_state["unknown_sample"]
            if algo == "Decision Trees":
                prediction = algorithms.dt_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.dt_model)
            elif algo == "Logistic Regression":
                prediction = algorithms.logistic_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.logistic_model)
            elif algo == "k-Nearest Neighbours":
                k = int(k)
                algorithms.knn_model = KNeighborsClassifier(n_neighbors=k)
                algorithms.knn_model.fit(algorithms.X_train, algorithms.Y_train)
                prediction = algorithms.knn_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.knn_model)
    else:
        st.write("Prediction not executed due to missing input or invalid state.")
