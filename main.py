import streamlit as st
import converted_script as algorithms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

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
    algo = st.multiselect("Choose the Algorithm for prediction:", options=("Decision Trees", "Logistic Regression", "k-Nearest Neighbours"))
    print(algo)
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
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Display the plot using Streamlit's st.pyplot()
    st.pyplot()

def plot_precision_recall_curve(algorithm_name, y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.plot(recall, precision, label=f'{algorithm_name} (AP={avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

def print_result(prediction):
    if not compare:
        if prediction[0] == 1:
            st.markdown("<h1 style='text-align: center;color: red;'>IT\'S A FAKE ACCOUNT<br></h1>",unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center;color: green;'> IT\'S NOT A FAKE ACCOUNT<br></h1>",unsafe_allow_html=True)
    else:
        if prediction[0] == 1:
            st.markdown("<h4 style='text-align: center;color: red;'>IT\'S A FAKE ACCOUNT</h4>",unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='text-align: center;color: green;'> IT\'S NOT A FAKE ACCOUNT</h4>",unsafe_allow_html=True)

def compare_report(pred):
    target_names = ['Human', 'Bot']
    report = classification_report(algorithms.Y_test, pred, target_names=target_names, output_dict=True)

    st.markdown("<h5 style='text-align:center;'>Classification Report</h5>", unsafe_allow_html=True)
    st.write(pd.DataFrame(report))

    test_data_accuracy = accuracy_score(pred, algorithms.Y_test)
    st.write("The accuracy is " + str(test_data_accuracy * 100) + "%")
    confusion_matrix(algorithms.Y_test, pred)

def print_classification_report(test_pred):
    test_data_accuracy = accuracy_score(test_pred, algorithms.Y_test)
    st.write("The accuracy is " + str(test_data_accuracy * 100) + "%")
    confusion_matrix(algorithms.Y_test, test_pred)
    
    st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
    st.write(pd.DataFrame(confusion_matrix(algorithms.Y_test, test_pred)))

    target_names = ['Human', 'Bot']
    report = classification_report(algorithms.Y_test, test_pred, target_names=target_names, output_dict=True)

    st.markdown("<h3>Classification Report</h3>", unsafe_allow_html=True)
    st.write(pd.DataFrame(report))

def comparision(a):
    unknown_sample = st.session_state["unknown_sample"]
    if a == "Decision Trees":
        prediction = algorithms.dt_model.predict(unknown_sample)
        print_result(prediction)
        compare_report(algorithms.X_test_prediction3)
        y_scores = algorithms.dt_model.predict_proba(algorithms.X_test)[:, 1]
        plot_precision_recall_curve("Decision Trees", algorithms.Y_test, y_scores)
    elif a == "Logistic Regression":
        prediction = algorithms.logistic_model.predict(unknown_sample)
        print_result(prediction)
        compare_report(algorithms.X_test_prediction1)
        y_scores = algorithms.logistic_model.predict_proba(algorithms.X_test)[:, 1]
        plot_precision_recall_curve("Logistic Regression", algorithms.Y_test, y_scores)
    else:
        global k 
        k = int(k)
        algorithms.knn_model = KNeighborsClassifier(n_neighbors=k)
        algorithms.knn_model.fit(algorithms.X_train, algorithms.Y_train)
        X_test_prediction2 = algorithms.knn_model.predict(algorithms.X_test)
        prediction = algorithms.knn_model.predict(unknown_sample)
        print_result(prediction)
        compare_report(X_test_prediction2)
        y_scores = algorithms.knn_model.predict_proba(algorithms.X_test)[:, 1]
        plot_precision_recall_curve("k-Nearest Neighbours", algorithms.Y_test, y_scores)

# Prediction button
if st.button("Start Prediction"):
    # st.write(f"Button pressed. form_submitted: {st.session_state['form_submitted']}")
    if st.session_state["form_submitted"]:
        # st.write("Prediction Algorithm:", algo)
        if compare:
            if len(algo) == 1:
                st.warning("Please select more than one algorithm for comparision")
            else:
                plt.figure(figsize=(10, 6))
                if len(algo) == 2:
                    col1,col2 = st.columns(2)

                    # Left column (col1)
                    with col1:
                        algo_name = algo[0]
                        st.markdown(f"<div style='text-align: center; font-weight: bold;font-size: 20px'>{algo_name}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        comparision(algo_name)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Right column (col2)
                    with col2:
                        algo_name = algo[1]
                        st.markdown(f"<div style='text-align: center; font-weight: bold;font-size: 20px'>{algo_name}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        comparision(algo[1])
                        st.markdown("</div>", unsafe_allow_html=True)
                elif len(algo) == 3:
                    col1,col2,col3 = st.columns(3)

                    # Left column (col1)
                    with col1:
                        algo_name = algo[0]
                        st.markdown(f"<div style='text-align: center; font-weight: bold;font-size: 20px'>{algo_name}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        comparision(algo[0])
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Middle column (col2)
                    with col2:
                        algo_name = algo[1]
                        st.markdown(f"<div style='text-align: center; font-weight: bold;font-size: 20px;'>{algo_name}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        comparision(algo[1])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Right column (col3)
                    with col3:
                        algo_name = algo[2]
                        st.markdown(f"<div style='text-align: center; font-weight: bold;font-size: 20px'>{algo_name}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        comparision(algo[2])
                        st.markdown("</div>", unsafe_allow_html=True)
                st.pyplot()
                            
        else:
            true_labels = [algorithms.Y_test] 
            unknown_sample = st.session_state["unknown_sample"]
            if algo == "Decision Trees":
                prediction = algorithms.dt_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.dt_model)
                print_classification_report(algorithms.X_test_prediction3)
            elif algo == "Logistic Regression":
                prediction = algorithms.logistic_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.logistic_model)
                print_classification_report(algorithms.X_test_prediction1)
            elif algo == "k-Nearest Neighbours":
                k = int(k)
                algorithms.knn_model = KNeighborsClassifier(n_neighbors=k)
                algorithms.knn_model.fit(algorithms.X_train, algorithms.Y_train)
                X_test_prediction2 = algorithms.knn_model.predict(algorithms.X_test)
                prediction = algorithms.knn_model.predict(unknown_sample)
                print_result(prediction)
                plot_roc_curve(algorithms.knn_model)
                print_classification_report(X_test_prediction2)
    else:
        st.write("Prediction not executed due to missing input or invalid state.")
