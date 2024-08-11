import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained KMeans model
@st.cache_resource
def load_model():
    return load('kmeans_model.joblib')

kmeans = load_model()

# Initialize LabelEncoder used during training
le = LabelEncoder()

# Define the features and categorical features
features_list = type float [
    "duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes",
    "wrongfragment", "hot", "loggedin", "numcompromised", "rootshell",
    "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles",
    "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate",
    "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate",
    "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate",
    "dsthostdiffsrvrate",  "dsthostsrvdiffhostrate","dsthostsamesrcportrate",
    "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate",
    "dsthostsrvrerrorrate", "attack", "lastflag"
]

categorical_features = ['protocoltype', 'service', 'flag', 'attack', 'lastflag']

# Sample feature encoding (use the same encoding used during training)
def preprocess_features(features):
    df = pd.DataFrame(features, columns=features_list)
    
    # Encode categorical features
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
    
    return df

def main():
    st.title('Network Traffic Cluster')

    
    
    with st.form("feature_form"):
        features = {}
        for feature in features_list:
            if feature in categorical_features:
                features[feature] = st.text_input(f"Enter {feature}")
            else:
                features[feature] = st.number_input(f"Enter {feature}", value=0.0)
        
        submitted = st.form_submit_button("Predict the Network Traffic Cluster")
        
        if submitted:
            # Convert the input features to DataFrame and preprocess
            input_df = preprocess_features([features])

            # Make prediction
            cluster_label = kmeans.predict(input_df)

            st.write(f'The network traffic belongs to cluster: {cluster_label[0]}')

if __name__ == '__main__':
    main()
