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

# Sample feature encoding (use the same encoding used during training)
def preprocess_features(features):
    df = pd.DataFrame(features, columns=[
        "duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes",
        "wrongfragment", "hot", "loggedin", "numcompromised", "rootshell",
        "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles",
        "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate",
        "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate",
        "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate",
        "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate",
        "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate",
        "dsthostsrvrerrorrate", "attack", "lastflag"  # Added missing features
    ])
    
    # Encode categorical features
    categorical_features = ['protocoltype', 'service', 'flag', 'attack', 'lastflag']
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
    
    return df

def main():
    st.title('Network Traffic Clustering')

    st.write('Enter the feature values for network traffic:')

    # Create input fields for all features
    features = {}
    for feature in [
        "duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes",
        "wrongfragment", "hot", "loggedin", "numcompromised", "rootshell",
        "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles",
        "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate",
        "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate",
        "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate",
        "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate",
        "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate",
        "dsthostsrvrerrorrate", "attack", "lastflag"  # Added missing features
    ]:
        if feature in ["protocoltype", "service", "flag", "attack", "lastflag"]:
            features[feature] = st.text_input(f"Enter {feature}")
        else:
            features[feature] = st.number_input(f"Enter {feature}", value=0.0)

    if st.button('Predict Cluster'):
        # Preprocess features
        input_df = preprocess_features([features])

        # Make prediction
        cluster_label = kmeans.predict(input_df)

        st.write(f'The network traffic belongs to cluster: {cluster_label[0]}')

if __name__ == '__main__':
    main()