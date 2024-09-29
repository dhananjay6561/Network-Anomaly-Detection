# Network Anomaly Detection

This project implements an advanced network anomaly detection system using K-Means clustering to identify and categorize unusual patterns in network traffic data. It combines machine learning techniques with cybersecurity principles to enhance network security monitoring and threat detection capabilities.

## Project Overview

The Network Anomaly Detection project aims to automatically identify and classify abnormal network behaviors that could indicate potential security threats, such as DDoS attacks, data exfiltration, or unauthorized access attempts. By leveraging unsupervised machine learning, specifically the K-Means clustering algorithm, this system can process large volumes of network traffic data and group similar patterns together, making it easier to spot outliers and anomalies.

Key features of the project include:

1. **Data Preprocessing**: Cleans and prepares raw network traffic data for analysis, including handling categorical features and normalizing numerical data.

2. **K-Means Clustering**: Applies the K-Means algorithm to segment network traffic into distinct clusters based on similarities in their features.

3. **Optimal Cluster Selection**: Utilizes the Elbow Method to determine the optimal number of clusters, balancing model complexity with explanatory power.

4. **Feature Importance Analysis**: Identifies the most influential features in distinguishing between different types of network traffic.

5. **Cluster Profiling**: Characterizes each cluster to understand what type of network behavior it represents (e.g., normal traffic, potential attacks, data exfiltration).

6. **Visualization**: Employs Principal Component Analysis (PCA) to create 2D visualizations of the high-dimensional network traffic data, aiding in the interpretation of clustering results.

7. **Real-time Prediction**: Includes a Streamlit web application that allows users to input new network traffic data and receive immediate predictions about which cluster (and thus, what type of behavior) it most closely matches.

The project consists of two main components:
1. A Jupyter notebook (`model.ipynb`) for data analysis, model training, and visualization.
2. A Streamlit web application (`webapp.py`) for real-time prediction using the trained model.

This system can be particularly useful for network administrators, cybersecurity professionals, and organizations looking to enhance their network monitoring capabilities. By automating the detection of anomalous network behaviors, it can help quickly identify potential security threats, allowing for faster response times and improved overall network security.

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (for environment management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dhananjay6561/network-anomaly-detection.git
   cd network-anomaly-detection
   ```

2. Create a Conda environment:
   ```bash
   conda create -n network-anomaly python=3.12
   conda activate network-anomaly
   ```

3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
   ```

## Usage

1. To train the model and generate visualizations, run the Jupyter notebook:
   ```bash
   jupyter notebook model.ipynb
   ```
   This notebook will guide you through the process of data preprocessing, model training, and result analysis.

2. To run the Streamlit web application for real-time predictions:
   ```bash
   streamlit run webapp.py
   ```
   This will launch a web interface where you can input network traffic features and receive cluster predictions.

## Contributing

We welcome contributions to improve the project! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

Potential areas for contribution include:
- Improving the clustering algorithm or trying other unsupervised learning techniques
- Enhancing the feature engineering process
- Expanding the web application with more detailed insights or visualizations
- Adding support for real-time network data ingestion

