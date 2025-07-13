import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt

def load_data(uploaded_file=None):
    """Load data from either uploaded file or default locations"""
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded file")
            return data
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            return None
    
    # Try default locations
    default_paths = [
        'telco_churn.csv',
        'customer_churn_data.csv',
        'data/telco_churn.csv',
        'data/customer_churn_data.csv',
        os.path.expanduser('~/telco_churn.csv')
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                st.success(f"Found data at: {path}")
                return data
            except Exception as e:
                st.error(f"Error reading {path}: {str(e)}")
                continue
    
    return None

def validate_data(data):
    """Validate and clean data"""
    if data is None:
        return False
    
    # Normalize column names and fix typos
    data.columns = data.columns.str.strip().str.lower()
    data = data.rename(columns={
        'seniorctitzen': 'seniorcitizen',
        'phoneservice': 'phone_service',
        'multiplelines': 'multiple_lines'
    })
    
    required_columns = [
        'customerid', 'tenure', 'churn', 
        'monthlycharges', 'totalcharges'
    ]
    
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Available columns:", data.columns.tolist())
        return False
    
    # Convert numeric columns
    try:
        data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')
        data['monthlycharges'] = pd.to_numeric(data['monthlycharges'], errors='coerce')
        data['totalcharges'] = pd.to_numeric(data['totalcharges'], errors='coerce')
    except Exception as e:
        st.error(f"Data conversion error: {str(e)}")
        return False
    
    return True

def main():
    st.title("Customer Churn Analytics Dashboard")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload your customer churn data (CSV)", 
        type="csv"
    )
    
    # Load data
    data = load_data(uploaded_file)
    
    if data is None and uploaded_file is None:
        st.warning("""
            No data file found. Please either:
            1. Place your CSV file in one of these locations:
               - telco_churn.csv
               - customer_churn_data.csv
               - data/telco_churn.csv
               - data/customer_churn_data.csv
               - Your home directory (~/telco_churn.csv)
            2. Or upload it using the file uploader above
        """)
        st.stop()
    
    # Validate data
    if not validate_data(data):
        st.error("Data validation failed")
        st.stop()
    
    # Dashboard filters
    st.sidebar.header("Filters")
    
    # Tenure filter
    min_tenure = int(data['tenure'].min())
    max_tenure = int(data['tenure'].max())
    tenure_range = st.sidebar.slider(
        "Customer Tenure (months)",
        min_tenure,
        max_tenure,
        (min_tenure, max_tenure)
    )
    
    # Churn filter
    churn_options = data['churn'].unique()
    selected_churn = st.sidebar.multiselect(
        "Churn Status",
        options=churn_options,
        default=churn_options
    )
    
    # Apply filters
    filtered_data = data[
        (data['tenure'].between(*tenure_range)) & 
        (data['churn'].isin(selected_churn))
    ]
    
    # Display metrics
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(filtered_data))
    churn_rate = filtered_data['churn'].value_counts(normalize=True).get('Yes', 0)
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Avg Monthly Charge", f"${filtered_data['monthlycharges'].mean():.2f}")
    
    # Visualization
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    filtered_data['churn'].value_counts().plot(
        kind='bar',
        color=['#1f77b4', '#ff7f0e'],
        ax=ax
    )
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # Data preview
    st.header("Filtered Data Preview")
    st.dataframe(
        filtered_data.head(10),
        column_config={
            "monthlycharges": st.column_config.NumberColumn("Monthly Charge", format="$%.2f"),
            "totalcharges": st.column_config.NumberColumn("Total Charges", format="$%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()