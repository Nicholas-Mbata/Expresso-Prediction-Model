import pandas as pd
import numpy as np
import streamlit as st

df= pd.read_csv(r"C:\\Users\\NMbata\\Downloads\\Expresso_churn_dataset (1).csv")

# Check for missing values
print(df.isnull().sum())

# Handle missing value in 'MRG' (impute with mode)
mode_mrg = df['MRG'].mode()[0]
df['MRG'].fillna(mode_mrg, inplace=True)

# Handle missing value in 'CHURN' (remove the row)
df.dropna(subset=['CHURN'], inplace=True)

# Impute numerical columns with the median
numerical_cols_to_impute_median = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'FREQ_TOP_PACK']
for col in numerical_cols_to_impute_median:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Create a missing value indicator for 'DATA_VOLUME'
df['DATA_VOLUME_MISSING'] = df['DATA_VOLUME'].isnull().astype(int)
# Impute 'DATA_VOLUME' with 0 (assuming missing means no data usage, this needs careful consideration)
df['DATA_VOLUME'].fillna(0, inplace=True)

# Consider dropping 'ZONE1' and 'ZONE2' due to high missing percentage
df.drop(columns=['ZONE1', 'ZONE2'], inplace=True)

# For 'ON_NET', 'ORANGE', 'TIGO', impute with 0 if missing means no calls to that network
cols_to_impute_zero = ['ON_NET', 'ORANGE', 'TIGO']
for col in cols_to_impute_zero:
    df[col].fillna(0, inplace=True)

# Impute 'REGION' with the mode
mode_region = df['REGION'].mode()[0]
df['REGION'].fillna(mode_region, inplace=True)

# Create a new category for missing values in 'TOP_PACK'
df['TOP_PACK'].fillna('No_Top_Pack', inplace=True)


# Check for missing values
df.isnull().sum()
#duplicates
# Check for duplicate rows
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify the removal of duplicates
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")
print(f"Shape of DataFrame after removing duplicates: {df.shape}")

#outliers


# Identify numerical columns (you might need to adjust this list based on your features)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Remove the target variable 'CHURN' and the missing indicator if present
if 'CHURN' in numerical_cols:
    numerical_cols.remove('CHURN')
if 'DATA_VOLUME_MISSING' in numerical_cols:
    numerical_cols.remove('DATA_VOLUME_MISSING')

# Function to cap outliers using IQR
def cap_outliers_iqr(series, lower_quantile=0.25, upper_quantile=0.75):
    Q1 = series.quantile(lower_quantile)
    Q3 = series.quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply outlier capping to selected numerical columns
cols_to_cap = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK', 'REGULARITY'] # Add other relevant numerical columns
for col in cols_to_cap:
    if col in df.columns:
        df[col] = cap_outliers_iqr(df[col])

# Identify categorical columns
categorical_cols = ['REGION', 'TENURE', 'MRG', 'TOP_PACK']

# Initialize LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column
for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])


print(df.head())

df.drop('user_id', axis=1, inplace=True)
# Define features (X) and target (y)
X = df.drop('CHURN', axis=1)
y = df['CHURN']

#Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

#Prediction for all Test Data
y_pred = model.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report # Import accuracy_score and classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save the trained model ---
import pickle

# Save the model to a file
model_filename = 'expresso_churn_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"\nTrained model saved as '{model_filename}'")

# --- Load the trained ML model ---
try:
    with open('expresso_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'expresso_churn_model.pkl' not found. Make sure it's in the same directory as this app.")
    st.stop()

st.title('EXPRESSO CHURN PREDICTION MODEL')
st.subheader('KEY IN THE FEATURES TO SEE THE PREDICTED OUTCOME')
tenure_options = ['K > 24 month', 'I 18-21 month', 'H 15-18 month', 'G 12-15 month',
                  'J 21-24 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month',
                  'C 1-3 month', 'B 0-1 month', 'A < 1 month']
tenure = st.selectbox('TENURE', tenure_options)

montant = st.number_input('MONTANT', value=0.0)
frequence_rech = st.number_input('FREQUENCE_RECH', value=0.0)
revenue = st.number_input('REVENUE', value=0.0)
arpu_segment = st.number_input('ARPU_SEGMENT', value=0.0)
frequence = st.number_input('FREQUENCE', value=0.0)
data_volume = st.number_input('DATA_VOLUME', value=0.0)
on_net = st.number_input('ON_NET', value=0.0)
orange = st.number_input('ORANGE', value=0.0)
tigo = st.number_input('TIGO', value=0.0)
regularity = st.number_input('REGULARITY', value=0.0)

top_pack_options = ['No_Top_Pack', 'other', 'Data C', 'All Net 500MB Day', ...] # Add all unique values from your training data
top_pack = st.selectbox('TOP_PACK', top_pack_options)

freq_top_pack = st.number_input('FREQ_TOP_PACK', value=0.0)

region_options = ['Dakar', 'Thiès', 'Saint-Louis', ...] # Add all unique region values
region = st.selectbox('REGION', region_options)

mrg_options = ['NO', 'YES'] # Add unique MRG values
mrg = st.selectbox('MRG', mrg_options)

# Add input fields for any other features your model uses
# Remember to handle the encoded categorical features correctly.
# For Label Encoded features, you'll need to map the user's selection back to the numerical label.

# --- Validation Button ---
if st.button('Predict Churn'):
    # --- Prepare the input data for the model ---
    input_data = pd.DataFrame({
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'REGULARITY': [regularity],
        'FREQ_TOP_PACK': [freq_top_pack],
        'TENURE': [tenure],
        'TOP_PACK': [top_pack],
        'REGION': [region],
        'MRG': [mrg],
        # Add other features here in the same order as they were during training
        'DATA_VOLUME_MISSING': [0], # Assuming you'll handle this based on user input if needed
        # ... Add the encoded columns here. You'll need to handle the encoding logic.
    })

    # --- Preprocess the input data to match the training data format ---
    # If you used Label Encoding, you'll need to perform the same encoding here.
    # You might need to load the LabelEncoder objects you used during training.

    # Example of manual label encoding (you might need to adjust based on your encoder's fit)
    label_encoder_tenure = LabelEncoder() # You might need to refit this with all possible values
    input_data['TENURE'] = label_encoder_tenure.fit_transform(input_data['TENURE'])

    label_encoder_top_pack = LabelEncoder()
    input_data['TOP_PACK'] = label_encoder_top_pack.fit_transform(input_data['TOP_PACK'])

    label_encoder_region = LabelEncoder()
    input_data['REGION'] = label_encoder_region.fit_transform(input_data['REGION'])

    label_encoder_mrg = LabelEncoder()
    input_data['MRG'] = label_encoder_mrg.fit_transform(input_data['MRG'])


    # Ensure the order of columns in input_data matches the order during training
    # Get the feature names the model was trained on (excluding the target)
    feature_names = list(X_train.columns)
    input_data = input_data[feature_names]


    # --- Make prediction ---
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1] # Probability of churn

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning(f'This customer is likely to churn (Probability: {probability[0]:.2f})')
    else:
        st.success(f'This customer is unlikely to churn (Probability: {probability[0]:.2f})')
