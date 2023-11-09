import streamlit as st
import pickle
import numpy as np

# Load the SVM model
svm_model = pickle.load(open('trained_wine_SVMclassification_model.sav', 'rb'))

# Function to classify wine quality using SVM model
def classify_quality(features):
    prediction = svm_model.predict(features)
    confidence = svm_model.decision_function(features)
    return prediction, confidence

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>🍷 Wine Quality Classifier 🍷</h1>", unsafe_allow_html=True)
    st.image('wine.jpg', use_column_width=True, caption="Image source: Google")  # Replace 'wine.jpg' with your image and add the source
    
    st.write("---")


    st.write(
        "Welcome to the Wine Quality Classifier! This app predicts the quality of wine based on its features. "
        "Simply provide the wine's characteristics, select its type, and let the model classify it for you."
    )

    # User input for wine features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=8.31, step=0.01)
        
    with col2:
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.65, step=0.001)
        
    with col3:
        citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.31, step=0.01)

    col4, col5, col6 = st.columns(3)
    
    with col4:
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=100.0, value=2.6, step=0.1)
        
    with col5:
        chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.065, step=0.001, format="%.3f")
        
    with col6:
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=13.0, step=1.0)

    col7, col8, col9 = st.columns(3)
    
    with col7:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=32.0, step=1.0)
        
    with col8:
        density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9947, step=0.0001, format="%.4f")
        
    with col9:
        pH = st.number_input("pH", min_value=0.0, max_value=7.0, value=3.16, step=0.01)

    col10, col11, col12 = st.columns(3)
    
    with col10:
        sulphates = st.number_input("Sulphates", min_value=0.0, max_value=3.0, value=0.68, step=0.01)

    with col11:
        alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

    with col12:
        wine_type_red = st.checkbox("Red Wine 🍷")
        wine_type_white = st.checkbox("White Wine 🥂")
        
        if wine_type_red and wine_type_white:
            wine_type_red = False
            wine_type_white = False

        if not wine_type_red and not wine_type_white:
            st.warning("Please select either Red Wine or White Wine.")
        
    # Predict wine quality when the user clicks the button
    if st.button('Predict Wine Quality'):
        # Check if either Red Wine or White Wine is selected
        if wine_type_red or wine_type_white:
            # Prepare the input features as a numpy array
            type_red = 1 if wine_type_red else 0
            type_white = 1 if wine_type_white else 0
        
            features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
                                type_red, type_white]).reshape(1, -1)

            # Predict wine quality using the SVM model
            prediction, confidence = classify_quality(features)
        
            if prediction[0] == 1:
                result = f"This wine has been classified as Legit"
                st.success(result)
            else:
                result = f"This wine has been classified as Fraud"
                st.error(result)
        else:
            st.warning("Please select either Red Wine or White Wine to make a prediction.")
    
    st.write("\n\n---\n\n")
    st.write(
        "**Data Source:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by "
        "data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009."
    )

if __name__ == '__main__':
    main()




