import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your dataset
data = pd.read_csv('D:\\NLP Project\\prediction_dataset_numeric.csv')  # Replace with your dataset path

# Display the head of the dataset
print("Dataset Head:")
print(data.head())

# Preprocessing
# Encode categorical variables if necessary
label_encoders = {}
for column in ['Car Model']:  # Add other categorical columns if necessary
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
# Features and target variable
X = data.drop(columns=['Car Model'])  # Replace with your target variable
y = data['Car Model']  # Assuming Car Model is your target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# Prediction Function
def predict_car_model(fuel_type, seat_capacity, car_category, cost_lakhs, engine_cc_kwh, milege_pkm_pc):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Fuel Type': [fuel_type],
        'Seating Capacity': [seat_capacity],
        'Car Category': [car_category],
        'Cost_lakhs': [cost_lakhs],
        'Engine_cc_kwh': [engine_cc_kwh],
        'Milege_pkm_pc': [milege_pkm_pc]
    })
    prediction = rf_classifier.predict(input_data)

    # Reverse label encoding to get original car model
    predicted_car_model = label_encoders['Car Model'].inverse_transform(prediction)
    return predicted_car_model  # Return the predicted car model

# Example usage
if __name__ == "__main__":
    # User input (replace these with actual user inputs)
    user_fuel_type = 1  # Input fuel type
    user_seat_capacity = 0      # Input seat capacity
    user_car_category = 0  # Input car category           # Input year
    user_cost_lakhs = 0        # Input cost in lakhs
    user_engine_cc_kwh = 0  # Input engine capacity
    user_milege_pkm_pc = 0    # Input mileage

    predicted_model = predict_car_model(user_fuel_type, user_seat_capacity, user_car_category, user_cost_lakhs, user_engine_cc_kwh, user_milege_pkm_pc)
    print(f"The predicted car model is: {predicted_model}")


    '''
    data = pd.read_csv('D:\\NLP Project\\car_data_final.csv')  # Replace with your dataset path
    # Preprocessing
    # Encode categorical variables if necessary
    label_encoders = {}
    for column in ['Car Model', 'Fuel Type', 'Car Category']:  # Add other categorical columns if necessary
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Features and target variable
    X = data.drop(columns=['Car Model'])  # Replace with your target variable
    y = data['Car Model']  # Assuming Car Model is your target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    # Check for null values and replace with 0 if necessary
    fuel_type = fuel_type if fuel_type is not None else '0'  # assuming fuel type cannot be numeric
    seating_capacity = seating_capacity if seating_capacity is not None else 0
    car_category = car_category if car_category is not None else '0'  # assuming car category cannot be numeric
    cost = cost if cost is not None else 0
    engine_size = engine_size if engine_size is not None else 0
    mileage = mileage if mileage is not None else 0
    # Step 2: Filter cars based on extracted attributes
    input_data = pd.DataFrame({
        'Fuel Type': [fuel_type],
        'Seating Capacity': [seating_capacity],
        'Car Category': [car_category],
        'Cost_lakhs': [cost],
        'Engine_cc_kwh': [engine_size],
        'Milege_pkm_pc': [mileage]
    })
    label_encoders = {}
    for column in ['Fuel Type', 'Car Category']:  # Add more columns as necessary
        le = LabelEncoder()
        input_data[column] = le.fit_transform(input_data[column])
        label_encoders[column] = le
        
    prediction = rf_classifier.predict(input_data)

    # Reverse label encoding to get original car model
    predicted_car_model = label_encoders['Car Model'].inverse_transform(prediction),predicted_car_model[0]'''
