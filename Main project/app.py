import pandas as pd
import spacy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load car dataset from CSV
def load_car_dataset(file_path):
    return pd.read_csv(file_path)
def chunk_relationship(user_input):
    # Process the user input using spaCy
    doc = nlp(user_input)
    # Extract the chunk relationships
    chunk_relationships = []
    for chunk in doc.noun_chunks:
        print(f"Chunk: {chunk.text}, Root: {chunk.root.text}, Root Dep: {chunk.root.dep_}, Head Text: {chunk.root.head.text}")
# Function to extract attributes from user input
def extract_query_attributes(user_input):
    fuel_type = None
    seating_capacity = None
    car_category = None
    cost = None
    engine_size = None
    mileage = None
    
    # Initialize comparison operators
    comparison_operators = {
        "greaterthan": "gt",
        "greater":"gt",
        "lessthan": "lt",
        "less": "lt",
        "lesser":"lt",
        "equalto": "eq",
        "equal" : "eq",
        "greaterthanorequalto": "gte",
        "lessthanorequalto": "lte",
        "above": "gt",
        "more" : "gt",
        "below": "lt",
        "minimum": "gte",
        "maximum": "lte"
    }
    # Define keywords for syntax-based extraction
    fuel_keywords = {"petrol", "diesel", "electric", "hybrid"}
    category_keywords = {"suv", "sedan", "hatchback", "truck", "ev", "mpv"}
    
    # Keywords for cost, engine size, and mileage
    cost_keywords = {"cost", "lakhs", "rate", "costs"}
    engine_keywords = {"engine", "cc", "kwh", }
    mileage_keywords = {"mileage", "pkm", "kmpl", "kilometer","km"}
    
    # Split the input by commas and process each segment individually
    segments = user_input.split(',')

    # Iterate over each segment
    for segment in segments:
        print("Checking segment:", segment)
        
        # Process the individual segment
        doc = nlp(segment.strip())
        
        print("Tokens and their POS tags:")
        for token in doc:
            print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}")
        # Check and extract relevant information from each segment
        for token in doc:
            # Extract fuel type
            if token.text.lower() in fuel_keywords:
                fuel_type = token.text.title()  # Capitalize the first letter
            
            # Extract seating capacity if token is a number
            if token.pos_ == "NUM":
                num = int(token.text)
                if 2 < num <= 7:  # Adjusted for seating capacity (3 to 7 seats)
                    seating_capacity = num
            
            # Extract car category
            if token.text.lower() in category_keywords:
                car_category = token.text.title()  # Capitalize the first letter
            
            # Check for cost-related keywords in the same segment
            if token.text.lower() in cost_keywords:
                for previous_token in doc[max(token.i - 1, 0):token.i]:
                    if previous_token.pos_ == "NUM":
                        cost = float(previous_token.text)
                        break  # Stop further checks for numbers in this segment
            
            # Check for engine-related keywords in the same segment
            if token.text.lower() in engine_keywords:
                for previous_token in doc[max(token.i - 1, 0):token.i]:
                    if previous_token.pos_ == "NUM":
                        engine_size = float(previous_token.text)
                        break  # Stop further checks for numbers in this segment

            # Check for mileage-related keywords in the same segment
            if token.text.lower() in mileage_keywords:
                for previous_token in doc[max(token.i - 1, 0):token.i]:
                    if previous_token.pos_ == "NUM":
                        mileage = float(previous_token.text)
                        break  # Stop further checks for numbers in this segment
        
        # Check comparison operators in each segment and apply to cost, engine size, mileage
        for token in doc:
            print(token)
            if token.text.lower() in comparison_operators:
                comparison = comparison_operators[token.text.lower()]
                print("Comparison found:", comparison)
                
                # Apply comparison operator to cost
                if any(keyword in segment.lower() for keyword in cost_keywords) and cost is not None:
                    if comparison == "lt":
                        cost = f"< {cost}"
                    elif comparison == "gt":
                        cost = f"> {cost}"
                    elif comparison == "lte":
                        cost = f"<= {cost}"
                    elif comparison == "gte":
                        cost = f">= {cost}"
                    elif comparison == "eq":
                        cost = f"= {cost}"

                # Apply comparison operator to engine size
                if any(keyword in segment.lower() for keyword in engine_keywords) and engine_size is not None:
                    if comparison == "lt":
                        engine_size = f"< {engine_size}"
                    elif comparison == "gt":
                        engine_size = f"> {engine_size}"
                    elif comparison == "lte":
                        engine_size = f"<= {engine_size}"
                    elif comparison == "gte":
                        engine_size = f">= {engine_size}"
                    elif comparison == "eq":
                        engine_size = f"= {engine_size}"

                # Apply comparison operator to mileage
                if any(keyword in segment.lower() for keyword in mileage_keywords) and mileage is not None:
                    if comparison == "lt":
                        mileage = f"< {mileage}"
                    elif comparison == "gt":
                        mileage = f"> {mileage}"
                    elif comparison == "lte":
                        mileage = f"<= {mileage}"
                    elif comparison == "gte":
                        mileage = f">= {mileage}"
                    elif comparison == "eq":
                        mileage = f"= {mileage}"

        print("Processed Segment Results:", fuel_type, seating_capacity, car_category, cost, engine_size, mileage)

    return fuel_type, seating_capacity, car_category, cost, engine_size, mileage

def remove_logical_symbols(value):
    # Use regex to remove any >, <, =, etc.
    if isinstance(value, str):
        value = re.sub(r'[><=]', '', value).strip()
    # Convert to float if possible
    return float(value) if value else 0

# Function to recommend cars based on user input
def recommend_car(user_input, car_data):
    # Step 1: Extract query attributes
    fuel_type, seating_capacity, car_category, cost, engine_size, mileage = extract_query_attributes(user_input)
    print(f"\nExtracted Query Attributes: Fuel Type={fuel_type}, Seating={seating_capacity}, Category={car_category}, Cost={cost}, Engine Size={engine_size}, Mileage={mileage}")
    filtered_cars = car_data
    chunk_relationship(user_input)
    # Standardizing the column names to lowercase without spaces
    filtered_cars.columns = filtered_cars.columns.str.strip().str.lower()
    columns_to_lowercase = ['car model', 'fuel type', 'car category']
    for col in columns_to_lowercase:
        if col in filtered_cars.columns:
            filtered_cars[col] = filtered_cars[col].str.strip().str.lower()
    print("Step 2: Standardized Column Names")
    print(filtered_cars)

    if fuel_type=='petrol':
        fuel_type_numeric=1
    elif fuel_type=='diesel':
        fuel_type_numeric=2
    elif fuel_type=='electric':
        fuel_type_numeric=3
    else:
        fuel_type_numeric=0
        
    if car_category=='hatchback':
        car_category_numeric=1
    elif car_category=='sedan':
        car_category_numeric=2
    elif car_category=='suv':
        car_category_numeric=3
    elif car_category=='mpv':
        car_category_numeric=4
    else:
        car_category_numeric=0
    # Filtering by fuel type, seating capacity, car category, cost, engine size, and mileage
    if fuel_type and 'fuel type' in filtered_cars.columns:
        filtered_cars = filtered_cars[filtered_cars['fuel type'] == fuel_type.lower()]
        print("Step 3: Filtered by Fuel Type")
        print(filtered_cars)
    if seating_capacity and 'seating capacity' in filtered_cars.columns:
        filtered_cars = filtered_cars[filtered_cars['seating capacity'] == seating_capacity]
        print("Step 4: Filtered by Seating Capacity")
        print(filtered_cars)
    if car_category and 'car category' in filtered_cars.columns:
        filtered_cars = filtered_cars[filtered_cars['car category'] == car_category.lower()]
        print("Step 5: Filtered by Car Category")
        print(filtered_cars)
    
    # Filtering based on cost, engine size, and mileage
    if cost and 'cost_lakhs' in filtered_cars.columns:
        if '<' in str(cost):
            limit = float(cost.split('<')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['cost_lakhs'] < limit]
        elif '>' in str(cost):
            limit = float(cost.split('>')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['cost_lakhs'] > limit]
        elif '<=' in str(cost):
            limit = float(cost.split('<=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['cost_lakhs'] <= limit]
        elif '>=' in str(cost):
            limit = float(cost.split('>=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['cost_lakhs'] >= limit]
        elif '=' in str(cost):
            limit = float(cost.split('=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['cost_lakhs'] == limit]
        print("Step 6: Filtered by Cost")
        print(filtered_cars)
        
    if engine_size and 'engine_cc_kwh' in filtered_cars.columns:
        if '<' in str(engine_size):
            limit = float(engine_size.split('<')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['engine_cc_kwh'] < limit]
        elif '>' in str(engine_size):
            limit = float(engine_size.split('>')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['engine_cc_kwh'] > limit]
        elif '<=' in str(engine_size):
            limit = float(engine_size.split('<=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['engine_cc_kwh'] <= limit]
        elif '>=' in str(engine_size):
            limit = float(engine_size.split('>=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['engine_cc_kwh'] >= limit]
        elif '=' in str(engine_size):
            limit = float(engine_size.split('=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['engine_cc_kwh'] == limit]
        print("Step 7: Filtered by Engine Size")
        print(filtered_cars)
        
    if mileage and 'milege_pkm_pc' in filtered_cars.columns:
        if '<' in str(mileage):
            limit = float(mileage.split('<')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['milege_pkm_pc'] < limit]
        elif '>' in str(mileage):
            limit = float(mileage.split('>')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['milege_pkm_pc'] > limit]
        elif '<=' in str(mileage):
            limit = float(mileage.split('<=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['milege_pkm_pc'] <= limit]
        elif '>=' in str(mileage):
            limit = float(mileage.split('>=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['milege_pkm_pc'] >= limit]
        elif '=' in str(mileage):
            limit = float(mileage.split('=')[1].strip())
            filtered_cars = filtered_cars[filtered_cars['milege_pkm_pc'] == limit]
        print("Step 8: Filtered by Mileage")
        print(filtered_cars)
    if filtered_cars.empty:
        print("No matches found.")
        return ["No matches found"]
    recommendations = filtered_cars['car model'].tolist()
    print("Final Recommendations:")
    print(recommendations)
    data = pd.read_csv('D:\\NLP Project\\prediction_dataset_numeric.csv')  
    label_encoders = {}
    for column in ['Car Model']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    # Features and target variable
    X = data.drop(columns=['Car Model'])
    y = data['Car Model'] 
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    fuel_type_numeric = fuel_type_numeric if fuel_type_numeric is not None else 0  # assuming fuel type cannot be numeric
    seating_capacity = seating_capacity if seating_capacity is not None else 0
    car_category_numeric = car_category_numeric if car_category_numeric is not None else 0  # assuming car category cannot be numeric
    cost = remove_logical_symbols(cost) if cost is not None else 0
    engine_size = remove_logical_symbols(engine_size) if engine_size is not None else 0
    mileage = remove_logical_symbols(mileage) if mileage is not None else 0
    input_data = pd.DataFrame({
        'Fuel Type': [fuel_type_numeric],
        'Seating Capacity': [seating_capacity],
        'Car Category': [car_category_numeric],
        'Cost_lakhs': [cost],
        'Engine_cc_kwh': [engine_size],
        'Milege_pkm_pc': [mileage]
    })
    prediction = rf_classifier.predict(input_data)
    # Reverse label encoding to get original car model
    predicted_car_model = label_encoders['Car Model'].inverse_transform(prediction)
    print("As per classification model : ",predicted_car_model[0])
    predicted_car_model=predicted_car_model[0]
    return recommendations,predicted_car_model

def car_recommendation_page():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(90deg, #00b4d8, #48cae4);
        }
        header {
            background-color: black;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 2em;
        }
        div.stButton > button {
            background-color: black;
            color: white;
            border: 2px solid white;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <header>Car Recommendation System</header>
    """, unsafe_allow_html=True)

    user_input = st.text_area("Describe your car preferences (e.g., 'I want a petrol SUV with 5 seats, cost less than 10 lakhs, engine less than 1500 cc, and mileage above 20 pkm')")

    # Load car dataset
    car_data = load_car_dataset("D:\\NLP Project\\car_data_final.csv")

    if st.button("Recommend"):
        if user_input:
            recommended_cars,predictedone = recommend_car(user_input, car_data)
            st.write("Recommended Car (using random forest classifier model) : "+predictedone)
            st.write("Recommended Cars:")
            for car in recommended_cars:
                st.write("- " + car)
        else:
            st.warning("Please enter your car preferences.")

    # Button to go back to the home screen
    if st.button("Back to Home"):
        st.session_state.page = "home"
def home_page():
    st.markdown("""
        <style>
        .container {
            display: flex;
            flex-direction: column; /* Align items vertically */
            align-items: center; /* Center horizontally */
            height: 100vh;
            justify-content: flex-start; /* Align items at the top */
            background: linear-gradient(90deg, #00b4d8, #48cae4);
        }
        .car-images {
            display: flex;
            justify-content: space-around; /* Space images evenly */
            width: 100%; /* Full width */
            margin-bottom: 20px; /* Space below images */
        }
        .center-text {
            font-size: 2.5em;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 20px; /* Space below the title */
        }
        .button {
            padding: 10px;
            background-color: black;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1.2em; /* Font size for the button */
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="car-images">'
                '<img src="https://via.placeholder.com/150" width="150" height="150">'
                '<img src="https://via.placeholder.com/150" width="150" height="150">'
                '<img src="https://via.placeholder.com/150" width="150" height="150">'
                '<img src="https://via.placeholder.com/150" width="150" height="150">'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="center-text">Car Recommendation System</div>', unsafe_allow_html=True)

    # Button to go to car recommendation system
    if st.button("Go to Car Recommendation System", key='home_button'):
        st.session_state.page = "recommendation"


def main():
    # Initialize session state
    if "page" not in st.session_state:
        car_recommendation_page()
    elif st.session_state.page == "recommendation":
        car_recommendation_page()

# Run the app
if __name__ == "__main__":
    main()