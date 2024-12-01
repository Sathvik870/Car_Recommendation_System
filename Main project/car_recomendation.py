import os
import spacy
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Sentence-BERT model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model for syntax parsing
nlp = spacy.load('en_core_web_sm')

# Function to load car descriptions from individual text files
def load_car_descriptions(directory_path):
    car_descriptions = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            car_name = os.path.splitext(filename)[0].replace('_', ' ').title()  # Extract car name from filename
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:  # Specify encoding
                description = file.read().strip()
                car_descriptions[car_name] = description
    return car_descriptions

# Function to load car categories from a separate text file
def load_car_categories(file_path):
    car_categories = {}
    with open(file_path, 'r') as file:
        for line in file:
            car, category = line.strip().split(': ', 1)
            car_categories[car] = category
    return car_categories

# Function to extract attributes (fuel type, seating capacity, car category) from user input
def extract_query_attributes(user_input):
    doc = nlp(user_input.lower())
    
    fuel_type = None
    seating_capacity = None
    car_category = None
    
    # Define keywords for syntax-based extraction
    fuel_keywords = {"petrol", "diesel", "electric", "hybrid"}
    category_keywords = {"suv", "sedan", "hatchback", "truck", "ev"}
    
    # Iterate over tokens to extract fuel type, seating capacity, and car category
    for token in doc:
        if token.text in fuel_keywords:
            fuel_type = token.text.title()  # e.g., Petrol, Diesel
        if token.like_num and token.text.isdigit():
            num = int(token.text)
            if 2 < num < 8:  # Seating capacity range (2-7)
                seating_capacity = num
        if token.text in category_keywords:
            car_category = token.text.title()  # e.g., SUV, Sedan
    
    return fuel_type, seating_capacity, car_category

# Function to split car descriptions into sections
def split_description(description):
    sections = {
        'performance': None,
        'comfort': None,
        'economy': None
    }
    
    # Example of splitting by keywords (extend with more sophisticated methods if needed)
    description_lower = description.lower()
    if 'performance' in description_lower:
        sections['performance'] = description
    if 'comfort' in description_lower:
        sections['comfort'] = description
    if 'economy' in description_lower:
        sections['economy'] = description

    return sections

# Function to recommend cars based on user input and car descriptions
def recommend_car(user_input, car_descriptions, car_categories):
    # Step 1: Extract query attributes (fuel type, seating, category)
    fuel_type, seating_capacity, car_category = extract_query_attributes(user_input)
    
    print(f"Extracted Query Attributes: Fuel Type={fuel_type}, Seating={seating_capacity}, Category={car_category}")
    
    # Step 2: Filter cars based on extracted attributes
    filtered_cars = {}
    for car, desc in car_descriptions.items():
        if car not in car_categories:
            continue
        
        category_info = car_categories[car].split(', ')
        
        # Assuming category_info structure: [category, fuel_type, seating_capacity]
        if len(category_info) < 3:
            continue
        
        car_type = category_info[0]  # Extract car category (SUV, Sedan, etc.)
        car_fuel_type = category_info[1]  # Extract fuel type
        car_seating_capacity = int(category_info[2])  # Extract seating capacity
        
        # Print out the car details for debugging
        print(f"Car: {car}, Type: {car_type}, Fuel: {car_fuel_type}, Seats: {car_seating_capacity}")
        
        # Apply filtering based on syntax analysis of user input
        if (fuel_type is None or car_fuel_type.lower() == fuel_type.lower()) and \
           (seating_capacity is None or car_seating_capacity == seating_capacity) and \
           (car_category is None or car_type.lower() == car_category.lower()):
            filtered_cars[car] = desc

    print(f"Filtered Cars: {filtered_cars}")

    # Return no results if no cars match the user's input constraints
    if not filtered_cars:
        return [("No matches found", 0)]
    
    # Step 3: Split user query into sections and calculate embeddings
    user_sections = split_description(user_input)
    query_embeddings = {key: model.encode(value, convert_to_tensor=True) if value else None
                        for key, value in user_sections.items()}
    
    # Step 4: Compute similarity with filtered car descriptions
    similarities = {}
    for car, description in filtered_cars.items():
        car_sections = split_description(description)
        total_similarity = 0
        section_count = 0
        
        # Compute similarity for each section (e.g., performance, comfort, economy)
        for section, query_embedding in query_embeddings.items():
            car_section = car_sections[section]
            if query_embedding is not None and car_section:
                car_embedding = model.encode(car_section, convert_to_tensor=True)
                similarity = cosine_similarity([query_embedding], [car_embedding])[0][0]
                total_similarity += similarity
                section_count += 1
        
        # Average the similarity across sections
        if section_count > 0:
            similarities[car] = total_similarity / section_count

    print(f"Similarities: {similarities}")
    
    # Step 5: Rank cars by similarity
    recommended_cars = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return recommended_cars[:3]  # Return top 3 cars
