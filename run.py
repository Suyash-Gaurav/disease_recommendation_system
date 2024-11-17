import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import shutil
from typing import Dict, List, Set, Tuple
import warnings
from functools import lru_cache

warnings.filterwarnings('ignore')

class DiseasePredictionSystem:
    def __init__(self, data_path: str = './'):
        """Initialize the system with data files from the specified path."""
        self.data_path = data_path
        self.load_data()
        self.setup_model()
        self.terminal_width = self._get_terminal_size()

    def load_data(self) -> None:
        """Load and preprocess all required data files."""
        # Load all data files at once using a dictionary comprehension
        data_files = {
            'training': 'Training.csv',
            'medications': 'medications.csv',
            'description': 'description.csv',
            'precautions': 'precautions_df.csv',
            'workout': 'workout_df.csv',
            'diets': 'diets.csv'
        }
        
        try:
            self.dataframes = {
                key: pd.read_csv(os.path.join(self.data_path, filename))
                for key, filename in data_files.items()
            }
            
            # Extract symptoms list once
            self.symptoms = list(self.dataframes['training'].columns[:-1])
            self.symptoms_set = set(self.symptoms)  # For O(1) lookup
            
            # Preprocess disease data into a more efficient structure
            self._prepare_disease_data()
            
        except FileNotFoundError as e:
            raise Exception(f"Required data files not found in {self.data_path}: {str(e)}")

    def _prepare_disease_data(self) -> None:
        """Prepare disease data in an optimized format."""
        self.disease_data = {}
        
        for idx, row in self.dataframes['medications'].iterrows():
            disease = row['Disease']
            self.disease_data[disease] = {
                'medication': row['Medication'],
                'description': self.dataframes['description'].get('Description', {}).get(idx, "No description available"),
                'precautions': [
                    self.dataframes['precautions'].get(f'Precaution_{i}', {}).get(idx, "N/A")
                    for i in range(1, 5)
                ],
                'workout': self.dataframes['workout'].get('workout', {}).get(idx, "No workout information available"),
                'diet': self.dataframes['diets'].get('Diet', {}).get(idx, "No diet information available")
            }

    def setup_model(self) -> None:
        """Load and setup the prediction model."""
        try:
            with open(os.path.join(self.data_path, 'forest_model.pkl'), 'rb') as file:
                self.model = pickle.load(file)
            
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.dataframes['training']['prognosis'])
            
        except FileNotFoundError:
            raise Exception("Model file not found. Ensure forest_model.pkl exists in the data directory.")

    @staticmethod
    def _get_terminal_size() -> int:
        """Get terminal width with fallback value."""
        try:
            return shutil.get_terminal_size().columns
        except:
            return 80

    @lru_cache(maxsize=128)
    def _format_symptom(self, idx: int, symptom: str, col_width: int) -> str:
        """Format symptom string with caching for repeated lookups."""
        formatted = f"{idx:2d}. {symptom.replace('_', ' ').capitalize()}"
        return formatted if len(formatted) <= col_width else formatted[:col_width-3] + "..."

    def display_symptoms(self) -> Dict[str, int]:
        """Display symptoms and get user input in an optimized format."""
        # Calculate layout
        cols = 4 if self.terminal_width >= 120 else 3 if self.terminal_width >= 90 else 2
        col_width = min(35, self.terminal_width // cols - 2)
        rows = (len(self.symptoms) + cols - 1) // cols

        # Print header
        print(f"\n{'Select Your Symptoms'.center(self.terminal_width, '=')}")

        # Display symptoms in columns
        for row in range(rows):
            row_symptoms = []
            for col in range(cols):
                idx = row + (col * rows)
                if idx < len(self.symptoms):
                    formatted = self._format_symptom(idx + 1, self.symptoms[idx], col_width)
                    row_symptoms.append(formatted.ljust(col_width))
            print("".join(row_symptoms))

        print("=" * self.terminal_width)

        while True:
            try:
                selected = self._get_user_input()
                if not selected:
                    continue
                    
                # Convert to symptom input dict efficiently
                return {symptom: 1 if i in selected else 0 
                       for i, symptom in enumerate(self.symptoms)}
                       
            except ValueError as e:
                print(f"Error: {str(e)}")

    def _get_user_input(self) -> Set[int]:
        """Get and validate user input."""
        selected_numbers = input("\nEnter symptom numbers (comma-separated): ").strip()
        
        if selected_numbers.lower() == 'q':
            raise KeyboardInterrupt
            
        # Process all numbers at once
        try:
            selected = {int(num.strip()) - 1 for num in selected_numbers.split(",") 
                       if num.strip().isdigit()}
            
            # Validate range
            if not all(0 <= num < len(self.symptoms) for num in selected):
                raise ValueError("Invalid symptom numbers selected.")
                
            if not selected:
                raise ValueError("Please select at least one symptom.")
                
            # Show selections
            print("\nSelected Symptoms:")
            for idx in selected:
                print(f"- {self.symptoms[idx].replace('_', ' ').capitalize()}")
                
            if input("\nConfirm these selections? (Y/N): ").lower().startswith('y'):
                return selected
                
        except ValueError as e:
            raise ValueError("Invalid input. Please enter numbers separated by commas.")
        
        return set()

    def predict_diseases(self, symptom_input: Dict[str, int], top_n: int = 3) -> List[Dict]:
        """Predict diseases and get recommendations efficiently."""
        # Convert input to array format expected by model
        input_array = np.array([list(symptom_input.values())], dtype=np.int8)
        
        # Get predictions
        probabilities = self.model.predict_proba(input_array)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_diseases = self.label_encoder.inverse_transform(top_indices)
        
        # Prepare recommendations
        return [
            {
                "rank": i + 1,
                "disease": disease,
                "confidence": f"{probabilities[idx]:.2%}",
                **self.disease_data[disease]
            }
            for i, (disease, idx) in enumerate(zip(top_diseases, top_indices))
        ]

    @staticmethod
    def display_recommendations(recommendations: List[Dict]) -> None:
        """Display recommendations in a clean format."""
        print("\n" + "=" * 80)
        print("Disease Predictions and Recommendations:".center(80))
        print("=" * 80 + "\n")
        
        for rec in recommendations:
            print(f"Rank {rec['rank']}: {rec['disease']}")
            print(f"Confidence: {rec['confidence']}")
            
            sections = [
                ("Description", 'description'),
                ("Recommended Medication", 'medication'),
                ("Precautions", 'precautions'),
                ("Recommended Workout", 'workout'),
                ("Dietary Recommendations", 'diet')
            ]
            
            for title, key in sections:
                print(f"\n{title}:")
                if isinstance(rec[key], list):
                    for i, item in enumerate(rec[key], 1):
                        if item != "N/A":
                            print(f"{i}. {item}")
                else:
                    print(rec[key])
                    
            print("\n" + "=" * 80)

    def run(self) -> None:
        """Run the disease prediction system."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("""
Disease Prediction and Recommendation System
==========================================

This tool helps identify potential health conditions based on your symptoms.
Enter 'q' at any time to quit.
""")

        while True:
            try:
                symptom_input = self.display_symptoms()
                recommendations = self.predict_diseases(symptom_input)
                self.display_recommendations(recommendations)
                
                if input("\nTry another diagnosis? (Y/N): ").lower() != 'y':
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if input("\nContinue? (Y/N): ").lower() != 'y':
                    break
                    
        print("\nThank you for using the Disease Prediction System.")

if __name__ == "__main__":
    system = DiseasePredictionSystem()
    system.run()