import pandas as pd
import random

# List of Indian names from Maharashtra
names = [
    "Atharv Khawale", "Priyaal Gaykwad", "Mukesh Gole", "Devansh Nandanwar", "Mayur Dhawale",
    "Rohan Patil", "Sneha Jadhav", "Aditya Deshmukh", "Pooja Shinde", "Siddharth Pawar",
    "Anjali Kulkarni", "Rahul Joshi", "Neha Sawant", "Kunal More", "Shruti Chavan",
    "Amit Bhosale", "Prachi Salunkhe", "Vishal Kadam", "Komal Gaikwad", "Sagar Thakur"
]

# Generate synthetic data
data = {
    "Name": random.choices(names, k=50),  # Randomly select names
    "Attendance": [random.randint(50, 100) for _ in range(50)],  # Attendance percentage
    "Test Scores": [random.randint(40, 100) for _ in range(50)],  # Test scores
    "Assignments": [random.randint(40, 100) for _ in range(50)],  # Assignment scores
    "Performance": [random.randint(50, 100) for _ in range(50)]  # Overall performance
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_path = "d:\\College Project\\Project\\Student-Performance-Prediction\\performance\\student_data.xlsx"
df.to_excel(output_path, index=False)

print(f"Dataset saved to {output_path}")
