import pandas as pd
import random
import os

def generate_sample_data(num_students=50):
    data = {
        "roll_no": [f"2024{str(i+1).zfill(3)}" for i in range(num_students)],
        "name": [f"Student {i+1}" for i in range(num_students)],
        "year_of_study": [random.randint(1, 4) for _ in range(num_students)],
        "participation": [random.randint(50, 100) for _ in range(num_students)],
        "assignments": [random.randint(40, 100) for _ in range(num_students)],
        "test_scores": [random.randint(40, 100) for _ in range(num_students)],
        "attendance": [random.randint(50, 100) for _ in range(num_students)],
        "final_grade": [random.randint(50, 100) for _ in range(num_students)]
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()
    
    # Save to Excel with specific column names
    output_path = "student_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Sample data generated and saved to {output_path}")
