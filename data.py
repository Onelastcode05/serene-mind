import pandas as pd
import numpy as np
import random

# Ensure reproducibility across runs
np.random.seed(42)
random.seed(42)

# SCALE UP: 10,000 unique simulated profiles
num_users = 10000

# Serene Personas & Demographics
personas = ['Teen', 'Exam Warrior', 'Parent', 'Working Woman', 'Corporate']
genders = ['Male', 'Female', 'Non-binary']

# EXPANDED PARAMETERS: Expanded clinical, professional, and behavioral stressors
STRESSOR_WEIGHTS = {
    # High Impact Stressors
    'Burnout': 0.15, 
    'Exam Anxiety': 0.15,
    'Clinical Anxiety': 0.14,
    'Depressive Mood': 0.14,
    'Manager Toxicity': 0.12, 
    'Fear of Failure': 0.12,
    'Chronic Illness': 0.12,
    'Grief & Loss': 0.12,
    
    # Medium Impact Stressors
    'Financial Pressure': 0.10, 
    'Career-Home Balance': 0.10, 
    'Relationship Friction': 0.10,
    'Sleep Deprivation': 0.09,
    'Invisible Load': 0.08,
    'Peer Pressure': 0.08,
    'Academic Overload': 0.08,
    'Imposter Syndrome': 0.07,
    
    # Mild/Moderate Impact Stressors
    'Teen Disconnect': 0.06,
    'Social Media': 0.04, 
    'Meeting Fatigue': 0.04,
    'Sedentary Lifestyle': 0.04,
    'Time Management': 0.04,
    'Dietary Neglect': 0.03
}

data = []

for user_id in range(1, num_users + 1):
    persona = np.random.choice(personas)
    
    # Define how many stressors this user has selected (Dynamically choosing 4 or 5)
    num_to_select = random.choice([4, 5])
    
    # Persona-specific profiling and targeted stressor distribution
    if persona == 'Teen':
        age = np.random.randint(13, 19)
        gender = np.random.choice(genders, p=[0.48, 0.48, 0.04])
        sleep_hours = np.random.normal(7.0, 1.2)
        screen_time = np.random.normal(7.0, 1.8)
        work_study_hours = np.random.normal(5.0, 1.2)
        
        pool = ['Social Media', 'Peer Pressure', 'Fear of Failure', 'Academic Overload', 'Time Management', 'Depressive Mood', 'Dietary Neglect']
        stressors = random.sample(pool, num_to_select)
        base_stress = 0.25
        
    elif persona == 'Exam Warrior':
        age = np.random.randint(15, 22)
        gender = np.random.choice(genders, p=[0.48, 0.48, 0.04])
        sleep_hours = np.random.normal(5.5, 0.8)
        screen_time = np.random.normal(3.5, 1.0)
        work_study_hours = np.random.normal(11.5, 1.5)
        
        pool = ['Exam Anxiety', 'Fear of Failure', 'Burnout', 'Academic Overload', 'Sleep Deprivation', 'Imposter Syndrome', 'Time Management']
        stressors = random.sample(pool, num_to_select)
        base_stress = 0.35 
        
    elif persona == 'Parent':
        age = np.random.randint(32, 55)
        gender = np.random.choice(genders, p=[0.48, 0.48, 0.04])
        sleep_hours = np.random.normal(6.5, 0.9)
        screen_time = np.random.normal(3.0, 1.2)
        work_study_hours = np.random.normal(8.0, 1.8)
        
        pool = ['Teen Disconnect', 'Financial Pressure', 'Career-Home Balance', 'Relationship Friction', 'Chronic Illness', 'Time Management', 'Invisible Load']
        stressors = random.sample(pool, num_to_select)
        base_stress = 0.30
        
    elif persona == 'Working Woman':
        age = np.random.randint(22, 45)
        gender = 'Female'
        sleep_hours = np.random.normal(6.0, 1.0)
        screen_time = np.random.normal(4.0, 1.2)
        work_study_hours = np.random.normal(8.5, 1.5)
        
        pool = ['Invisible Load', 'Career-Home Balance', 'Imposter Syndrome', 'Burnout', 'Financial Pressure', 'Sleep Deprivation', 'Relationship Friction']
        stressors = random.sample(pool, num_to_select)
        base_stress = 0.32
        
    else: # Corporate
        age = np.random.randint(22, 60)
        gender = np.random.choice(genders, p=[0.58, 0.40, 0.02])
        sleep_hours = np.random.normal(6.0, 0.8)
        screen_time = np.random.normal(5.5, 1.5)
        work_study_hours = np.random.normal(9.5, 1.8)
        
        pool = ['Burnout', 'Manager Toxicity', 'Meeting Fatigue', 'Time Management', 'Sedentary Lifestyle', 'Financial Pressure', 'Clinical Anxiety']
        stressors = random.sample(pool, num_to_select)
        base_stress = 0.30
        
    # Boundary capping for metric safety
    sleep_hours = np.clip(sleep_hours, 2.0, 10.0)
    screen_time = np.clip(screen_time, 0.0, 16.0)
    work_study_hours = np.clip(work_study_hours, 0.0, 16.0)
    
    # Wellness points (App usage/gamification tracking)
    wellness_points = np.random.randint(0, 100)

    # Compute behavioral variations
    behavior_penalty = 0.0
    if sleep_hours < 6.0: 
        behavior_penalty += (6.0 - sleep_hours) * 0.08
    if work_study_hours > 9.0: 
        behavior_penalty += (work_study_hours - 9.0) * 0.04
    if screen_time > 6.0:
        behavior_penalty += (screen_time - 6.0) * 0.02
    
    # Calculate text stressor sum from 4-5 selected elements
    text_penalty = sum([STRESSOR_WEIGHTS.get(s, 0.05) for s in stressors])
    
    # Scale down relief slightly so 4-5 heavy stressors aren't entirely erased by wellness points
    wellness_relief = wellness_points * 0.0015
    
    # Introduce natural variance noise
    noise = np.random.normal(0, 0.03)
    
    # Consolidate target stress score calculation (bounded 0.0 to 1.0)
    final_stress_score = np.clip(base_stress + behavior_penalty + text_penalty - wellness_relief + noise, 0.0, 1.0)

    data.append({
        "user_id": f"usr_{user_id:05d}",
        "persona": persona,
        "age": age,
        "gender": gender,
        "sleep_avg_hours": round(sleep_hours, 1),
        "screen_time_hours": round(screen_time, 1),
        "work_study_hours": round(work_study_hours, 1),
        "wellness_points": wellness_points,
        "primary_stressors": ", ".join(stressors), 
        "stress_score": round(final_stress_score, 3)
    })

# Compile matrix and export to file system
df = pd.DataFrame(data)
df.to_csv('serene_mock_data.csv', index=False)

print(f"✓ Successfully generated scaled dataset: 'serene_mock_data.csv' with {len(df)} entries.")
print("\nQuick dataset profile peek:")
print(df[['persona', 'primary_stressors', 'stress_score']].head(3))