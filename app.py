import streamlit as st
import pandas as pd
import csv
import random

# Function to read the CSV file
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# Fitness Function
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Genetic Algorithm Functions
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, ratings, generations, population_size, crossover_rate, mutation_rate, elitism_size, all_programs):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule, ratings), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population

    best_schedule = max(population, key=lambda schedule: fitness_function(schedule, ratings))
    return best_schedule

# Streamlit Interface
st.title(" Genetic Algorithm for TV Program Schedular(JIE42903 Assignment)")
st.markdown("""
### Developed by: Wan Muhammad Hafiz Bin Wan Ibrahim  
""")

# Dataset Display
st.subheader("📊 Program Ratings Dataset")
file_path = "program_ratings_modified.csv"

try:
    dataset_df = pd.read_csv(file_path)
    st.dataframe(dataset_df, use_container_width=True)
except FileNotFoundError:
    st.error(" The file 'program_ratings_modified.csv' was not found. Please make sure it's in the same folder as this app.")
    st.stop()

ratings = read_csv_to_dict(file_path)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 6 + len(all_programs)))  # time slots: 6 AM onwards

# Sidebar Controls

st.sidebar.header("⚙️ Genetic Algorithm Settings")
GEN = st.sidebar.slider("Generations", 10, 500, 100, 10)
POP = st.sidebar.slider("Population Size", 10, 100, 50, 5)
EL_S = st.sidebar.slider("Elitism Size", 1, 5, 2, 1)

# Parameter Inputs for 3 Trials
st.subheader(" Set Parameters for Each Trial")

cols = st.columns(3)
trial_params = []
for i, col in enumerate(cols, start=1):
    with col:
        st.markdown(f"**Trial {i} Parameters**")
        CO_R = st.slider(f"Trial {i} - Crossover Rate", 0.0, 0.95, 0.8, 0.01, key=f"co{i}")
        MUT_R = st.slider(f"Trial {i} - Mutation Rate", 0.01, 0.05, 0.02, 0.01, key=f"mut{i}")
        trial_params.append((CO_R, MUT_R))

# Run All Trials Button
if st.button("🚀 Run All Trials"):
    results = []
    for i, (CO_R, MUT_R) in enumerate(trial_params, start=1):
        initial_schedule = all_programs.copy()
        random.shuffle(initial_schedule)

        best_schedule = genetic_algorithm(
            initial_schedule,
            ratings,
            generations=GEN,
            population_size=POP,
            crossover_rate=CO_R,
            mutation_rate=MUT_R,
            elitism_size=EL_S,
            all_programs=all_programs
        )

        total_fitness = fitness_function(best_schedule, ratings)
        results.append({
            "Trial": f"Trial {i}",
            "Crossover Rate": CO_R,
            "Mutation Rate": MUT_R,
            "Total Rating": round(total_fitness, 2),
            "Schedule": best_schedule
        })

    # Display results for each trial
    for res in results:
        st.markdown(f"### 📈 {res['Trial']} Results")
        st.write(f"**Crossover Rate:** {res['Crossover Rate']} | **Mutation Rate:** {res['Mutation Rate']}")
        df = pd.DataFrame({
            "Time Slot": [f"{t}:00" for t in all_time_slots],
            "Program": res["Schedule"]
        })
        st.table(df)
        st.write(f"**Total Fitness (Rating):** {res['Total Rating']}")
        st.markdown("---")

    # Summary Table
    st.subheader(" Trial Summary")
    summary_df = pd.DataFrame([
        {
            "Trial": res["Trial"],
            "Crossover Rate": res["Crossover Rate"],
            "Mutation Rate": res["Mutation Rate"],
            "Total Rating": res["Total Rating"]
        }
        for res in results
    ])
    st.dataframe(summary_df, use_container_width=True)
    st.success("✅ All trials completed successfully!")
