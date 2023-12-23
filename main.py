# Fouzan Abdullah
# 6840797

import random
import matplotlib.pyplot as plt
import numpy as np


population_size = 100
chromosome_length = 40
mutation_rate = 1
num_generations = 50
crossover_rate = 0.1
best_fitness_data = []
avg_population_fitness_data = []


# fitness function
def fitness_function(key, text):
    expected_frequencies = [
        0.085, 0.016, 0.0316, 0.0387, 0.121, 0.0218, 0.0209, 0.0496, 0.0733, 0.0022,
        0.0081, 0.0421, 0.0253, 0.0717, 0.0747, 0.0207, 0.001, 0.0633, 0.0673, 0.0894,
        0.0268, 0.0106, 0.0183, 0.0019, 0.0172, 0.0011
    ]

    # Sanitize the cipher text and key
    text = text.lower()
    text = ''.join(filter(str.isalpha, text))

    key = key.lower()
    key = ''.join(filter(str.isalpha, key))

    key = [ord(char) - 97 for char in key]

    plain = []
    key_ptr = 0

    for char in text:
        key_char = 0

        if len(key) > 0:
            # Ignore any value not in the expected range
            while key[key_ptr] > 25 or key[key_ptr] < 0:
                key_ptr = (key_ptr + 1) % len(key)
            key_char = key[key_ptr]
            key_ptr = (key_ptr + 1) % len(key)

        decrypted_char = chr(((ord(char) - 97 + 26 - key_char) % 26) + 97)
        plain.append(decrypted_char)

    # Count the occurrences of each character
    char_counts = [0] * 26
    for char in plain:
        char_counts[ord(char) - 97] += 1

    # Calculate the total difference between the expected frequencies and the actual frequencies
    score = sum(abs((char_counts[i] / len(plain)) - expected_frequencies[i]) for i in range(26))

    return score


# function that initializes the population
def initialize_population(pop_size, chrom_length):
    character_set = "abcdefghijklmnopqrstuvwxyz-"
    return [''.join(random.choice(character_set) for _ in range(chrom_length)) for _ in range(pop_size)]


# function for selection
def select_parents(population, fitness_func, text):
    # Select parents based on a tournament selection
    parent1 = max(random.sample(population, 3), key=lambda ind: fitness_func(ind, text))
    parent2 = max(random.sample(population, 3), key=lambda ind: fitness_func(ind, text))
    return parent1, parent2


# function for one point crossover
def one_point_crossover(parent1, parent2):
    # choosing a random index
    crossover_point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2


# function for uniform crossover
def uniform_crossover(parent1, parent2):
    child1 = ['' for _ in range(len(parent1))]
    child2 = ['' for _ in range(len(parent1))]

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return ''.join(child1), ''.join(child2)


# function for three point crossover
def three_point_crossover(parent1, parent2):
    crossover_point = sorted(random.sample(range(1, parent1), 3))

    child1 = (
        parent1[:crossover_point[0]] +
        parent2[:crossover_point[0]:crossover_point[1]] +
        parent2[:crossover_point[1]:crossover_point[2]] +
        parent2[:crossover_point[2]]
    )
    child2 = (
            parent1[:crossover_point[0]] +
            parent2[:crossover_point[0]:crossover_point[1]] +
            parent2[:crossover_point[1]:crossover_point[2]] +
            parent2[:crossover_point[2]]
    )
    return child1, child2


# asks the user which crosover method to pick
selected_crossover_method = input("Choose the crossover method (o (for one point), u (for uniform), "
                                  "or t (for three point)): ").lower()


# function for mutation
def mutate(chromosome, mutation_rate):
    character_set = "abcdefghijklmnopqrstuvwxyz-"
    mutated_chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = random.choice(character_set)
    return ''.join(mutated_chromosome)


def elitism_selection(current_population, previous_population, fitness_func, text, elite_percentage=0.1):
    combined_population = current_population + previous_population
    combined_fitness_scores = [fitness_function(individual, text) for individual in combined_population]

    sorted_population = [x for _, x in sorted(zip(combined_fitness_scores, combined_population),
                                              key=lambda x: x[0])]

    elite_count = int(elite_percentage * len(sorted_population))
    elite_individuals = sorted_population[:elite_count]

    random_count = population_size - elite_count
    new_population = elite_individuals + initialize_population(random_count, chromosome_length)

    return new_population


with open('Data1.txt', 'r') as file:
    text = file.read()

fitness_data_one_point = []
fitness_data_uniform = []
fitness_data_three_point = []

# Main Genetic Algorithm loop

# Initialize the initial population
population = initialize_population(population_size, chromosome_length)

best_individual = []
best_fitness = float('inf')
fitness_values = []
min_fitness_values = []
max_fitness_values = []
median_fitness_values = []
mean_fitness_values = []
std_deviation_values = []

# Main Genetic Algorithm loop
for generation in range(num_generations):
    # Evaluate the fitness of each individual in the population
    fitness_scores = [fitness_function(individual, text) for individual in population]

    # Sort the population based on fitness scores in ascending order
    population_and_fitness = list(zip(population, fitness_scores))
    population_and_fitness.sort(key=lambda x: x[1])

    fitness_data_one_point.append(population_and_fitness[0][1])

    # Find the elites from the current generation
    elite1, elite2 = population_and_fitness[0][0], population_and_fitness[1][0]

    # Update the best individual if needed
    if population_and_fitness[0][1] < best_fitness:
        best_individual = population_and_fitness[0][0]
        best_fitness = population_and_fitness[0][1]

    population = elitism_selection(population, [], fitness_function, text, elite_percentage=0.1)
    new_population = population
    # Create the rest of the next generation
    while len(new_population) < population_size:
        parent1, parent2 = select_parents(population, fitness_function, text)
        if selected_crossover_method == 'o':
            child1, child2 = one_point_crossover(parent1, parent2)
        elif selected_crossover_method == 'u':
            child1, child2 = uniform_crossover(parent1, parent2)
        elif selected_crossover_method == 't':
            child1, child2 = three_point_crossover(parent1, parent2)
        else:
            raise ValueError("Invalid choice, please pick 'o' for one point and 'u' for uniform.")

        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.extend([child1, child2])

    population = new_population

    fitness_data_uniform.append(population_and_fitness[0][1])
    fitness_data_one_point.append(population_and_fitness[0][1])
    fitness_data_three_point.append(population_and_fitness[0][1])

    if selected_crossover_method == 'o':
        fitness_values = fitness_data_one_point
    elif selected_crossover_method == 'u':
        fitness_values = fitness_data_uniform
    elif selected_crossover_method == 't':
        fitness_values = fitness_data_three_point
    else:
        raise ValueError("Invalid choice, please pick 'o' for one point and 'u' for uniform.")

    min_fitness = np.min(fitness_values)
    max_fitness = np.max(fitness_values)
    median_fitness = np.median(fitness_values)
    mean_fitness = np.mean(fitness_values)
    std_deviation = np.std(fitness_values)

    min_fitness_values.append(min_fitness)
    max_fitness_values.append(max_fitness)
    median_fitness_values.append(median_fitness)
    mean_fitness_values.append(mean_fitness)
    std_deviation_values.append(std_deviation)

    best_fitness = population_and_fitness[0][1]
    avg_population_fitness = sum(fitness_scores) / len(fitness_scores)

    best_fitness_data.append(best_fitness)
    avg_population_fitness_data.append(avg_population_fitness)
    print(f"Generation {generation}: Best Fitness = {population_and_fitness[0][1]}")

print(f"Best individual found: {best_individual}")


def decrypted_text(key, encrypted_text):
    decrypted_text = ""
    key_length = len(key)

    with open('Data2.txt', 'r') as file:
        encrypted_text = file.read()

        for i in range(len(encrypted_text)):
            encrypted_char = encrypted_text[i]
            key_char = key[i % key_length]

            if encrypted_char.isalpha():
                is_upper = encrypted_char.isupper()
                encrypted_char = encrypted_char.lower()
                key_char = key_char.lower()
                decrypted_char = chr(((ord(encrypted_char) - ord(key_char) + 26) % 26) + ord('a'))

                if is_upper:
                    decrypted_char = decrypted_char.upper()

                decrypted_text += decrypted_char
            else:
                decrypted_text += encrypted_char

    return decrypted_text


with open('Data1.txt', 'r') as file:
    encrypted_text = file.read()
    score = fitness_function(best_individual, text)
    print(f"Best fitness: {score}")
    decrypted_text = decrypted_text(best_individual, encrypted_text)
    print(decrypted_text)

generations = range(num_generations)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(generations, best_fitness_data, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Average Best Fitness per Generation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(generations, avg_population_fitness_data, label='Avg Population Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Average Population Fitness per Generation')
plt.legend()

plt.tight_layout()
plt.show()

fitness_data = []

with open('summary_stats.txt', 'w') as writer:

    for i in range(num_generations):
        writer.write(f"Generation: {i + 1}\n "
                     f" Min_Values: {min_fitness_values[i]}\n "
                     f" Max_Values: {max_fitness_values[i]}\n "
                     f" Median_Values: {median_fitness_values[i]}\n "
                     f" Mean_Values: {mean_fitness_values[i]}\n "
                     f" Std.Deviation: {std_deviation_values[i]}\n")
