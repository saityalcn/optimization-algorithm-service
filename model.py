import random
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from functools import partial
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import joblib

linear_regression_model = joblib.load('linear_regression_model.joblib')
random_forest_regressor= joblib.load('random_forest_model.joblib')


param_ranges = {
    'cement': (102, 540),
    'slag': (0, 359),
    'ash': (0, 200),
    'water': (122, 247),
    'superplastic': (0, 32),
    'coarseagg': (800, 1145),
    'fineagg': (594, 992),
    'age': (1, 70)
}

def initialize_parameters():
    initial_parameters = {}
    for parameter, (min_value, max_value) in param_ranges.items():
        initial_parameters[parameter] = np.random.uniform(min_value, max_value)

    return initial_parameters

def calculate_strength(sample_input, regressinModelKey):
    reshaped_array = pd.DataFrame([sample_input])
    print(regressinModelKey)

    if(regressinModelKey == "linearRegressor"):
        return linear_regression_model.predict(reshaped_array)[0]

    else:
        return random_forest_regressor.predict(reshaped_array)[0]

# Genetik algoritma parametreleri
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2

# Başlangıç popülasyonu oluştur
def create_individual(param_ranges):
    return {param: random.uniform(param_range[0], param_range[1]) for param, param_range in param_ranges.items()}

# Fitness fonksiyonu
def evaluate(individual, TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX, regressinModelKey):
    # Burada, veri setine göre strength hesaplaması yapılmalıdır.
    calculated_strength = calculate_strength(individual, regressinModelKey)

    # Hedef aralığa uygunluk kontrolü
    target_fitness = 1.0 - abs(calculated_strength - TARGET_STRENGTH_MIN) / (TARGET_STRENGTH_MAX - TARGET_STRENGTH_MIN)

    return target_fitness

def genetic_algorithm(TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX, regressinModelKey):
  population = [create_individual(param_ranges) for _ in range(POPULATION_SIZE)]

  for generation in range(GENERATIONS):
      fitness_values = [evaluate(individual, TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX, regressinModelKey) for individual in population]

      elite_indices = sorted(range(POPULATION_SIZE), key=lambda i: fitness_values[i], reverse=True)[:int(0.1 * POPULATION_SIZE)]
      elites = [population[i] for i in elite_indices]

      new_population = elites[:]

      while len(new_population) < POPULATION_SIZE:
          parent1 = random.choice(population)
          parent2 = random.choice(population)

          if random.random() < CROSSOVER_PROBABILITY:
              crossover_point = random.choice(list(param_ranges.keys()))
              child = {param: parent1[param] if random.random() < 0.5 else parent2[param] for param in param_ranges.keys()}
          else:
              child = parent1.copy()

          for param in param_ranges.keys():
              if random.random() < MUTATION_PROBABILITY:
                  child[param] = random.uniform(param_ranges[param][0], param_ranges[param][1])

          new_population.append(child)

      population = new_population

  evaluate_with_args = partial(evaluate, TARGET_STRENGTH_MIN=TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX=TARGET_STRENGTH_MAX, regressinModelKey=regressinModelKey)
  best_individual = max(population, key=evaluate_with_args)

  return best_individual

def gradient_descent(initial_parameters, learning_rate, max_iterations, target_strength_range, regressinModelKey):
    current_parameters = {key: value for key, value in initial_parameters.items()}

    target_strength_min, target_strength_max = target_strength_range

    for iteration in range(max_iterations):
        gradient = calculate_gradient(current_parameters, target_strength_min, regressinModelKey)

        for parameter in current_parameters:
            current_parameters[parameter] -= learning_rate * gradient[parameter]

        current_strength = calculate_strength(current_parameters, regressinModelKey)
        if target_strength_min <= current_strength <= target_strength_max:
            break

    return current_parameters

def calculate_gradient(parameters, target_strength, regressinModelKey):
    epsilon = 1e-6  # Küçük bir epsilon değeri
    gradient = {}

    for parameter in parameters:
        original_value = parameters[parameter]

        parameters[parameter] = original_value + epsilon
        strength_plus_epsilon = calculate_strength(parameters, regressinModelKey)

        parameters[parameter] = original_value - epsilon
        strength_minus_epsilon = calculate_strength(parameters, regressinModelKey)

        gradient[parameter] = (strength_plus_epsilon - strength_minus_epsilon) / (2 * epsilon)

        parameters[parameter] = original_value

    return gradient

def gradient_descent_algorithm(target_strength_min, target_strength_max, regressinModelKey):
  initial_parameters = initialize_parameters()
  target_strength_range = (target_strength_min, target_strength_max)
  learning_rate = 0.01
  max_iterations = 1000

  return gradient_descent(initial_parameters, learning_rate, max_iterations, target_strength_range, regressinModelKey)

def dynamic_programming_algorithm(orders, raw_materials):
  maxOfRawMaterials = max(raw_materials.values())
  dp_matrix = [[0] * (maxOfRawMaterials + 1) for _ in range(len(orders) + 1)]
  weights = []
  values = []

  weights.append(0)
  values.append(0)

  for i in range(1, len(orders)+1):
    weights.append(max(orders[i-1].values()))
    values.append(1)

  for i in range(1, len(orders) + 1):
    for j in range(1, maxOfRawMaterials + 1):
      if weights[i - 1] <= j:
        dp_matrix[i][j] = max(dp_matrix[i - 1][j], values[i - 1] + dp_matrix[i - 1][math.ceil(j - weights[i - 1])])
      else:
        dp_matrix[i][j] = dp_matrix[i - 1][j]

  selected_items = []
  i, j = len(orders), maxOfRawMaterials
  while i > 0 and j > 0:
    if dp_matrix[i][j] != dp_matrix[i - 1][j]:
        selected_items.append(i - 1)
        j -= math.ceil(weights[i - 1])

    i -= 1

  selected_items.reverse()
  return [orders[i] for i in selected_items]

def linear_programming_algorithm(veriler, max_values):
    model = LpProblem(name="Veri_Optimizasyon", sense=LpMaximize)
    secenekler = [LpVariable(f"veri_{i}", cat="Binary") for i in range(len(veriler))]

    model += lpSum(secenekler[i] for i in range(len(veriler)))

    for param in max_values:
        model += lpSum(veriler[i][param] * secenekler[i] for i in range(len(veriler))) <= max_values[param]

    model.solve()

    return [veriler[i] for i in range(len(veriler)) if secenekler[i].value() == 1]

def optimization_model(orders, rawMaterials, algorithmKey, operationsResearchMethodKey, regressinModelKey):
    veriler = []
    selected_data = []

    # quantity 100 denince genetik algoritma çıktılarında her bir field - age hariç - 100 ile çarpılacak.
    
    for order in orders:
        if(algorithmKey == "genetic"):
            veriler.append(genetic_algorithm(order["min"], order["max"], regressinModelKey))

        elif(algorithmKey == "gradDescent"):
            veriler.append(gradient_descent_algorithm(order["min"], order["max"], regressinModelKey))

    
    if(operationsResearchMethodKey == "dynamicProgramming"):
        selected_data = dynamic_programming_algorithm(veriler, rawMaterials)

    else:
        selected_data = linear_programming_algorithm(veriler, rawMaterials)

    print("END")

    return selected_data