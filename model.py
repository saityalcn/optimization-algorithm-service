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

regression_model = joblib.load('regression_model.joblib')

def calculate_strength(sample_input):
  reshaped_array = pd.DataFrame([sample_input])
  return regression_model.predict(reshaped_array)[0]

# Genetik algoritma parametreleri
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2

# Başlangıç popülasyonu oluştur
def create_individual(param_ranges):
    return {param: random.uniform(param_range[0], param_range[1]) for param, param_range in param_ranges.items()}

# Fitness fonksiyonu
def evaluate(individual, TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX):
    # Burada, veri setine göre strength hesaplaması yapılmalıdır.
    calculated_strength = calculate_strength(individual)

    # Hedef aralığa uygunluk kontrolü
    target_fitness = 1.0 - abs(calculated_strength - TARGET_STRENGTH_MIN) / (TARGET_STRENGTH_MAX - TARGET_STRENGTH_MIN)

    return target_fitness

def genetic_algorithm(TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX):

  # Parametrelerin genişlik ve min-max değerleri
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

  population = [create_individual(param_ranges) for _ in range(POPULATION_SIZE)]

  # Genetik algoritma ana döngüsü
  for generation in range(GENERATIONS):
      # Fitness değerlerini hesapla
      fitness_values = [evaluate(individual, TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX) for individual in population]

      # Elitizm: En iyi bireyleri seç
      elite_indices = sorted(range(POPULATION_SIZE), key=lambda i: fitness_values[i], reverse=True)[:int(0.1 * POPULATION_SIZE)]
      elites = [population[i] for i in elite_indices]

      # Yeni popülasyonu oluştur
      new_population = elites[:]

      while len(new_population) < POPULATION_SIZE:
          # Seçim (turnuva seçimi)
          parent1 = random.choice(population)
          parent2 = random.choice(population)

          # Çaprazlama (one-point crossover)
          if random.random() < CROSSOVER_PROBABILITY:
              crossover_point = random.choice(list(param_ranges.keys()))
              child = {param: parent1[param] if random.random() < 0.5 else parent2[param] for param in param_ranges.keys()}
          else:
              child = parent1.copy()

          # Mutasyon (uniform mutation)
          for param in param_ranges.keys():
              if random.random() < MUTATION_PROBABILITY:
                  child[param] = random.uniform(param_ranges[param][0], param_ranges[param][1])

          new_population.append(child)

      # Yeni popülasyonu eski popülasyon olarak güncelle
      population = new_population

  # En iyi bireyi bul ve sonuçları görüntüle
  evaluate_with_args = partial(evaluate, TARGET_STRENGTH_MIN=TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX=TARGET_STRENGTH_MAX)
  best_individual = max(population, key=evaluate_with_args)
  #print("En iyi parametreler:", best_individual)
  #print("En iyi fitness değeri:", evaluate(best_individual, TARGET_STRENGTH_MIN, TARGET_STRENGTH_MAX))
  #print("Parametrelerle Tahmin Edilen Strength Değeri: ", calculate_strength(best_individual))

  return best_individual

def optimization_model(orders, rawMaterials):
    veriler = []

    # quantity 100 denince genetik algoritma çıktılarında her bir field - age hariç - 100 ile çarpılacak.
    
    for order in orders:
        veriler.append(genetic_algorithm(order["min"], order["max"]))

    #print(len(veriler))

    for veri in veriler: 
        print(veri)

    # Lineer Programlama Modeli Oluştur
    model = LpProblem(name="Veri_Optimizasyon", sense=LpMaximize)
    # Değişkenleri Tanımla
    secenekler = [LpVariable(f"veri_{i}", cat="Binary") for i in range(len(veriler))]

    # Parametreler için maksimum kısıtları ekleyin
    max_values = rawMaterials
    # Hedef Fonksiyonu (örneğin, toplam hedefi maksimize etmeye çalışıyoruz)
    model += lpSum(secenekler[i] for i in range(len(veriler)))

    for param in max_values:
        model += lpSum(veriler[i][param] * secenekler[i] for i in range(len(veriler))) <= max_values[param]


    # Çözümü Yazdır
    #print("Çözüm Durumu:", LpProblem.status[model.status])
    #print("En İyi Değer:", model.objective.value())

    # Modeli Çöz
    model.solve()

    # Seçilen verileri yazdır
    selected_data = [veriler[i] for i in range(len(veriler)) if secenekler[i].value() == 1]
    print("Seçilen Veriler:")
    for data in selected_data:
        print(data)

    return selected_data