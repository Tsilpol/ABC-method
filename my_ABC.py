import numpy as np

# Gradient Descent for Local Optimization
def gradient_descent(func, grad, start, learning_rate=0.01, max_iter=100, tol=1e-6):
    x = np.array(start)
    for _ in range(max_iter):
        grad_val = grad(x)
        x_new = x - learning_rate * grad_val
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, func(x)

# Artificial Bee Colony Algorithm
class ArtificialBeeColony:
    def __init__(self, func, bounds, grad, num_bees=20, max_iter=100, limit=10):
        self.func = func
        self.bounds = bounds
        self.grad = grad
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.limit = limit
        self.dim = len(bounds)  # Ensure correct dimensionality for 5D problem
        self.population = self.initialize_population()  # Proper population initialization
        self.fitness = np.array([self.evaluate_solution(sol) for sol in self.population])
        self.trial = np.zeros(num_bees)

    def initialize_population(self):
        """Ensure population size is correct (num_bees x dim)."""
        return np.array([np.random.uniform(low, high, self.dim) for low, high in self.bounds])

    def evaluate_solution(self, solution):
        value = self.func(solution)
        return 1 / (1 + value) if value >= 0 else 1 + abs(value)

    def employed_bees_phase(self):
        for i in range(self.num_bees):
            # Randomly choose another bee to compare with
            k = np.random.choice([j for j in range(self.num_bees) if j != i])
            
            # Generate the candidate solution
            phi = np.random.uniform(-1, 1, self.dim)
            new_solution = self.population[i] + phi * (self.population[i] - self.population[k])

            # Ensure the new solution stays within bounds
            new_solution = np.clip(new_solution, *zip(*self.bounds))

            # Evaluate new solution
            new_fitness = self.evaluate_solution(new_solution)

            if new_fitness > self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def onlooker_bees_phase(self):
        prob = self.fitness / np.sum(self.fitness)
        for _ in range(self.num_bees):
            i = np.random.choice(range(self.num_bees), p=prob)
            k = np.random.choice([j for j in range(self.num_bees) if j != i])
            phi = np.random.uniform(-1, 1, self.dim)
            new_solution = self.population[i] + phi * (self.population[i] - self.population[k])
            new_solution = np.clip(new_solution, *zip(*self.bounds))
            new_fitness = self.evaluate_solution(new_solution)

            if new_fitness > self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def scout_bees_phase(self):
        for i in range(self.num_bees):
            if self.trial[i] > self.limit:
                self.population[i] = np.random.uniform(*zip(*self.bounds))
                self.fitness[i] = self.evaluate_solution(self.population[i])
                self.trial[i] = 0

    def optimize(self):
        best_solution = None
        best_value = float('inf')

        for _ in range(self.max_iter):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()

            # Local optimization with Gradient Descent
            for i in range(self.num_bees):
                local_solution, local_value = gradient_descent(self.func, self.grad, self.population[i])
                if local_value < 1 / self.fitness[i] - 1:
                    self.population[i] = local_solution
                    self.fitness[i] = self.evaluate_solution(local_solution)

            # Check for global best
            min_index = np.argmin(1 / self.fitness - 1)
            if 1 / self.fitness[min_index] - 1 < best_value:
                best_value = 1 / self.fitness[min_index] - 1
                best_solution = self.population[min_index]

            # Termination condition (DoubleBox)
            if best_value < 1e-6:
                break

        return best_solution, best_value

# Define benchmark functions and their gradients (with 5 dimensions)
def sphere(x):
    return np.sum(x**2)

def grad_sphere(x):
    return 2 * x

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def grad_rosenbrock(x):
    grad = np.zeros_like(x)
    grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1]**2)
    return grad

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def grad_ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    grad = np.zeros_like(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    grad += a * b * np.exp(-b * np.sqrt(sum1 / d)) * x / (np.sqrt(sum1 / d) * d)
    grad += c * np.exp(sum2 / d) * np.sin(c * x) / d
    return grad

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def grad_rastrigin(x):
    A = 10
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def griewank(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def grad_griewank(x):
    term1 = x / 2000
    term2 = np.sin(x / np.sqrt(np.arange(1, len(x) + 1))) / np.sqrt(np.arange(1, len(x) + 1))
    grad = term1 + np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) * term2
    return grad

# Define bounds for 5D problem
bounds = [(-5, 5)] * 5  # Example for 5D functions
functions = [
    (sphere, grad_sphere),
    (rosenbrock, grad_rosenbrock),
    (ackley, grad_ackley),
    (rastrigin, grad_rastrigin),
    (griewank, grad_griewank)
]

for func, grad in functions:
    abc = ArtificialBeeColony(func=func, bounds=bounds, grad=grad)
    best_solution, best_value = abc.optimize()
    print(f"Function: {func.__name__}")
    print("Best Solution:", best_solution)
    print("Best Value:", best_value)
