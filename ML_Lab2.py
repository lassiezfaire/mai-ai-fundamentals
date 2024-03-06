import pygmo as pg


class SphereFunction:
    def __init__(self):
        self.title = 'Sphere Function'

    def fitness(self, x):
        return [x[0] ** 2 + x[1] ** 2]

    def get_bounds(self):
        return [-4.5, 4.5], [-4.5, 4.5]


class MatyasFunction:
    def __init__(self):
        self.title = 'Matyas Function'

    def fitness(self, x):
        return [0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]]

    def get_bounds(self):
        return [-10, -10], [10, 10]


def optimization(algorithm, problem):
    prob = pg.problem(problem)
    algo = pg.algorithm(algorithm(gen=100))

    pop = pg.population(prob, size=10)
    result = algo.evolve(pop)

    best_solution = result.champion_x

    print(f"Алгоритм: {algorithm.__name__}, {problem.title}")
    print(f"Лучшее решение: {best_solution}")
    print(f"Значение функции: {problem.fitness(best_solution)[0]}")
    print("\n")


# Сравнение для функции сферы
sphere = SphereFunction()
optimization(pg.de, sphere)
optimization(pg.gwo, sphere)
optimization(pg.sea, sphere)

# Сравнение для функции Матьяса
matyas = MatyasFunction()
optimization(pg.de, matyas)
optimization(pg.gwo, matyas)
optimization(pg.sea, matyas)
