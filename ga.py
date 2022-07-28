#!/usr/bin/env python

from gsp import GangSchedulingProblem
from collections import namedtuple
from random import choice, randrange
from numpy import dot, zeros, array, matrix, random, sum, abs, transpose



class Problem:
    
    def __init__(self, problem_file):
        gsp = GangSchedulingProblem(problem_file)
        columns, costs = [], []
        for rotation in gsp.generate_rotations():
            column = zeros(len(gsp.tasks), dtype='uint8')
            for task in rotation.tasks:
                column[task] = 1
            columns.append(column)
            costs.append(rotation.cost)

        A = array(columns).transpose()
        m, n = A.shape

        alpha = [set() for row in range(m)]
        beta =  [set() for col in range(n)]
        for row, col in transpose(A.nonzero()):
            alpha[row].add(col)
            beta [col].add(row)

        self.A = A
        self.costs = array(costs)
        self.nr_tasks, self.nr_rotations = m, n
        self.alpha = list(map(frozenset, alpha))
        self.beta  = list(map(frozenset, beta))

    def __repr__(self):
        return '<GSP problem, %dx%d>' % (self.nr_tasks, self.nr_rotations)



Solution = namedtuple('Solution', 'columns covering fitness unfitness')

def make_solution(problem, columns):
    covering = dot(problem.A, columns)
    fitness = dot(problem.costs, columns)
    unfitness = sum(abs(covering - 1))
    return Solution(columns, covering, fitness, unfitness)

def initial_solution(problem):
    columns = zeros(problem.nr_rotations, dtype='uint8')
    I = frozenset(range(problem.nr_tasks))
    S, U = set(), set(I)
    while U:
        i = choice(list(U))
        J = [j for j in problem.alpha[i] if not (problem.beta[j] & (I - U))]
        if J:
            j = choice(J)
            columns[j] = 1
            U -= problem.beta[j]
        else:
            U.remove(i)
    #columns = random.randint(2, size=problem.nr_rotations).astype('uint8')
    return make_solution(problem, columns)

def binary_tournament(population):
    population_size = len(population)
    candidates = [randrange(population_size) for _ in range(2)]
    return min(candidates, key=lambda index: population[index].fitness)

def matching_selection(problem, population):
    '''Indices of the solutions selected for crossover.'''
    P1 = binary_tournament(population)
    if population[P1].unfitness == 0:
        P2 = binary_tournament(population)
    else:
        cols_P1 = population[P1].columns
        def compatibility(k):
            cols_Pk = population[k].columns
            comp = sum(cols_P1 | cols_Pk) - sum(cols_P1 & cols_Pk)
            return comp, population[P1].fitness  # use fitness to break ties
        P2 = max((k for k, sol in enumerate(population) if k != P1),
                 key=compatibility)
    return (P1, P2)

def uniform_crossover(parent1, parent2):
    '''Columns of the child after crossover.'''
    mask = random.randint(2, size=parent1.columns.size)
    return mask * parent1.columns + (1 - mask) * parent2.columns

def static_mutation(columns, M_s=3):
    for _ in range(M_s):
        j = randrange(columns.size)
        columns[j] = 1 - columns[j]
    return columns

def adaptive_mutation(columns, M_a=5, epsilon=0.5):
    return columns


def repair(solution):
    # ...
    return solution
    
def ranking_replacement(population, child):
    # Labels for solutions according to relation with child.
    # Keys are pairs: (better fitness than child?, better unfitness than child?)
    groups = {
        (False, False): 1, (True, False): 2,
        (False, True):  3, (True, True):  4,
    }
    solution_group = (groups[solution.fitness   < child.fitness,
                             solution.unfitness < child.unfitness]
                      for solution in population)

    # replacement criteria:
    # 1. group with smallest label
    # 2. worst unfitness
    # 3. worst fitness
    sort_key = lambda k_sol_group: (k_sol_group[1][1], -k_sol_group[1][0].unfitness, -k_sol_group[1][0].fitness)    
    worst_k, (worst_sol, worst_group) = min(((k, (sol, group)) for k, (sol, group) in enumerate(zip(population, solution_group))), key=sort_key)

    population[worst_k] = child
    return worst_k



def best_solution(population):
    sort_key = lambda k_sol: (k_sol[1].unfitness, k_sol[1].fitness)        
    best_k, best_sol = min(((k, sol) for k, sol in enumerate(population)), key=sort_key)
    return best_k
    

def ga(problem, population_size=100, nr_iterations=1000):
    population = [initial_solution(problem) for k in range(population_size)]
    best_k = best_solution(population)
    for t in range(nr_iterations):
        p1, p2 = matching_selection(problem, population)
        child = uniform_crossover(population[p1], population[p2])
        child = static_mutation(child)
        child = adaptive_mutation(child)
        child = repair(child)
        child = make_solution(problem, child)
        child_k = ranking_replacement(population, child)
        if best_solution([population[best_k], child]) == 1:
            best_k = child_k
            print("Found better child!") 
    return population[best_k]

    

    
def main():
    problem_file = open("gsp_50tasks.txt")
    problem = Problem(problem_file)
    print(ga(problem)) 

if __name__ == '__main__':
    main()

