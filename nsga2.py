import numpy as np
import random
import sys
import copy
import functools
import matplotlib.pyplot as plt

class Individual:
    avg_sup = 0
    avg_uti = 0
    n = 0
    s = []
    rank = 0
    distance = 0

    def __init__(self, pattern):
        self.pattern = pattern




def input_data(data):
    url = '/Users/caoheng/PycharmProjects/NSGA-II/dataset/'

    # Input items
    items = []
    for line in open(url + data + '.txt'):
        il = [int(i) for i in line.split()]
        items.append(il)

    # Input weight
    weight = []
    for line in open(url + data + '-weight.txt'):
        li = [int(i) for i in line.split()]
        weight.append(li)

    # Initialize the database
    database = []
    t = tuple(zip(items, weight))
    for m, n in t:
        database.append(tuple(zip(m, n)))

    return database


def init_first_generation(database, num_population, k, items, max_sup, max_uti):


    # Initialize the first generation.
    population = []

    #
    # for l in np.linspace(0, 1, num_population/2):
    #     # Sort the items by support and utility
    #     sorted_items = sorted(items, key=lambda it: calc_sort_priority(database, it, l, max_sup, max_uti), reverse=True)
    #
    #
    #     # individual
    #     pattern = np.zeros(shape=(k, len(items)), dtype=int)
    #     sum_sup = 0
    #     sum_uti = 0
    #     for i in range(k):
    #         pattern[i, sorted_items[i]-1] = 1
    #         sum_sup += calc_sup(database, [sorted_items[i]])
    #         sum_uti += calc_uti(database, [sorted_items[i]])
    #
    #     individual = Individual(pattern)
    #     individual.avg_sup = sum_sup/k
    #     individual.avg_uti = sum_uti/k
    #     population.append(individual)
    #
    #     # progress bar
    #     print("Initializing...[" + "%.2f" % (len(population) / num_population * 100) + "%]", end='\r')

    for i in range(int(num_population)):
        random_transactions = random.sample(database, k)
        pattern = np.zeros(shape=(k, len(items)), dtype=int)
        sum_sup = 0
        sum_uti = 0
        for idx, t in enumerate(random_transactions):
            il = random.sample([i for i, _ in t], random.randint(1, 5))
            sum_sup += calc_sup(database, il)
            sum_uti += calc_uti(database, il)
            for item in il:
                pattern[idx, item-1] = 1
        individual = Individual(pattern)
        individual.avg_sup = sum_sup/k
        individual.avg_uti = sum_uti/k
        population.append(individual)


        # progress bar
        print("Process...[" + "%.2f" % (len(population) / num_population * 100) + "%]", end='\r')




    return population


def calc_sup(database, il):
    n = 0
    for t in database:
        if set(il) <= set([i for i, _ in t]):
            n = n+1
    return n/len(database)



def calc_uti(database, il):
    uti = 0
    for t in database:
        uti += sum([w for i, w in t if il.__contains__(i)])
    return uti



def calc_sort_priority(database, item, l, max_sup, max_uti):
    return l*calc_sup(database, [item])/max_sup + (1-l)*calc_uti(database, [item])/max_uti




def fast_non_dominated_sort(population):
    for p in population:
        p.n = 0
        p.s = []
        for q in population:
            if is_dominated(p, q):
                p.s.append(q)
            elif is_dominated(q, p):
                p.n += 1


        if p.n == 0:
            p.rank = 1

    i = 1
    while [individual.rank for individual in population].__contains__(i):
        for p in [individual for individual in population if individual.rank == i]:
            for q in p.s:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i+1
        i += 1




def is_dominated(p, q):
    """
    :return: True if p dominates q
    """
    sup_p = p.avg_sup
    sup_q = q.avg_sup
    uti_p = p.avg_uti
    uti_q = q.avg_uti

    if (sup_p > sup_q and uti_p >= uti_q) or (sup_p >= sup_q and uti_p > uti_q) or (sup_p > sup_q and uti_p > uti_q):
        return True
    else:
        return False





def main_loop(first_population, count_generation, database, c_p, m_p, n_items, k):
    for g in range(count_generation):
        new_population = make_new_population(database, first_population, c_p, m_p, n_items, k)

        # combine parent and offspring population
        combined_population = first_population.copy()
        combined_population.extend(new_population)
        fast_non_dominated_sort(combined_population)
        next_population = []

        # until the parent population if filled
        i = 1
        while True:
            front_i = [individual for individual in combined_population if individual.rank == i]
            n_front = len(front_i)
            if len(next_population) + n_front >= len(first_population):
                break

            crowding_distance_assignment(front_i)
            next_population.extend(front_i)
            i += 1

        front_i = [individual for individual in combined_population if individual.rank == i]
        front_i.sort(key=functools.cmp_to_key(crowded_comparison_operator))

        next_population.extend(front_i[: len(first_population)-len(next_population)-1])
        first_population = next_population

        # progress bar
        print("Initializing...[" + "%.2f" % (g / count_generation * 100) + "%]", end='\r')

        # plot
        x = np.array([i.avg_sup for i in first_population])
        y = np.array([i.avg_uti for i in first_population])
        plt.xlabel('sup')
        plt.ylabel('uti')
        plt.plot(x, y, 'ro')
        plt.show()

    return first_population




def crowded_comparison_operator(i, j):
    if i.rank < j.rank or (i.rank == j.rank and i.distance > j.distance):
        return -1
    else:
        return 1



def crowding_distance_assignment(population):
    max_sup = max([i.avg_sup for i in population])
    max_uti = max([i.avg_uti for i in population])

    # Initialize distance
    for individual in population:
        individual.distance = 0

    sorted_by_sup = sorted(population, key=lambda it: it.avg_sup, reverse=True)
    sorted_by_uti = sorted(population, key=lambda it: it.avg_uti, reverse=True)


    # support
    sorted_by_sup[0].distance = sys.maxsize
    sorted_by_sup[-1].distance = sys.maxsize

    for idx in range(1, len(population)-1):
        sorted_by_sup[idx].distance += (sorted_by_sup[idx-1].avg_sup - sorted_by_sup[idx+1].avg_sup)/\
                                      (sorted_by_sup[0].avg_sup - sorted_by_sup[-1].avg_sup)/max_sup


    # utility
    sorted_by_uti[0].distance = sys.maxsize
    sorted_by_uti[-1].distance = sys.maxsize
    for idx in range(1, len(population)-1):
        sorted_by_uti[idx].distance += (sorted_by_uti[idx-1].avg_uti - sorted_by_uti[idx+1].avg_uti)/\
                                       (sorted_by_uti[0].avg_uti - sorted_by_uti[-1].avg_uti)/max_uti




def make_new_population(database, population, c_p, m_p, n_items, k):
    new_population = tournament_selection(population)
    recombination(database, new_population, c_p, n_items, k)
    mutation(population, m_p, n_items, k)
    return new_population




def recombination(database, population, c_p, n_items, k):
    count_combination = int(c_p * len(population))
    for _ in range(count_combination):
        while True:
            # which row
            r = random.randint(0, k-1)

            # two individuals
            i_1, i_2 = random.sample(population, 2)


            # crossover points
            points = [random.randint(0, n_items) for _ in range(2)]
            left = min(points)
            right = max(points)

            # process
            tmp = copy.deepcopy(i_1.pattern[r, left:right])
            i_1.pattern[r, left:right] = i_2.pattern[r, left:right]
            i_2.pattern[r, left:right] = tmp



            # check
            if is_suitable_recombination(database, i_1.pattern, r) and\
                is_suitable_recombination(database, i_2.pattern, r):
                calc_avg_sup(database, i_1, k)
                calc_avg_uti(database, i_2, k)
                break
            tmp = copy.deepcopy(i_1.pattern[r, left:right])
            i_1.pattern[r, left:right] = i_2.pattern[r, left:right]
            i_2.pattern[r, left:right] = tmp





def calc_avg_sup(database, individual, k):
    sum_sup = 0
    for r in individual.pattern:
        il = [i for i in r if i == 1]
        sum_sup += calc_sup(database, il)

    individual.avg_sup = sum_sup/k




def calc_avg_uti(database, individual, k):
    sum_uti = 0
    for r in individual.pattern:
        il = [i for i in r if i == 1]
        sum_uti += calc_uti(database, il)

    individual.avg_uti = sum_uti / k





def is_suitable_recombination(database, pattern, r):
    for il in pattern:
        if id(il) != id(pattern[r]) and np.array_equal(il, pattern[r]):
            return True

    il = [i for i in pattern[r] if i == 1]
    for t in database:
        if not set(il) <= set([i for i,_ in t]):
            return True

    return False





def mutation(population, m_p, n_items, k):
    count_mutation = int(m_p * len(population))
    for _ in range(count_mutation):
        selected_population, = random.sample(population, 1)
        r = random.randint(0, k-1)
        c = random.randint(0, n_items-1)
        selected_population.pattern[r, c] = 1- selected_population.pattern[r, c]




def tournament_selection(population):
    max_sup = max([i.avg_sup for i in population])
    max_uti = max([i.avg_uti for i in population])
    new_population = []
    # possibility
    n = 0
    l = []
    for individual in population:
        n += individual.avg_uti/max_sup + individual.avg_uti/max_uti
        l.append(n)

    pl = []
    for i in l:
        pl.append(i/l[-1])

    # selection
    for _ in range(len(population)):
        r = random.random()
        for idx, j in enumerate(pl):
            if r < j:
                new_population.append(copy.deepcopy(population[idx]))
                break
    return new_population








