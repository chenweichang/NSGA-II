import matplotlib.pyplot as plt
from nsga2 import *


# Configuration
k = 10
num_population = 100
count_generation = 100


# Input data
database = input_data('chess')


# items
items = []
for t in database:
    il = [i for i, _ in t]
    items.extend(il)
items = list(set(items))



# Compute the maximum support
max_sup = max([calc_sup(database, [i]) for i in items])


# Compute the maximum utility
max_uti = max([calc_uti(database, [i]) for i in items])



# Initialize
first_population = init_first_generation(database, num_population, k, items, max_sup, max_uti)



x = np.array([i.avg_sup for i in first_population])
y = np.array([i.avg_uti for i in first_population])
plt.xlabel('sup')
plt.ylabel('uti')
plt.plot(x, y, 'ro')
plt.show()



# Main Loop
result = main_loop(first_population, count_generation, database, 0.1, 0.1, len(items), k)








