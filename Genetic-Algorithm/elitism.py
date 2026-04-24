import copy
from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty for elitism!")

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update HoF with the initial population
    halloffame.update(population)

    # Record stats for generation 0
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        hof_size = len(halloffame)
        offspring = toolbox.select(population, len(population) - hof_size)

        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(copy.deepcopy(halloffame))

        # Update the Hall of Fame with the newly formed population
        halloffame.update(offspring)

        # Replace the current population with the offspring (including the elites)
        population[:] = offspring

        # Append stats to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook