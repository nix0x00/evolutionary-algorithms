import random
import copy
import math
import multiprocessing
import numpy as np
from deap import base, tools, creator, algorithms
from dataset_2 import requests, staff, supplies
from functions import build_gene, cxTwoPointCustomMate, mutCustom, optimize_route, distance, print_solution
import matplotlib.pyplot as plt


"""
Requirement:
    1. Delivery emergency supplies to locations
    2. Schedule medical personnel to visit locations
    
Chromosome:
 ________________Request ID # 1______________, _____...
[[VehID, StaffID, DeliveryDay, StaffVisitDay], [......]]
"""

DAYS = 20
VEHICLE_CAPACITY = 15
VEHICLE_MAX_DISTANCE = 400
VEHICLE_DISTANCE_COST = 40
VEHICLE_DAY_COST = 300
VEHICLE_FIXED_COST = 12000

STAFF_DISTANCE_COST = 25
STAFF_DAY_COST = 200
STAFF_FIXED_COST = 6000
STAFF_WORK_THRESHOLD = 5
MAX_VEHICLES = 5

NUM_RUNS = 10


# POP_SIZE = 460
# NGEN = 400
POP_SIZE = 660
NGEN = 750
CXPB = 0.8
MUTPB = 0.2
INDPB = 0.3


# MULTI-OBJECTIVE FITNESS
def fitness(individual):
    total_v_cost = 0
    total_v_dist = 0
    total_s_cost = 0
    total_s_dist = 0
    total_penalties = 0
    total_delay_penalties = 0

    vehicle_usage = set()
    vehicles_per_day = {}
    vehicle_plan = {}

    staff_usage = set()
    staff_work_days = {s_id: set() for s_id in staff} # [] => set()
    staff_plan = {}

    for id, gene in enumerate(individual):
        r_id = id + 1
        v_id, s_id, dd, svd = gene

        request = requests[r_id]
        medical_staff = staff[s_id]

        staff_plan.setdefault((svd, s_id), []).append(r_id)
        vehicle_plan.setdefault((dd, v_id), []).append(r_id)
        vehicles_per_day.setdefault(dd, set()).add(v_id)
        staff_work_days[s_id].add(svd) # NEW

        # 1. PENALTY: Although staff wil match requested speciality, but we will still add condition just in case
        if medical_staff[request['specialty_needed']] != 1:
            total_penalties += 5000
        
        # 2. PENALTY: Check if delivery day (dd) is within the delivery range otherwise penalty
        if dd > request['day_end']:
            days_to_penalise = dd - request['day_end']
            penalty = (days_to_penalise * supplies[request['supply_id']]['penalty_per_day'] * supplies[request['supply_id']]['criticality'])
            total_penalties += penalty
            total_delay_penalties += penalty

        # PENALTY: Delivery cannot happen before the start day
        if dd < request['day_start']:
            total_penalties += 5000
    
        # PENALTY: Staff can only visit after the supply has been delivered i.e. on same day or the next day.
        if svd < dd or svd > dd + 1:
            total_penalties += 5000
            total_delay_penalties += 5000
        if svd > request['day_end']:
            total_penalties += 7000
            total_delay_penalties += 7000


    daily_vehicle_costs = {}
    # VEHICLE ROUTING
    for (day, v_id), req_list in vehicle_plan.items():
        route = optimize_route(req_list=req_list, requests=requests)

        curr_v_load, curr_v_dist, curr_loc = 0, 0, 1
        vehicle_usage.add(v_id)

        for r_id in route:
            req = requests[r_id]
            loc_id = req['location_id']

            supply_load = supplies[req['supply_id']]['size'] * req['quantity']

            # calculate distance to X and from X back to Depot 1
            dist_to = distance(curr_loc, loc_id)

            #Exclude return distance
            # dist_back = distance(loc_id, 1)

            # if the load increases the capacity then return to depot
            if curr_v_load + supply_load > VEHICLE_CAPACITY:
                # Exclude MID Route return distance to depot 1
                # curr_v_dist += distance(curr_loc, 1)
                curr_v_load, curr_loc = 0, 1 # reset

                dist_to = distance(curr_loc, loc_id) # re calculate distance from 1 to target
                total_penalties += 3000
                # continue

            curr_v_dist += dist_to
            curr_v_load += supply_load
            curr_loc = loc_id

            # PENALTY: if single request exceeds the capacity
            if supply_load > VEHICLE_CAPACITY:
                total_penalties += 5000

        # Exclude distance for Depot
        # curr_v_dist += distance(curr_loc, 1)

        # PENALTY: daily distance limit
        if curr_v_dist > VEHICLE_MAX_DISTANCE:
            total_penalties += 5000

        # COST: VEHICLE COSTS
        v_standard_cost = VEHICLE_DAY_COST + (curr_v_dist * VEHICLE_DISTANCE_COST)
        daily_vehicle_costs[(day, v_id)] = v_standard_cost

        total_v_dist += curr_v_dist
        # COST: VEHICLE COSTS
        total_v_cost += v_standard_cost

    # COST/PENALTY: VEHICLE DAILY LIMIT PENALTY
    for day, vset in vehicles_per_day.items():
        if len(vset) > MAX_VEHICLES:
            v_list = list(vset)
            v_list.sort(key=lambda x: daily_vehicle_costs[(day, x)], reverse=True)

            extra_vehicles = len(vset) - MAX_VEHICLES
            for i in range(extra_vehicles):
                veh_id = v_list[i]
                total_v_cost += daily_vehicle_costs[(day, veh_id)]

    # COST: VEHICLE FIXED COST
    total_v_cost += (len(vehicle_usage) * VEHICLE_FIXED_COST)


    # STAFF ROUTING
    for (day, s_id), req_list in staff_plan.items():
        medical_staff = staff[s_id]
        home_loc = medical_staff["location_id"]

        curr_s_loc, curr_s_dist = home_loc, 0

        staff_usage.add(s_id)

        route = optimize_route(req_list, requests)

        # PENALTY: Staff should not exceed allowed patients visits per day
        if len(route) > medical_staff["max_patients_per_day"]:
            total_penalties += 5000

        for r_id in route:
            loc = requests[r_id]["location_id"]
            dist_to = distance(curr_s_loc, loc)
            curr_s_loc = loc

            curr_s_dist += dist_to

        # Staff returns to home location, calculate the distance
        # Excluding the return distance
        # dist_back = distance(curr_s_loc, home_loc)
        # curr_s_dist += dist_back


        # PENALTY: Staff should not exceed max distance per day
        if curr_s_dist > medical_staff['max_distance']:
            total_penalties += 5000


        total_s_dist += curr_s_dist
        # COST: Distance cost
        total_s_cost += STAFF_DAY_COST + (curr_s_dist * STAFF_DISTANCE_COST)

    # COST: Fixed cost for the active staff
    total_s_cost += (len(staff_usage) * STAFF_FIXED_COST)

    # PENALTY: Staff must not work on 6th day if they have been working for 5 days consecutively.
    for s_id, days in staff_work_days.items():
        sorted_days = sorted(list(days))
        streak = 1

        for i in range(1, len(sorted_days)):
            if sorted_days[i] == sorted_days[i-1] + 1:
                streak += 1
                if streak > 5:
                    total_penalties += 10000
            else:
                streak = 1

    total_cost = total_v_cost + total_s_cost + total_penalties 

    individual.metrics = {
        "vehicle_distance": total_v_dist,
        "vehicle_usage_days": vehicles_per_day,
        "vehicle_usage": vehicle_usage,
        "staff_distance": total_s_dist,
        "vehicle_cost": total_v_cost,
        "staff_cost": total_s_cost,
        "staff_usage": staff_usage,
        "staff_usage_days": staff_work_days,
        "penalties": total_penalties,
        "delay_penalties": total_delay_penalties,
        "total_cost": total_cost
    }

    # ================= NSGA-II OBJECTIVES =================
    return (
        total_cost,  # Obj 1: Cost
        total_v_dist + total_s_dist,  # Obj 2: Distance
        total_penalties,
    )

# INITIALIZATION
def create_individual():
    return [build_gene(requests[r_id], DAYS) for r_id in requests]

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", cxTwoPointCustomMate)
toolbox.register("mutate", mutCustom, indpb=INDPB, DAYS=DAYS, requests=requests, staff=staff)
toolbox.register("select", tools.selNSGA2)

stats = tools.Statistics(lambda individual: individual.fitness.values[0])
stats.register("min", np.min)

# NSGA-II LOOP
def run_nsga2(run_id):
    random.seed()
    population = toolbox.population(n=POP_SIZE)

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(population, len(population))
    hof = tools.ParetoFront()
    final_population, logbook = algorithms.eaMuPlusLambda(
        population, 
        toolbox, 
        mu=POP_SIZE, 
        lambda_=POP_SIZE, 
        cxpb=CXPB, 
        mutpb=MUTPB, 
        ngen=NGEN, 
        stats=stats,
        halloffame=hof,
        verbose=((run_id + 1) == NUM_RUNS)
    )
    history = logbook.select("min")
    return hof, history


if __name__ == '__main__':
    all_runs_results = []
    best_nsga_cost = float('inf')
    nsga2_history = None
    massive_combined_population = []
    
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    with multiprocessing.Pool() as pool:
        all_runs_results = pool.map(run_nsga2, range(NUM_RUNS))

    for final_pop, history in all_runs_results:
        massive_combined_population.extend(final_pop)
 
    # Extract Pareto Front
    pareto_front = tools.sortNondominated(massive_combined_population, len(massive_combined_population), first_front_only=True)[0]

    print(f"\nTotal Pareto Solutions: {len(pareto_front)}")

    best_nsga = min(pareto_front, key=lambda ind: ind.fitness.values[0] - ind.fitness.values[2])
    
    print_solution(best_nsga, requests, DAYS, "best_nsga.txt")

    # VISUALIZATION
    cost = [ind.fitness.values[0] - ind.fitness.values[2] for ind in pareto_front]
    dist = [ind.fitness.values[1] for ind in pareto_front]

    plt.scatter(dist, cost)
    plt.xlabel("Distance")
    plt.ylabel("Cost")
    plt.title("Pareto Front")
    plt.show()