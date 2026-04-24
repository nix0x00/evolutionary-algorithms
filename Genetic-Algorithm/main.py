import random
import copy
import math
import multiprocessing
import numpy as np
from deap import base, tools, creator, algorithms
from dataset_2 import requests, staff, supplies
from functions import build_gene, cxTwoPointCustomMate, mutCustom, optimize_route, distance, print_solution, get_staff_for_request
from elitism import eaSimpleWithElitism
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


POP_SIZE = 660
NGEN = 750
CXPB = 0.9
MUTPB = 0.3
INDPB = 1/4


# FITNESS
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
        staff_work_days[s_id].add(svd)

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
        # However, if that next day should be within the request's start and end day.
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

            # if the load increases the capacity then return to depot
            if curr_v_load + supply_load > VEHICLE_CAPACITY:
                curr_v_load, curr_loc = 0, 1 # reset

                dist_to = distance(curr_loc, loc_id) # re calculate distance from 1 to target
                total_penalties += 4000

            curr_v_dist += dist_to
            curr_v_load += supply_load
            curr_loc = loc_id

            # PENALTY: if single request exceeds the capacity
            if supply_load > VEHICLE_CAPACITY:
                total_penalties += 5000

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

    return (total_cost,)


def create_individual():
    return [build_gene(requests[r_id], DAYS) for r_id in requests]

toolbox = base.Toolbox()
creator.create("fitnessMin", base=base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.fitnessMin)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("createPopulation", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", cxTwoPointCustomMate)
toolbox.register(
    "mutate", mutCustom, indpb=INDPB, DAYS=DAYS, requests=requests, staff=staff
)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda individual: individual.fitness.values[0])
stats.register("min", np.min)
stats.register("mean", np.mean)

all_times_best = None
all_times_worst = None
avg_cost = 0
log = None

def run_ga_iteration(run_id):
    random.seed() 
    
    population = toolbox.createPopulation(POP_SIZE)
    hall_of_fame = tools.HallOfFame(5)

    final_population, log = eaSimpleWithElitism(
        population=population,
        halloffame=hall_of_fame,
        stats=stats,
        toolbox=toolbox,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        verbose=((run_id + 1) == NUM_RUNS)
    )
    
    return hall_of_fame[0], log


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    with multiprocessing.Pool() as pool:
        best_individuals = pool.map(run_ga_iteration, range(NUM_RUNS))
        
    for best, logbook in best_individuals:
        cost = best.metrics['total_cost']

        avg_cost += (cost - best.metrics['penalties'])
        
        if all_times_worst is None or cost > all_times_worst.metrics['total_cost']:
            all_times_worst = best
            
        if all_times_best is None or cost < all_times_best.metrics['total_cost']:
            all_times_best = best
            log = logbook

    all_times_best.metrics['total_worst'] = all_times_worst.metrics['total_cost'] - all_times_worst.metrics['penalties']
    all_times_best.metrics['avg_cost'] = avg_cost/NUM_RUNS
    best_ga = all_times_best
    print_solution(all_times_best, requests, DAYS, "best_ga.txt")

    plt.plot(log.select("min"), color="red")
    plt.plot(log.select("mean"), color="green")
    plt.xlabel("generations")
    plt.ylabel("min/avg fitness per population")
    plt.show()
