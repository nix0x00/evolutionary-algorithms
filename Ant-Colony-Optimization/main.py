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


NUM_ANTS = 75
ITERATIONS = 500
EVAPORATION_RATE = 0.1758445017408108
Q = 13294.184088858867  # Pheromone deposit factor (scaled to your high costs)
ALPHA = 0.6839026392514785  # Pheromone importance
BETA = 1.9990941123903003  # Heuristic importance


search_space = {}

for r_id, req in requests.items():
    valid_choices = []
    # Only consider staff that actually have the required specialty (avoids 500 penalty)
    possible_staff = [
        s_id for s_id, s_data in staff.items() if s_data[req["specialty_needed"]] == 1
    ]

    for v_id in range(1, MAX_VEHICLES + 1):
        for s_id in possible_staff:
            # Delivery day: from allowed start to end of schedule
            for dd in range(req["day_start"], DAYS + 1):
                # Staff visit must be on the delivery day or the day after (avoids 300 penalty)
                for svd in range(dd, min(dd + 2, DAYS + 1)):

                    # Heuristic: We want to heavily prefer delivery days that don't incur penalties
                    delay = max(0, dd - req["day_end"])
                    delay_penalty = (
                        delay
                        * supplies[req["supply_id"]]["penalty_per_day"]
                        * supplies[req["supply_id"]]["criticality"]
                    )

                    # Inverse of penalty (add 1 to avoid division by zero).
                    # If penalty is 0, heuristic is 1.0. If penalty is high, heuristic is near 0.
                    heuristic_value = 1.0 / (1.0 + delay_penalty)

                    valid_choices.append(
                        {"tuple": (v_id, s_id, dd, svd), "heuristic": heuristic_value}
                    )

    search_space[r_id] = valid_choices

# Initialize Pheromones to 1.0 for all valid choices
pheromones = {
    r_id: {choice["tuple"]: 1.0 for choice in choices}
    for r_id, choices in search_space.items()
}


class AntSolution:
    def __init__(self, gene_list):
        self.gene_list = gene_list
        self.metrics = {}
        self.total_cost = 0

    # Add this method so functions.py can loop over it!
    def __iter__(self):
        return iter(self.gene_list)


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
            total_penalties += 500
        
        # 2. PENALTY: Check if delivery day (dd) is within the delivery range otherwise penalty
        if dd > request['day_end']:
            days_to_penalise = dd - request['day_end']
            penalty = (days_to_penalise * supplies[request['supply_id']]['penalty_per_day'] * supplies[request['supply_id']]['criticality'])
            total_penalties += penalty
            total_delay_penalties += penalty

        # PENALTY: Delivery cannot happen before the start day
        if dd < request['day_start']:
            total_penalties += 500
    
        # PENALTY: Staff can only visit after the supply has been delivered i.e. on same day or the next day.
        if svd < dd or svd > dd + 1:
            total_penalties += 800
        if svd > request['day_end']:
            total_penalties += 1000


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
                # curr_v_dist += distance(curr_loc, 1)
                curr_v_load, curr_loc = 0, 1 # reset

                dist_to = distance(curr_loc, loc_id) # re calculate distance from 1 to target
                total_penalties += 1000
                # continue

            curr_v_dist += dist_to
            curr_v_load += supply_load
            curr_loc = loc_id

            # PENALTY: if single request exceeds the capacity
            if supply_load > VEHICLE_CAPACITY:
                total_penalties += 2000

        # PENALTY: daily distance limit
        if curr_v_dist > VEHICLE_MAX_DISTANCE:
            total_penalties += 300

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
            total_penalties += 2000

        for r_id in route:
            loc = requests[r_id]["location_id"]
            dist_to = distance(curr_s_loc, loc)
            curr_s_loc = loc

            curr_s_dist += dist_to


        # PENALTY: Staff should not exceed max distance per day
        if curr_s_dist > medical_staff['max_distance']:
            total_penalties += 400


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
                    total_penalties += 550
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
    individual.total_cost = total_cost

    return total_cost


# MAIN ACO ALGORITHM
print("Starting Ant Colony Optimization...")

def run_aco(run_id):
    np.random.seed()
    random.seed()

    global_best_ant = None
    history_best = []
    history_mean = []

    for iteration in range(ITERATIONS):
        colony = []
        iteration_costs = []

        # 1. Ants construct solutions
        for _ in range(NUM_ANTS):
            solution_genes = []

            for r_id in sorted(requests.keys()):
                choices = search_space[r_id]
                probabilities = []

                # Calculate probability for each choice
                for choice in choices:
                    tau = pheromones[r_id][choice["tuple"]]
                    eta = choice["heuristic"]
                    prob = (tau**ALPHA) * (eta**BETA)
                    probabilities.append(prob)

                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    probabilities = [1.0 / len(choices)] * len(choices)
                else:
                    probabilities = [p / prob_sum for p in probabilities]

                # Roulette wheel selection
                chosen_idx = np.random.choice(len(choices), p=probabilities)
                solution_genes.append(choices[chosen_idx]["tuple"])

            # Create ant and evaluate
            ant = AntSolution(solution_genes)
            cost = fitness(ant)
            colony.append(ant)
            iteration_costs.append(cost)

            # Track global best
            if global_best_ant is None or cost < global_best_ant.total_cost:
                global_best_ant = ant

        # 2. Pheromone Evaporation
        for r_id in requests.keys():
            for tpl in pheromones[r_id]:
                pheromones[r_id][tpl] *= 1.0 - EVAPORATION_RATE

        # Sort colony by cost (lowest is best)
        colony.sort(key=lambda x: x.total_cost)

        # Let top 10% of ants deposit pheromones
        top_ants = colony[: max(1, int(NUM_ANTS * 0.1))]

        for ant in top_ants:
            deposit_amount = Q / ant.total_cost
            for i, gene in enumerate(ant.gene_list):
                pheromones[i + 1][gene] += deposit_amount

        # Global Best deposits extra (Max-Min approach flavor)
        deposit_amount_best = (Q * 2) / global_best_ant.total_cost
        for i, gene in enumerate(global_best_ant.gene_list):
            pheromones[i + 1][gene] += deposit_amount_best

        # Logging
        mean_cost = np.mean(iteration_costs)
        history_best.append(global_best_ant.total_cost)
        history_mean.append(mean_cost)

        if (iteration + 1) % 10 == 0:
            print(
                f"Gen {iteration + 1:3d} | Best Cost: {global_best_ant.total_cost:.0f} | Mean Cost: {mean_cost:.0f} | Penalties: {global_best_ant.metrics['penalties']:.0f}"
            )

    return global_best_ant, history_best, history_mean


if __name__ == "__main__":
    all_time_best_ant = None
    history_mean_ant = None
    global_best_history = None

    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    with multiprocessing.Pool() as pool:
        print(f"Launching {NUM_RUNS} parallel ACO runs...")
        results = pool.map(run_aco, range(NUM_RUNS))

    for i, (best_ant, history_best, history_mean) in enumerate(results):
        if all_time_best_ant is None or best_ant.total_cost < all_time_best_ant.total_cost:
            all_time_best_ant = best_ant
            global_best_history = history_best
            history_mean_ant = history_mean

    best_aco = all_time_best_ant


    # RESULTS
    print("\n=== ACO OPTIMIZATION COMPLETE ===")
    print_solution(best_aco, requests, DAYS, "best_aco.txt")

    # Plotting
    plt.plot(global_best_history, color="red", label="Global Best Cost")
    plt.plot(history_mean_ant, color="green", label="Generation Mean Cost")
    plt.title("ACO Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.show()
