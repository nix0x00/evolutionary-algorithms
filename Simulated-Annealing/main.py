import random
import copy
import math
import multiprocessing
import numpy as np
from deap import base, tools, creator, algorithms
from dataset_2 import requests, staff, supplies
from functions import build_gene, cxTwoPointCustomMate, mutCustom, optimize_route, distance, print_solution, get_staff_for_request
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


class MyList(list):
    pass

def create_individual():
    return MyList([build_gene(requests[r_id], DAYS) for r_id in requests])


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
        staff_work_days[s_id].add(svd) # NEW

        # 1. PENALTY: Although staff wil match requested speciality, but we will still add condition just in case
        if medical_staff[request['specialty_needed']] != 1:
            total_penalties += 20000
        
        # 2. PENALTY: Check if delivery day (dd) is within the delivery range otherwise penalty
        if dd > request['day_end']:
            days_to_penalise = dd - request['day_end']
            penalty = (days_to_penalise * supplies[request['supply_id']]['penalty_per_day'] * supplies[request['supply_id']]['criticality']) + 1000
            total_penalties += penalty
            total_delay_penalties += penalty

        # PENALTY: Delivery cannot happen before the start day
        if dd < request['day_start']:
            total_penalties += 15000
    
        # PENALTY: Staff can only visit after the supply has been delivered i.e. on same day or the next day.
        if svd < dd or svd > dd + 1:
            total_penalties += 15000
            total_delay_penalties += 15000
        if svd > request['day_end']:
            total_penalties += 16000
            total_delay_penalties += 16000


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
                curr_v_load, curr_loc = 0, 1 # reset

                dist_to = distance(curr_loc, loc_id) # re calculate distance from 1 to target
                total_penalties += 30000
                # continue

            curr_v_dist += dist_to
            curr_v_load += supply_load
            curr_loc = loc_id

            # PENALTY: if single request exceeds the capacity
            if supply_load > VEHICLE_CAPACITY:
                total_penalties += 15000


        # PENALTY: daily distance limit
        if curr_v_dist > VEHICLE_MAX_DISTANCE:
            total_penalties += 15000

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
            total_penalties += 40000

        for r_id in route:
            loc = requests[r_id]["location_id"]
            dist_to = distance(curr_s_loc, loc)
            curr_s_loc = loc

            curr_s_dist += dist_to


        # PENALTY: Staff should not exceed max distance per day
        if curr_s_dist > medical_staff['max_distance']:
            total_penalties += 15000

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
                    total_penalties += 15000
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

def get_smart_neighbor(individual):
    neighbor = copy.deepcopy(individual)
    idx = random.randrange(len(neighbor))
    v_id, s_id, dd, svd = neighbor[idx]
    r_id = idx + 1
    r = requests[r_id]
    
    choice = random.random()
    
    if choice < 0.30:
        # Move request to a clustered day (safer version)
        other_idx = random.randrange(len(neighbor))
        new_dd = neighbor[other_idx][2]
        new_svd = random.choice([new_dd, min(new_dd + 1, DAYS)])
        
        neighbor[idx] = (neighbor[other_idx][0], s_id, new_dd, new_svd)
        
    elif choice < 0.60:
        # Staff consolidation (safe)
        active_staff_on_day = [ind[1] for ind in neighbor if ind[3] == svd]
        possible_staff = [sid for sid in active_staff_on_day if staff[sid][r["specialty_needed"]] == 1]
        
        if possible_staff:
            new_sid = random.choice(possible_staff)
        else:
            valid_ids = list(get_staff_for_request(r_id, requests, staff).keys())
            new_sid = random.choice(valid_ids)
        
        neighbor[idx] = (v_id, new_sid, dd, svd)

    elif choice < 0.85:
        # Safe day shift
        new_dd = random.randint(r["day_start"], r["day_end"])
        new_svd = random.choice([new_dd, min(new_dd + 1, DAYS)])
        
        neighbor[idx] = (v_id, s_id, new_dd, new_svd)
        
    else:
        # LESS destructive swap (partial swap instead of full)
        idx2 = random.randrange(len(neighbor))
        
        v2, s2, dd2, svd2 = neighbor[idx2]
        
        # Swap only days (safer)
        neighbor[idx] = (v_id, s_id, dd2, random.choice([dd2, min(dd2 + 1, DAYS)]))
        neighbor[idx2] = (v2, s2, dd, random.choice([dd, min(dd + 1, DAYS)]))
        
    return neighbor

def simulated_annealing(run_id):
    random.seed()
    temp = 15000.0          # slightly reduced
    alpha = 0.999           # faster cooling
    min_temp = 1e-3
    iterations_per_temp = 20
    
    current_sol = create_individual()
    current_cost = fitness(current_sol)[0]
    best_sol = copy.deepcopy(current_sol)
    best_cost = current_cost
    
    history = []
    no_improvement = 0

    for step in range(20000):
        for _ in range(iterations_per_temp):
            neighbor = get_smart_neighbor(current_sol)
            neighbor_cost = fitness(neighbor)[0]
            
            delta = neighbor_cost - current_cost
            
            # ✅ SAFE acceptance
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / temp) if temp > 1e-8 else 0
                accept = random.random() < prob
            
            if accept:
                current_sol = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_sol = copy.deepcopy(current_sol)
                    best_cost = current_cost
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1

        history.append(best_cost)

        # Cooling
        temp *= alpha

        # ✅ Early stopping
        if temp < min_temp:
            print("Temperature threshold reached.")
            break

        # ✅ Reheating (escape local minima)
        if no_improvement > 5000:
            temp = min(temp * 1.2, 10000)  # clamp to initial temp
            no_improvement = 0
            print(f"Reheating at step {step}, temp reset to {temp:.2f}")

        if step % 500 == 0:
            print(f"Step: {step} | Temp: {temp:.2f} | Best: {int(best_cost)}")

    return best_sol, history

if __name__ == "__main__":
    best_solution, cost_history = None, None
    all_time_lowest_cost = float('inf')

    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    with multiprocessing.Pool() as pool:
        print(f"Launching {NUM_RUNS} parallel SA runs...")
        results = pool.map(simulated_annealing, range(NUM_RUNS))
    
    for best_sol, hist in results:
        current_run_best_cost = fitness(best_sol)[0]

        if current_run_best_cost < all_time_lowest_cost:
            all_time_lowest_cost = current_run_best_cost
            best_solution = best_sol
            cost_history = hist

    fitness(best_solution) 
    best_sa = best_solution
    print_solution(best_solution, requests, DAYS, "best_sa.txt")

    plt.plot(cost_history, color="green", label="Global Best Cost")
    plt.title('SA Convergence')
    plt.show()
    