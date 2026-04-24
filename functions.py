import random
from dataset_2 import staff, locations, supplies
import math
import os


def get_staff_for_request(request_id, requests, staff):
    specialty = requests[request_id]["specialty_needed"]

    matched_staff = {
        staff_id: staff_info
        for staff_id, staff_info in staff.items()
        if staff_info.get(specialty) == 1
    }

    return matched_staff


# VehID, StaffID, DeliveryDay, StaffVisitDay
def build_gene(r, DAYS):
    v_id = random.randint(1, 5)

    # Delivery day within feasible window
    max_delivery = min(r["day_end"], DAYS)
    dd = random.randint(r["day_start"], max_delivery)

    # Staff visit MUST be after delivery
    svd = random.randint(dd, DAYS)

    # Choose only valid staff
    possible_staff = [
        s_id for s_id, s in staff.items() if s[r["specialty_needed"]] == 1
    ]
    s_id = random.choice(possible_staff)

    return (v_id, s_id, dd, svd)


def mutCustom(individual, indpb, DAYS, requests, staff):
    for i in range(len(individual)):
        if random.random() < indpb:
            vehicle_id, staff_id, delivery_day, staff_day = individual[i]

            r_id = list(requests.keys())[i]
            r = requests[r_id]

            # Always define valid staff
            possible_staff = list(get_staff_for_request(r_id, requests, staff).keys())

            if random.random() < 0.7:
                # small tweak
                delivery_day += random.choice([-1, 0, 1])
                delivery_day = max(r["day_start"], min(delivery_day, r["day_end"]))
            else:
                delivery_day = random.randint(r["day_start"], r["day_end"])

            if random.random() < 0.2:
                vehicle_id = random.randint(1, 3)

            if random.random() < 0.5:
                staff_id = random.choice(possible_staff)

            if random.random() < 0.7:
                staff_day = delivery_day + random.choice([0, 1])
            else:
                staff_day = random.randint(delivery_day, DAYS)

            # enforce constraint
            staff_day = max(delivery_day, min(staff_day, DAYS))

            individual[i] = (vehicle_id, staff_id, delivery_day, staff_day)

    return (individual,)


def cxTwoPointCustomMate(ind1, ind2):
    point1 = random.randint(0, len(ind1) - 2)
    point2 = random.randint(point1 + 1, len(ind1))

    ind1[point1:point2], ind2[point1:point2] = ind2[point1:point2], ind1[point1:point2]

    return ind1, ind2


def distance(loc1, loc2):
    x1, y1 = locations[loc1]["x"], locations[loc1]["y"]
    x2, y2 = locations[loc2]["x"], locations[loc2]["y"]
    return math.hypot(x1 - x2, y1 - y2)


def optimize_route(req_list, requests):
    if not req_list:
        return []

    remaining = req_list[:]
    route = []
    curr_loc = 1  # depot

    while remaining:
        next_req = min(
            remaining, key=lambda r: distance(curr_loc, requests[r]["location_id"])
        )
        route.append(next_req)
        curr_loc = requests[next_req]["location_id"]
        remaining.remove(next_req)

    return route


def print_solution(individual, requests, DAYS, file_name="best_ga.txt"):
    name = "Name = Uzair Mustafa"
    print(name)
    
    with open(file_name, 'w') as file:
        file.write(f"{name}\n")

    vehicle_plan = {}
    staff_plan = {}

    # BUILD PLANS
    for i, gene in enumerate(individual):
        r_id = i + 1
        v_id, s_id, dd, svd = gene

        vehicle_plan.setdefault(dd, {}).setdefault(v_id, []).append(r_id)
        staff_plan.setdefault(svd, {}).setdefault(s_id, []).append(r_id)

    # PRINT DAY-WISE OUTPUT
    for day in range(1, DAYS + 1):
        v_day = vehicle_plan.get(day, {})
        s_day = staff_plan.get(day, {})

        with open(file_name, 'a') as file:
            file.write(f"DAY = {day}   NUMBER_OF_VEHICLES = {len(v_day)}\n")
        
        print(f"\n{'='*70}")
        print(
            f"DAY = {day}  |  NUMBER_OF_VEHICLES = {len(v_day)}  |  NUMBER_OF_MEDICAL_STAFF = {len(s_day)}"
        )
        print(f"{'='*70}")

        # -------- VEHICLES --------
        for v_id, req_list in v_day.items():
            route = optimize_route(req_list, requests)

            route_locs = [requests[r]["location_id"] for r in route]

            with open(file_name, 'a') as file:
                file.write(f"1 {' '.join(str(l) for l in route_locs)} \n")

            # Print: depot → locations → depot
            print(
                f"(V_ID: {v_id})      Locations >   1 -> {' -> '.join(str(l) for l in route_locs)} -> 1",
                f"      |   [REQ_IDS:     {', '.join(str(l) for l in route)} ]",
            )

        with open(file_name, 'a') as file:
            file.write(f"NUMBER_OF_MEDICAL_STAFF = {len(s_day)} \n")

        # -------- STAFF --------
        print(f"NUMBER_OF_MEDICAL_STAFF = {len(s_day)}")

        for s_id, req_list in s_day.items():
            # Optimize staff visit order (optional but consistent)
            home_loc = staff[s_id]["location_id"]
            route = optimize_route(req_list, requests)

            route_locs = [requests[r]["location_id"] for r in route]

            with open(file_name, 'a') as file:
                file.write(f"{home_loc} {' '.join(str(l) for l in route_locs)}\n")

            print(
                f"(S_ID: {s_id})      Locations >   {home_loc} -> {' -> '.join(str(l) for l in route_locs)} -> {home_loc}",
                f"      |   [REQ_IDS:     {', '.join(str(l) for l in route)} ]",
            )

    # SUMMARY
    metrics = individual.metrics

    with open(file_name, 'a') as file:
        file.write(f"\nSUMMARY: TOTAL_VEHICLE_DISTANCE = {int(metrics['vehicle_distance'])}\n")
        file.write(f"VEHICLE_USAGE_DAYS = {len(metrics['vehicle_usage_days'])}\n")
        file.write(f"UNIQUE_VEHICLES_USED = {len(metrics['vehicle_usage'])}\n")
        file.write(f"TOTAL_STAFF_DISTANCE = {int(metrics['staff_distance'])}\n")
        file.write(f"STAFF_USAGE_DAYS = {len(metrics['staff_usage_days'])}\n")
        file.write(f"UNIQUE_STAFF_USED = {len(metrics['staff_usage'])}\n")
        file.write(f"TOTAL_DELAY_PENALTIES = {int(metrics['delay_penalties'])}\n")
        file.write(f"TOTAL_COST = {int(metrics['total_cost']) - int(metrics['penalties'])}\n")


    print("\nSUMMARY:")
    print(f"TOTAL_VEHICLE_DISTANCE = {int(metrics['vehicle_distance'])}")
    print(f"VEHICLE_USAGE_DAYS = {len(metrics['vehicle_usage_days'])}")
    print(f"UNIQUE_VEHICLES_USED = {len(metrics['vehicle_usage'])}")

    print(f"TOTAL_STAFF_DISTANCE = {int(metrics['staff_distance'])}")
    print(f"STAFF_USAGE_DAYS = {len(metrics['staff_usage_days'])}")
    print(f"UNIQUE_STAFF_USED = {len(metrics['staff_usage'])}")

    print(f"TOTAL_DELAY_PENALTIES = {int(metrics['delay_penalties'])}")
    print(f"TOTAL_PENALTIES = {int(metrics['penalties'])}")
    print(f"TOTAL_COST = {int(metrics['total_cost']) - int(metrics['penalties'])}")
