from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    data = {}
    # Set the number of vehicles
    data['num_vehicles'] = 4
    # Set the number of cities
    data['num_cities'] = 5
    # Define the distances between cities
    data['distances'] = [
        [0, 1, 2, 3, 4],  # Distance from city 0 to cities 0, 1, 2, 3, 4
        [1, 0, 2, 3, 4],  # Distance from city 1 to cities 0, 1, 2, 3, 4
        [2, 2, 0, 3, 4],  # Distance from city 2 to cities 0, 1, 2, 3, 4
        [3, 3, 3, 0, 4],  # Distance from city 3 to cities 0, 1, 2, 3, 4
        [4, 4, 4, 4, 0],  # Distance from city 4 to cities 0, 1, 2, 3, 4
    ]
    # Define the mapping of cities to vehicles
    data['city_to_vehicle'] = [[0, 1], [0], [1, 2], [2], [3]]  # City i is mapped to vehicles j

    return data

def main():
    # Create the data model
    data = create_data_model()

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(data['num_cities'], data['num_vehicles'], data['city_to_vehicle'])

    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)

    # Set the cost function (distance) between cities
    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distances'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set the search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Print the solution
    if solution:
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                plan_output += ' {} ->'.format(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            plan_output += ' {}\n'.format(manager.IndexToNode(index))
            print(plan_output)
    else:
        print('No solution found !')

if __name__ == '__main__':
    main()


