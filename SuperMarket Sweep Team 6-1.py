import os
import csv
import math
import multiprocessing as mp
from pprint import pprint
import pandas as pd
from gurobipy import GRB, Model


def read_data():
    #Read in Data
    with open("Supermarket Sweep.csv", "r") as infile:
        reader = csv.reader(infile)
        reader_list = list(reader)

    dist_list = [["item_i_index", "item_j_index", "item_i", "item_j", "distance", "d_ij", "c_j"]]         

    def calculate_distance(data_list):
        for i in range(1, len(data_list)):
            for j in range(1, len(data_list)):
                dist = 0
                if data_list[i][2] == data_list[j][2]:
                    dist = abs(int(data_list[i][3]) - int(data_list[j][3]))
                    dist_list.append([i-1,j-1, reader_list[i][0], reader_list[j][0], dist, dist/10, reader_list[j][1]])
                else:
                    dist = min(int(data_list[i][3]) + int(data_list[j][3]) + abs(int(data_list[i][2]) - int(data_list[j][2])), (110 - int(data_list[i][3])) + (110 - int(data_list[j][3])) + abs(int(data_list[i][2]) - int(data_list[j][2]))) 
                    dist_list.append([i-1,j-1, reader_list[i][0], reader_list[j][0], dist, dist/10, reader_list[j][1]])

        return dist_list
            
    output_list = calculate_distance(reader_list)

    with open('PartA.csv', 'w', newline='') as outputfile:
        w = csv.writer(outputfile)
        w.writerows(output_list)

    with open("Supermarket Sweep.csv", "r") as infile:
        master_reader = csv.reader(infile)
        master_list = list(master_reader)
        
    df = pd.read_csv('Supermarket Sweep.csv')
    item_list = df.iloc[:, 0]
    with open("PartA.csv", "r") as infile:
        distance_reader = csv.reader(infile)
        distance_list = list(distance_reader)
    
    return master_list, distance_list, item_list

def parameter_maker(master_list,distance_list):
    #Create List of Prices with Start Node as 0 index and rest of items being indices 1 - 56
    prices = []
    for i in range(1,len(master_list)):
        prices.append(float(master_list[i][1]))

    #Number of Locations
    items = []
    for i in range(len(master_list)-1):
        items.append(i)

    #Create List of List of Distances -- First list is distance from item 0 to item j
    distances = []
    for i in items:
        local_list = []
        for j in items:
            local_list.append(distance_list[i * len(items) + j + 1][5])
        distances.append(local_list)
    
    return prices,items,distances

def df_maker(time, capacity, obj_val, results_df, full_tour, item_list):
    results_df_copy = results_df.copy()
    print(f'Solution Tour: {full_tour}')
    item_names = []
    # Create a new row dictionary
    for node in full_tour:
        item_names.append(item_list[node])
    new_row = {'Time': time, 'Capacity': capacity, 'Objective Value': obj_val, 'Tour': full_tour, 'Items': item_names}
    new_df = pd.DataFrame(data=[new_row])
    print(f'Solution Items: {item_names}')
    # Appends new row dictionary to the results DataFrame
    results_df_copy = pd.concat([results_df, new_df], ignore_index=True)
    print(results_df_copy)
    # Return the updated results DataFrame
    return results_df_copy

def Subtour_Eliminator(time, capacity, results_df, obj_val,subtours_cleaned, value_of_tour_from_start, switch, full_tour, item_list, last_solved_solution):
    # Helper function to convert a subtour to a list of edges
    def converter(subtour):
            list_of_pairs = [[subtour[x],subtour[x+1]] for x in range(len(subtour)-1)]
            list_of_pairs.append([subtour[len(subtour)-1],subtour[0]])     
            return list_of_pairs
    
    # Remove subtours that contain the starting position and have already been found
    for i in subtours_cleaned:
        if 0 in i and last_solved_solution  <= value_of_tour_from_start:
            subtours_cleaned.remove(i)

    # Convert remaining subtours to lists of edges and store them for future use
    if len(subtours_cleaned) > 0:
        for i in subtours_cleaned:
            subtour_storage.append(converter(i))

    # Set switch to True if there are still remaining subtours
    if 0 != len(subtours_cleaned):
        switch = True

    # If all subtours have been found, save the results dataframe to a CSV file
    if switch == False:
        
        results_df = df_maker(time, capacity,obj_val, results_df,full_tour, item_list)
        #results_df.to_csv(f'Value data/Capacity x Time y .csv', index=False)
        results_df.to_csv(f'Value data/Capacity {str(capacity)} Time {str(time)}.csv', index=False)
        results_df.to_csv(f'Value data/All_Opt_Vals_Capacity_Time.csv', index=False)
    return results_df,switch

def supermarket_model(prices,items,distances, time, capacity, iteration, results_df, item_list, last_solved_solution):
        #Create Model
        num_processes = math.floor(mp.cpu_count()-1)
        m = Model('Supermarket Sweep')
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)
        m.setParam(GRB.Param.Threads,num_processes)
        #Variables
        x = m.addVars(items, items, vtype = GRB.BINARY, name ='x')

        #Constraints

        #Start at 0 Once
        m.addConstr(sum(x[0,j] for j in range(1,len(items))) == 1)
        #End at 0 Once
        m.addConstr(sum(x[i,0] for i in range(1,len(items))) == 1)

        #Start From Each Item at Most Once
        for i in items:
            m.addConstr(sum(x[i,j] for j in items) <= 1)

        #End at Each Item at Most Once
        for j in items:
            m.addConstr(sum(x[i,j] for i in items) <= 1)

        #Cannot go From Item I to Item I
        for i in items:
            m.addConstr(x[i,i] == 0)

        #Capacity of 15
        m.addConstr(sum(x[i,j] for i in items for j in range(1,len(items))) <= capacity)

        #Time Constraint
        m.addConstr(sum(x[i,j] * distances[i][j] for i in items for j in items)  
                    + (2* sum(x[i,j] for i in items for j in range(1,len(items)))) <= time)

        #If traveled to, must leave from
        for k in items:
            m.addConstr(sum(x[i,k] for i in items) == sum(x[k,j] for j in items))

        #Subtour Elimination
        if len(subtour_storage) > 0:
            for i in range(len(subtour_storage)):
                m.addConstr(sum(x[subtour_storage[i][j][0],subtour_storage[i][j][1]] 
                                for j in range(len(subtour_storage[i]))) <= len(subtour_storage[i])-1)
        
        #Solve
        objective = sum(x[i,j] * prices[j] for i in items for j in range(1,len(items)))
        m.setObjective(objective, GRB.MAXIMIZE)
        m.optimize()

        # Print Optimal Solution
        print('Optimal Value: ', round(m.objVal,2))
        obj_val = round(m.objVal,2)
        solution =[]
        for v in m.getVars():
            solution.append((v.varName, v.x))

        #Algorithm

        # arcs = ['x[0,22]', 'x[22,0]']
        arcs = [i[0] for i in solution if round(i[1]) == 1]

        # start_end = [[0, 22], [22, 0]]       
        start_end = [[int(num) for num in arc.strip('x[]').split(',')] for arc in arcs]
        
        # all_nodes = [0, 22]
        all_nodes = list(set(node for pair in start_end for node in pair))

        # node_dict {0: 22, 22: 0}
        node_dict = {i[0]: i[1] for i in start_end}

        print("All Nodes:", all_nodes)
        print("Start End:", start_end)
        #print("Node Dict:", node_dict)

        list_of_subtours = []
        if node_dict:
            for start in node_dict:
                if any(start in subtour for subtour in list_of_subtours):
                    continue
                subtour = [start]
                end = node_dict[start]
                while end != start and end in node_dict:
                    subtour.append(end)
                    end = node_dict[end]
                # [[0, 2, 32, 22, 0], [25, 56, 54, 33, 27, 25]]
                list_of_subtours.append(subtour + [start])                
        print(f'List of Subtours {list_of_subtours}')
        full_tour = list_of_subtours[0]
        subtours_cleaned = []
        for i in list_of_subtours:
            subtours_cleaned.append(i[:-1])

        switch = False

        def compute_tour_value(tour_indices, prices):
            return sum(prices[j] for j in tour_indices if j < len(prices))

        value_of_tour_from_start = round(sum(compute_tour_value(tour_indices, prices) for tour_indices in subtours_cleaned if 0 in tour_indices),2)

        print("Last Solved Solution : ", last_solved_solution)
        print("Value of Start Node Tour: ", value_of_tour_from_start)
        
        #Subtour_Eliminator
        results_df, switch = Subtour_Eliminator(time, capacity, results_df, obj_val,subtours_cleaned, value_of_tour_from_start, switch, full_tour,item_list, last_solved_solution)

        return obj_val, subtours_cleaned, start_end, switch, solution,results_df,last_solved_solution 

if __name__ == '__main__':
    '''
    To change the time or capacity parameters, You need to set one to be static and the other to loop.
    IE: if we want to loop capcity for 1 to 15 we would have Capcity for the top loop
    for range(1,15) and then Time would be the lower For loop with for Time in [90]

    In the case we want to model Time we would flip the variables and do for Time in range(5,90+1)
    and then have for Capcacity in [15].   
    '''
    if not os.path.exists('Value data'):
        os.makedirs('Value data')
    master_list, distance_list, item_list= read_data()
    prices,items,distances = parameter_maker(master_list , distance_list)
    scenario = 0
    results_df = pd.DataFrame(columns=['Time', 'Capacity', 'Objective Value', 'Tour'])
    for Capacity in range(1,15+1):
        for Time in [90]:
            scenario += 1
            print(f'scenario{scenario}')
            if scenario == 1: 
                last_solved_solution = 0
            
            subtour_storage = []
            list_of_pairs = []
            iteration = 0
            obj_val, subtours_cleaned, start_end, switch, solution,results_df, last_solved_solution = supermarket_model(prices,items,distances, Time, Capacity, iteration, results_df, item_list, last_solved_solution)
            
            while switch == True:
                obj_val, subtours_cleaned, start_end, switch, solution,results_df, last_solved_solution = supermarket_model(prices,items,distances, Time, Capacity, iteration,results_df, item_list, last_solved_solution)
                iteration +=1
                print("iteration #: ",iteration)
                
            last_solved_solution  = round(obj_val,2)

print(results_df.iloc[-1])
print(results_df.columns)




    


