from gurobipy import *
import networkx as nx
from tqdm import tqdm
import pickle
import os
import json
import time
import datetime
import glob


# Load graph from file
weighted_graph = nx.read_graphml("batch/auerberg_alex/jaccard_index_auerberg_alex.graphml")

# Load trees
with open("batch/auerberg_alex/T1_auerberg_atkis_alex.pkl", "rb") as f:
    T1 = pickle.load(f)
with open("batch/auerberg_alex/T2_auerberg_osm_alex.pkl", "rb") as f:
    T2 = pickle.load(f)

# Select whether you use ILP or LP. For ILP input GRB.INTEGER, for LP input GRB.CONTINUOUS
def optimization(weighted_graph,T1,T2,program_type):
    # Run Gurobi optimization

    # Get the set of all edges
    all_edges = list(weighted_graph.edges())
    # Create a dictionary to map edges to their indexes
    edge_to_index = {edge: i for i, edge in enumerate(all_edges)}

    # Create the model
    optimization_model = Model("bipartite solver")

    # Add decision variables x.
    # If ILP, then vtype=GRB.BINARY and no lb,ub
    # IF LP, then vtype=GRB.CONTINUOUS and lb=0,ub=1
    x = optimization_model.addVars(len(all_edges), vtype=program_type ,lb=0, ub=1, name="x")

    # Define objective function
    obj_function = LinExpr([weighted_graph.edges[edge]['weight'] for edge in all_edges], x.values())
    optimization_model.setObjective(obj_function, GRB.MAXIMIZE)

    # Set constraints for T1
    for path in tqdm(T1.leaf_list, desc=f"Setting constraints for {T1}"):
        indexes = [edge_to_index[edge] for edge in weighted_graph.edges(path)]
        constraint_sum = LinExpr([1] * len(indexes), [x[i] for i in indexes])
        optimization_model.addConstr(constraint_sum <= 1)

    # Set constraints for T2
    for path in tqdm(T2.leaf_list, desc=f"Setting constraints for {T2}"):
        indexes = [edge_to_index[edge[::-1]] for edge in weighted_graph.edges(path)]
        constraint_sum = LinExpr([1] * len(indexes), [x[i] for i in indexes])
        optimization_model.addConstr(constraint_sum <= 1)

    # Optimize
    optimization_model.optimize()

    is_set_empty = True

    # Dictionary holding edge ID and value of decision variable for cases where x > 0
    matched_edges = {}

    # Dictionary holding edges with decision variable value 0
    zero_edges = {}

    for v in optimization_model.getVars():
        if v.x > 0:
            #print( '%s: %g ' % (v.varName,v.x))
            matched_edge_id = (int(v.varName.split('[')[1].split(']')[0]))
            matched_edges[all_edges[matched_edge_id]] = v.x
            print(f"Decision variable is {v.x}")
        elif v.x == 0:
            matched_edge_id = (int(v.varName.split('[')[1].split(']')[0]))
            zero_edges[all_edges[matched_edge_id]] = v.x

    '''
    # Define function for calculating number of leafs
    def number_of_leaf_nodes(tree, node):
        descendants = nx.descendants(tree, node)
        leaf_nodes = [n for n in descendants if tree.out_degree(n) == 0]
        # this is because amount of leaves will be either 0 which means the match is
        # one of the base polygons, or 2 or more. It can't be 1 because parent node
        # is by default made by merging two or more polygons
        if len(leaf_nodes) >= 2:
            return len(leaf_nodes)
        else:
            return 1
    '''
    # This part is printing m:n matches.
    # It iterates through keys while first,second are unpacking the tupples
    #for first,second in matched_edges:
        #m = number_of_leaf_nodes(T1.G,first)
        #n = number_of_leaf_nodes(T2.G,second)
        #print(f"Edge {first}:{second} is a {m}:{n} match")

    return zero_edges, matched_edges

# Canzar's Matching algorithm
def matching(weighted_graph, T1, T2, lambda_ = 0):

    # Lambda is optional argument and a constant which is subtracted from weights
    # in order to try to impact behaviour of linear program
    if lambda_ != 0:
        for u, v, data in weighted_graph.edges(data=True):
            data['weight'] -= lambda_
        print(f"All edges in initial graph have subtracted lambda value {lambda_}.")
    # Run LP
    zero_edges, matched_edges = optimization(weighted_graph,T1,T2, GRB.CONTINUOUS)
    print("Number of zero edges is", len(zero_edges))
    print("Number of matched edges is", len(matched_edges))
    print("Matched edges are:")
    print(matched_edges)

    all_edges = zero_edges.copy()  # Create a copy of dict1
    all_edges.update(matched_edges)
    if zero_edges:
        reduced_graph = weighted_graph.copy()
        reduced_graph.remove_edges_from(zero_edges.keys())
        print("Number of edges in input graph is", weighted_graph.number_of_edges())
        print("Number of edges in reduced graph is", reduced_graph.number_of_edges())
        m = matching(reduced_graph, T1, T2)
        return m
    # Check for sum of decision variable values for edges which are incident to
    # nodes which belong to paths to which edges node of current edge belong to
    for edge in weighted_graph.edges():
        print("Current edge in the weighted graph is ",edge)
        # For T1
        nodes_in_T1 = []
        node_in_T1 = edge[0]
        ancestors_T1 = nx.ancestors(T1.G,node_in_T1)
        descendants_in_T1 = nx.descendants(T1.G,node_in_T1)
        nodes_in_T1.extend(ancestors_T1)
        nodes_in_T1.extend(descendants_in_T1)
        nodes_in_T1.append(node_in_T1)
        # For T2
        nodes_in_T2 = []
        node_in_T2 = edge[1]
        ancestors_T2 = nx.ancestors(T2.G, node_in_T2)
        descendants_in_T2 = nx.descendants(T2.G, node_in_T2)
        nodes_in_T2.extend(ancestors_T2)
        nodes_in_T2.extend(descendants_in_T2)
        nodes_in_T2.append(node_in_T2)
        # Additonal edges which can't be detected below
        # Extract edges with the specified starting node
        #edges_starting_from_node = [v for (u, v) in all_edges if u == node_in_T1]
        #nodes_in_T2.extend(edges_starting_from_node)
        # Clean duplicates
        #nodes_in_T2 = list(set(nodes_in_T2))
        # Extract edges with the specified ending node
        #edges_ending_at_node = [u for (u, v) in all_edges if v == node_in_T2]
        #nodes_in_T1.extend(edges_ending_at_node)
        # Clean duplicates
        #nodes_in_T1 = list(set(nodes_in_T1))
        # Holders for sum (of decision variable x values)
        total_sum = 0
        # Detected edges are edges which are found to be in conflict with current edge, and they
        # belong to the latest solution ie they have some value for decision variable x
        detected_edges= []
        '''
        for start_node in nodes_in_T1:
            for end_node in nodes_in_T2:
                if (start_node,end_node) == edge:
                    continue
                print("Current edge checked",(start_node,end_node) )
                if (start_node,end_node) in all_edges:
                    total_sum = total_sum + all_edges[(start_node,end_node)]
                    detected_edges.append((start_node,end_node))
        '''
        for (u,v) in all_edges:
            if u in nodes_in_T1:
                detected_edges.append((u,v))
            if v in nodes_in_T2:
                detected_edges.append((u,v))
            detected_edges = list(set(detected_edges))
            if edge in detected_edges:
                detected_edges.remove(edge)

        # Clean duplicates one more time
        detected_edges = list(set(detected_edges))

        # Update the sum with weights of the conflicting edges
        for (u,v) in detected_edges:
            total_sum = total_sum + all_edges[(u,v)]

        if total_sum <= 3 and total_sum > 0: # This is value alpha = 3
            shifted_graph = weighted_graph.copy()
            weight_of_current_edge = weighted_graph.edges[edge]['weight']
            for detected_edge in detected_edges:
                weight_of_detected_edge = weighted_graph.edges[detected_edge]['weight']
                new_weight = weight_of_detected_edge - weight_of_current_edge
                nx.set_edge_attributes(shifted_graph, {detected_edge: new_weight}, 'weight')
            print(f"Edges in conflict with edge {edge} are edges: {detected_edges}")
            print("Decision variables sum for conflicting edges is ", total_sum)
            #if total_sum == 0:
            # * is used to unpack a tupple
            #shifted_graph.remove_edge(*edge)
            m = matching(shifted_graph, T1, T2)
            if not set(detected_edges).intersection(m) and edge not in m:
                #if edge not in m:
                print(f"Extending M with edge {edge}")
                m.append(edge)
            return m
        #m = matching(weighted_graph, T1, T2)
        else:
            print("Final matching line reached.")
            mylist=[]
            for edge in weighted_graph.edges():
                if all_edges[edge] == 1:
                    mylist.append(edge)
                    print(all_edges[edge])
                    print("Appended edge of x=1, this is edge",edge)
            return mylist




# Function for manually calculating value of objective function from Canzar output
def objFunctVal(inputGraph,resultingEdges):
    for (u,v) in resultingEdges:
        weight= inputGraph[u][v]['weight']
        print(weight)
    objFunct = sum([inputGraph[u][v]['weight'] for (u,v) in resultingEdges])
    return objFunct

'''
# Run Canzar separately
# Arguments are graph, tree 1, tree 2 and lambda (optional, default is 0)
final_result_canzar = matching(weighted_graph,T1,T2)
print(final_result_canzar)
print("Number of matches is ",len(final_result_canzar))
print("Value of objective function is ", objFunctVal(weighted_graph,final_result_canzar))
'''

'''
# Run ILP separately
final_result_ilp = optimization(weighted_graph,T1,T2,GRB.INTEGER)
print(final_result_ilp[1])
print("Number of matches is ", len(final_result_ilp[1]))
'''

'''
# This part is used for saving matched nodes in two separate shp files
# only in the case of using ILP or Canzar separately
# For using ILP and Canzar in one go, use the code after and outside this comment

# Get a list of all matched nodes from T1 and T2
matched_nodes_T1 = []
matched_nodes_T2 = []

# If using final_result_canzar, loop through final_result_canzar list, each item is a tupple with edge (two nodes)
# If using final_result_ilp, loop through final_result_ilp[1].keys():
for edge in final_result_canzar:
    u = edge[0]
    v = edge[1]
    matched_nodes_T1.append(u)
    matched_nodes_T2.append(v)


# Get a list of all matched nodes from T2
T1.saveShpGrouped(matched_nodes_T1)
T2.saveShpGrouped(matched_nodes_T2)
'''
# Run ILP and Canzar at the same time
# -one lambda value for both algorithms can be chosen, or leave it out - default value is 0
# Results of both algorithms will be displayed in the end for comparison
# shp files are save for
# -ILP tree 1 matched nodes
# -ILP tree 2 matched nodes
# -Canzar tree 1 matched nodes
# -Canzar tree 2 matched nodes
def runBothAlgorithms(bipartiteGraph, tree1, tree2, name , lambda_=0):
    # This dictionary will hold final results in a format
    # bothResults = ILP:(obj_f, [matched_edges]),CANZAR:(obj_f, [matched_edges])
    bothResults = {}
    # First important thing is that weights in bipartite graph are adjusted by the lambda value
    # This is done before optimisation is ran so input graph has weights already adjusted
    if lambda_ != 0:
        for u, v, data in bipartiteGraph.edges(data=True):
            data['weight'] -= lambda_
        print(f"All edges in initial graph have subtracted lambda value {lambda_}.")
    # Now that graphs are adjusted, the ILP can be ran.
    # Run timer for ILP
    start_time_ilp = time.time()
    final_result_ilp = optimization(bipartiteGraph, tree1, tree2, GRB.INTEGER)
    end_time_ilp = time.time()
    total_time_ilp = end_time_ilp-start_time_ilp
    matched_nodes_T1_ILP = []
    matched_nodes_T2_ILP = []
    for edge in final_result_ilp[1].keys():
        u = edge[0]
        v = edge[1]
        matched_nodes_T1_ILP.append(u)
        matched_nodes_T2_ILP.append(v)

    # Value of the objective function is printed by default by Gurobi prompt
    # Since I can't catch that variable, I go indirectly and pull weights from the bipartite graph with
    # the results of matching tupples
    objValueILP = objFunctVal(bipartiteGraph, final_result_ilp[1].keys())

    # Store values in bothResults variable so it is possible to show it in the end
    bothResults['ILP'] = (objValueILP,final_result_ilp[1].keys())

    # Create folder where shp files will be saved

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a new directory with the format "optimization_timestamp"
    dir_name = f'{name}_{timestamp}'
    os.makedirs(dir_name)

    # Get the current working directory
    current_directory = os.getcwd()
    current_directory = os.path.join("optimizations/")
    timestamped_directory = os.path.join(current_directory,dir_name)

    # Define the name of the new directory to be created
    new_directory_ILP = 'shp_files_ILP'

    # Create the full path to the new directory
    output_directory_ILP = os.path.join(timestamped_directory, new_directory_ILP)

    # Create the new directory if it doesn't exist
    os.makedirs(output_directory_ILP, exist_ok=True)

    # Save SHP files for matched nodes in the specified directory, format is of "list,directory_name"
    T1.saveShpGrouped(matched_nodes_T1_ILP,output_directory_ILP)
    T2.saveShpGrouped(matched_nodes_T2_ILP,output_directory_ILP)


    # Now run Canzar approximation algorithm
    # Time Canzar
    start_time_canzar = time.time()
    final_result_canzar = matching(bipartiteGraph, tree1, tree2)
    end_time_canzar = time.time()
    total_time_canzar = end_time_canzar-start_time_canzar
    matched_nodes_T1_CANZAR = []
    matched_nodes_T2_CANZAR = []
    for edge in final_result_canzar:
        u = edge[0]
        v = edge[1]
        matched_nodes_T1_CANZAR.append(u)
        matched_nodes_T2_CANZAR.append(v)

    # Value of the objective function is printed by default by Gurobi prompt
    # Since I can't catch that variable, I go indirectly and pull weights from the bipartite graph with
    # the results of matching tupples
    objValueCANZAR = objFunctVal(bipartiteGraph, final_result_canzar)

    # Store values in bothResults variable so it is possible to show it in the end
    bothResults['CANZAR'] = (objValueCANZAR, final_result_canzar)

    # Define the name of the new directory to be created
    new_directory_CANZAR = 'shp_files_CANZAR'

    # Create the full path to the new directory
    output_directory_CANZAR = os.path.join(timestamped_directory, new_directory_CANZAR)

    # Create the new directory if it doesn't exist
    os.makedirs(output_directory_CANZAR, exist_ok=True)

    # Save SHP files for matched nodes in the specified directory, format is of "list,directory_name"
    T1.saveShpGrouped(matched_nodes_T1_CANZAR, output_directory_CANZAR)
    T2.saveShpGrouped(matched_nodes_T2_CANZAR, output_directory_CANZAR)

    print(f"Value of objective function of ILP is {bothResults['ILP'][0]}")
    print(f"Value of objective function of CANZAR is {bothResults['CANZAR'][0]}")
    print(f"Time for running ILP is {total_time_ilp}")
    print(f"Time for running Canzar is {total_time_canzar}")
    print(f"Number of matched edges in ILP is {len(bothResults['ILP'][1])}")
    print(f"Number of matched edges in CANZAR is {len(bothResults['CANZAR'][1])}")
    #print(f"Matched edges of ILP are {bothResults['ILP'][1]}")
    #print(f"Matched edges of CANZAR are {bothResults['CANZAR'][1]}")

    # Create the output file name within the new directory
    report_file = os.path.join(timestamped_directory, 'report.txt')

    # Save report
    with open(report_file, 'w') as file:
        file.write(f"Value of objective function of ILP is {bothResults['ILP'][0]}\n")
        file.write(f"Value of objective function of CANZAR is {bothResults['CANZAR'][0]}\n")
        file.write(f"Time for running ILP is {total_time_ilp}\n")
        file.write(f"Time for running Canzar is {total_time_canzar}\n")
        file.write(f"Number of matched edges in ILP is {len(bothResults['ILP'][1])}\n")
        file.write(f"Number of matched edges in CANZAR is {len(bothResults['CANZAR'][1])}\n")
    '''
    Writing all edges and leafs to a file
    '''

    # Save detailed file about matches
    def number_of_leaf_nodes(tree, node):
        descendants = nx.descendants(tree, node)
        leaf_nodes = [n for n in descendants if tree.out_degree(n) == 0]
        # this is because amount of leaves will be either 0 which means the match is
        # one of the base polygons, or 2 or more. It can't be 1 because parent node
        # is by default made by merging two or more polygons
        if len(leaf_nodes) >= 2:
            return len(leaf_nodes), leaf_nodes
        else:
            return 1, []

    def writer_function(matched_edges):
        # Total number of matched base polygons is total number of matched leaves
        total_matched_T1 = 0
        total_matched_T2 = 0
        # List for all matches and construction of json template
        matches = []
        # Iterate through keys while first, second are unpacking the tuples
        for first, second in matched_edges[1]:
            m, leaf_nodes_first = number_of_leaf_nodes(T1.G, first)
            n, leaf_nodes_second = number_of_leaf_nodes(T2.G, second)

            # Update the counter
            total_matched_T1 += m
            total_matched_T2 += n

            # Create a match dictionary
            match = {
                "edge": [first, second],
                "match": f"{m}:{n}",
                "leaf_nodes_first": leaf_nodes_first,
                "leaf_nodes_second": leaf_nodes_second
            }

            matches.append(match)

        # Create a header
        total_pol_T1 = len(T1.leaf_list)
        total_pol_T2 = len(T2.leaf_list)

        print(f"There are {total_matched_T1} matched polygons out of {total_pol_T1} in T1."
              f" This means that a total of {(total_matched_T1 / total_pol_T1) * 100}% of polygons in this dataset are matched. " )
        print(f"There are {total_matched_T2} matched polygons out of {total_pol_T2} in T2."
              f" This means that a total of {(total_matched_T2 / total_pol_T2) * 100}% of polygons in this dataset are matched. ")

        header = {
            "number of matched edges":  len(matched_edges[1]),
            "value_of_objective_function": matched_edges[0],
            "n_matched_polygons_T1": total_matched_T1,
            "n_total_polygons_T1": total_pol_T1,
            "percentage_matched_polygons_T1": (total_matched_T1/total_pol_T1 * 100),
            "n_matched_polygons_T2": total_matched_T2,
            "n_total_polygons_T2": total_pol_T2,
            "percentage_matched_polygons_T2": (total_matched_T2 / total_pol_T2 * 100)
        }
        matches.insert(0, header)
        # Save as json
        # Write matches list to the JSON file
        json.dump(matches, file, indent=4)

    # Open a file in write mode
    filepath_ilp = os.path.join(timestamped_directory, "matches_ILP.json")
    with open(filepath_ilp, "w") as file:
        # Write matches from ILP
        writer_function(bothResults['ILP'])
    filepath_canzar = os.path.join(timestamped_directory, "matches_canzar.json")
    with open(filepath_canzar, "w") as file:
        # Write matches from Canzar
        writer_function(bothResults['CANZAR'])
    '''
    End of writing of edges to a file
    '''
    # Convert lists to sets
    set1 = set(bothResults['ILP'][1])
    set2 = set(bothResults['CANZAR'][1])

    # Check if objective function value for ILP and CANZAR is equal
    if {bothResults['ILP'][0]} == {bothResults['CANZAR'][0]}:
        print("100% match of edges between ILP and Canzar approximation algorithm.")
        with open(report_file, 'a') as file:
            file.write("100% match of edges between ILP and Canzar approximation algorithm.\n")
    else:
        # Find common edges
        common_edges = set1 & set2
        print(f"Edges present in both matchings ({len(common_edges)}): {common_edges}")

        # Find unique edges in the first list
        unique_to_list1 = set1 - set2
        print(f"Edges present only in the ILP ({len(unique_to_list1)}): {unique_to_list1}")

        # Find unique edges in the second list
        unique_to_list2 = set2 - set1
        print(f"Edges present only in the CANZAR ({len(unique_to_list2)}): {unique_to_list2}")

        # Find percentage of how close is Canzar to ILP
        percentage_of_overlap = (bothResults['CANZAR'][0] / bothResults['ILP'][0]) * 100

        with open(report_file, 'a') as file:
            file.write(f"Edges present in both matchings ({len(common_edges)}): {common_edges}\n")
            file.write(f"Edges present only in the ILP ({len(unique_to_list1)}): {unique_to_list1}\n")
            file.write(f"Edges present only in the CANZAR ({len(unique_to_list2)}): {unique_to_list2}\n")
            file.write(f"Percentage of match between CANZAR and ILP (fCANZAR / fILP) is {percentage_of_overlap}\n")

# Lambda value is optional as the fourth argument
runBothAlgorithms(weighted_graph,T1,T2,"auerberg_alex", )


'''
# Batch processing
# Define the parent directory containing the folders
parent_directory = 'batch'

# Iterate over each folder in the parent directory
for folder_name in os.listdir(parent_directory):
    print(f"Current dataset is {folder_name}")
    folder_path = os.path.join(parent_directory, folder_name)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Find the graphml file
        graphml_file = glob.glob(os.path.join(folder_path, '*.graphml'))[0]

        # Find the pkl files
        pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

        # Assign files to variables based on their names
        for pkl_file in pkl_files:
            if 'T1' in os.path.basename(pkl_file):
                T1_file = pkl_file
            elif 'T2' in os.path.basename(pkl_file):
                T2_file = pkl_file

        # Load graph from file
        weighted_graph = nx.read_graphml(graphml_file)

        # Load trees
        with open(T1_file, "rb") as f:
            T1 = pickle.load(f)
        with open(T2_file, "rb") as f:
            T2 = pickle.load(f)

        # Here you can use weighted_graph, T1, and T2 as needed
        print(f'Processed folder: {folder_name}')
        print(f'GraphML file: {graphml_file}')
        print(f'T1 file: {T1_file}')
        print(f'T2 file: {T2_file}')

        # Run code for all values of lambda
        lambda_list = [0,0.1,0.2,0.3]

        for current_lambda in lambda_list:
            print(f"Current lambda value is {current_lambda}")
            runBothAlgorithms(weighted_graph, T1, T2, name=f"{folder_name}_{current_lambda}", lambda_=current_lambda)

'''


