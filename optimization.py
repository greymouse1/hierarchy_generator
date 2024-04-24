from gurobipy import *
import networkx as nx
from tqdm import tqdm
import pickle


# Load graph from file
weighted_graph = nx.read_graphml("jaccard_index.graphml")

# Load trees
with open("T1.pkl", "rb") as f:
    T1 = pickle.load(f)
with open("T2.pkl", "rb") as f:
    T2 = pickle.load(f)

def optimization(weighted_graph,T1,T2):
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
    x = optimization_model.addVars(len(all_edges), vtype=GRB.CONTINUOUS ,lb=0, ub=1, name="x")

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
        elif v.x == 0:
            matched_edge_id = (int(v.varName.split('[')[1].split(']')[0]))
            zero_edges[all_edges[matched_edge_id]] = v.x


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

    # This part is printing m:n matches
    #for first,second in matched_edges:
        #m = number_of_leaf_nodes(T1.G,first)
        #n = number_of_leaf_nodes(T2.G,second)
        #print(f"Edge {first}:{second} is a {m}:{n} match")

    return zero_edges, matched_edges

# Matching algorithm
def matching(weighted_graph, T1, T2):
    zero_edges, matched_edges = optimization(weighted_graph,T1,T2)
    print("Number of zero edges is", len(zero_edges))
    print("Number of matched edges is", len(matched_edges))

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
        # For T2
        nodes_in_T2 = []
        node_in_T2 = edge[1]
        ancestors_T2 = nx.ancestors(T2.G, node_in_T2)
        descendants_in_T2 = nx.descendants(T2.G, node_in_T2)
        nodes_in_T2.extend(ancestors_T2)
        nodes_in_T2.extend(descendants_in_T2)
        # Holders for sum (of decision variable x values)
        total_sum = 0
        # Detected edges are edges which are found to be in conflict with current edge, and they
        # belong to the latest solution ie they have some value for decision variable x
        detected_edges= []
        for start_node in nodes_in_T1:
            for end_node in nodes_in_T2:
                if (start_node,end_node) in all_edges:
                    total_sum = total_sum + all_edges[(start_node,end_node)]
                    detected_edges.append((start_node,end_node))
        if not detected_edges:
            continue
        if total_sum <= 3: # This is value alpha = 3
            shifted_graph = weighted_graph.copy()
            weight_of_current_edge = weighted_graph.edges[edge]['weight']
            for detected_edge in detected_edges:
                weight_of_detected_edge = weighted_graph.edges[detected_edge]['weight']
                new_weight = weight_of_detected_edge - weight_of_current_edge
                nx.set_edge_attributes(shifted_graph, {detected_edge: new_weight}, 'weight')
            print(f"Edges in conflict with edge {edge} are edges: {detected_edges}")
            print("Total sum is ", total_sum)
            m = matching(shifted_graph, T1, T2)
            if not set(detected_edges).intersection(m):
                if edge not in m:
                    print(f"Extending M with edge {edge}")
                    m.append(edge)
            return m

    #m = matching(weighted_graph, T1, T2)
    print("Final matching line reached, returning empty list")
    return []




final_result = matching(weighted_graph,T1,T2)

print(final_result)
print("Number of matches is ",len(final_result))