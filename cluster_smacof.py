import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import matplotlib.pyplot as plt
import time

print("imports done")

df = pd.read_excel('/Users/eric/Downloads/house_data.xlsx')

print("loaded connections")

Connections = df.to_numpy()

Connections_log = np.log(1+Connections)

number_of_nodes = len(Connections)

Dissimilarities = np.max(Connections_log)-Connections_log

Weights = Connections_log**2

print("finished initial calculations")

def _calculate_stress(D_target, W, X):
    D_embedding = squareform(pdist(X, 'euclidean'))
    difference = D_target - D_embedding
    stress = np.sum(W * difference*difference)
    return stress

def weighted_smacof(D_target, W, X_init, n_components=2, max_iter=100, eps=0.00000001, random_state=None):


    N = D_target.shape[0]
    
    if X_init is None:
        X = np.random.rand(N, n_components)*2
    else:
        X = X_init.copy()

    np.fill_diagonal(W, 0)
    np.fill_diagonal(D_target, 0)

    V = -W.copy()
    np.fill_diagonal(V, np.sum(W,axis=1))
    
    try:
        V_inv = np.linalg.pinv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V + 1e-8 * np.eye(N))

    D_X = squareform(pdist(X, 'euclidean'))

    for iteration in range(max_iter):
        old_DX = np.sum(D_X)

        ratio = np.zeros_like(D_X)
        non_zero_dist = D_X > 1e-12
        ratio[non_zero_dist] = D_target[non_zero_dist] / D_X[non_zero_dist]

        B = -(W * ratio)
        
        diag_B = np.sum(-B, axis=1)
        np.fill_diagonal(B, diag_B)
        
        X = V_inv @ B @ X
        
        D_X = squareform(pdist(X, 'euclidean'))
        
        if iteration >= 10:
            if abs(1- np.sum(D_X)/old_DX) < eps:
                break

    return X

def clusters_to_matrix(cluster_dict: dict, distance_matrix, weight_matrix) -> np.ndarray:

    if not cluster_dict:
        # Handle empty dictionary case
        num_clusters = 0
        num_nodes = distance_matrix.shape[0]
        return np.zeros((0, 0)), np.zeros((0, 0))
    
    node_ids = np.array(list(cluster_dict.keys()))
    cluster_ids = np.array(list(cluster_dict.values()))
    
    num_clusters = np.max(cluster_ids) + 1
    num_nodes = distance_matrix.shape[0] 

    cluster_map_matrix = np.zeros((num_clusters, num_nodes), dtype=np.int8)
    cluster_map_matrix[cluster_ids, node_ids] = 1
    
    weight_agg = cluster_map_matrix @ weight_matrix @ cluster_map_matrix.T
    distance_sum_agg = cluster_map_matrix @ distance_matrix @ cluster_map_matrix.T

    cluster_sizes = cluster_map_matrix.sum(axis=1)
    pair_counts = np.outer(cluster_sizes, cluster_sizes)

    distance_agg = np.divide(distance_sum_agg, pair_counts,
                             out=np.zeros_like(distance_sum_agg, dtype=float),
                             where=pair_counts != 0)
    print(distance_sum_agg)

    return weight_agg, distance_agg

def next_positions(current_pos, current_partition):
    
    max_index = max(current_partition.keys())
    numpy_array = np.zeros((max_index + 1,2))

    for index in range(max_index):
        numpy_array[index] = current_pos[current_partition[index]]

    return numpy_array

def hierarchical_smacof_layout(adjacency_matrix, weight_matrix, n_components=2):

    Connections_zscore = 2**(zscore(adjacency_matrix, axis=1))
    
    D_orig = np.asarray(adjacency_matrix).astype(float)
    W_orig = np.asarray(weight_matrix).astype(float)

    G = nx.from_numpy_array(Connections_zscore)
    partitions = community_louvain.best_partition(G, resolution=reso)

    W_agg, D_agg = clusters_to_matrix(partitions, D_orig, W_orig)

    X_init = None

    X_current = weighted_smacof(
        D_agg, W_agg, X_init, n_components=n_components, 
        max_iter=150, eps=0.000001)

    X_init = next_positions(X_current, partitions)

    X_final = weighted_smacof(
        D_orig, W_orig, X_init, n_components=n_components, 
        max_iter=600, eps=0.00000001)

    return X_final, X_current

def plot_network(node_positions, connections, title, connection_threshold=0, node_colors=None, node_size=100):
   
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Plot connections (edges) - only those above threshold
    n_nodes = len(node_positions)
    max_connection = np.max(connections)
    
    # Store connection lines in a dictionary for interactivity
    connection_lines = {}
    
    # Count connections above threshold for reporting
    connections_shown = 0
    total_possible_connections = 0
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Only process upper triangle of connection matrix
            total_possible_connections += 1
            if connections[i, j] > connection_threshold:
                connections_shown += 1
                
                # Calculate normalized connection strength (0 to 1)
                normalized_strength = connections[i, j] / max_connection
                
                # Width of the line is based on connection strength
                width = normalized_strength
                
                # Alpha (opacity) is also based on connection strength
                # Scale from 0.2 (min) to 0.9 (max) to ensure visibility
                alpha = 0.2 + (normalized_strength * 0.7)
                
                line = ax.plot([node_positions[i, 0], node_positions[j, 0]],
                         [node_positions[i, 1], node_positions[j, 1]],
                         'gray', linewidth=width, alpha=alpha)[0]
                
                # Store the line with the node pair as key
                connection_lines[(i, j)] = {
                    'line': line,
                    'strength': connections[i, j],
                    'normalized_strength': normalized_strength
                }
    
    # Set default node colors if not provided
    if node_colors is None:
        node_colors = 'skyblue'
    
    # Plot nodes with custom colors and make them pickable
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1], 
                c=node_colors, 
                s=node_size, 
                edgecolor='black', 
                zorder=10,
                picker=True,  # Enable picking
                pickradius=5)  # Picking radius in points
    
    # Dictionary to store node labels
    node_labels = {}
    
    # If there are many nodes, skip labels to improve performance
    if n_nodes <= 50:  # Only label if fewer than 50 nodes
        for i, (x, y) in enumerate(node_positions):
            label = ax.text(x, y, f"{i+1}", fontsize=8, ha='center', va='center')
            node_labels[i] = label
    
    # Update title to include information about connections shown
    if connection_threshold > 0:
        title = f"{title}\n(Showing {connections_shown}/{total_possible_connections} connections, threshold > {connection_threshold})"
    
    orig_title = title
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')

    # Create an annotation object for showing connection strength (initially hidden)
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), 
                       textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                       arrowprops=dict(arrowstyle="->"), visible=False)
    
    return fig, ax

def load_node_colors_from_excel(filepath, column_name=None, sheet_name=0):
    """
    Load node colors from an Excel file.
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file containing node colors
    column_name : str, optional
        Name of the column containing color values. If None, the first column is used.
    sheet_name : str or int, default=0
        Sheet name or index to read from
        
    Returns:
    --------
    list
        List of colors for each node
    """
    try:
        # Load the Excel file
        color_df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # If column name is not specified, use the first column
        if column_name is None:
            column_name = color_df.columns[0]
            print(f"Using '{column_name}' as the color column")
            
        # Extract colors from the specified column
        node_colors = color_df[column_name].tolist()
        
        # Check if we have enough colors for all nodes
        if len(node_colors) < number_of_nodes:
            print(f"Warning: Excel file contains {len(node_colors)} colors, but there are {number_of_nodes} nodes.")
            # Repeat colors if there aren't enough
            node_colors = (node_colors * ((number_of_nodes // len(node_colors)) + 1))[:number_of_nodes]
        elif len(node_colors) > number_of_nodes:
            print(f"Warning: Excel file contains {len(node_colors)} colors, but only {number_of_nodes} will be used.")
            node_colors = node_colors[:number_of_nodes]
            
        return node_colors
    
    except Exception as e:
        print(f"Error loading colors from Excel: {e}")
        print("Using default colors instead")
        return 'skyblue'  # Return default color if there's an error
    
reso = 0.7

test = 0

graph = 1

if test == 1:
    for j in range(1,20):
        reso = j/10
        times = 0
        stresses = 0

        for i in range(1,10):

            start_time = time.time()

            node_locations, b = hierarchical_smacof_layout(Dissimilarities, Weights)

            end_time = time.time()

            times = times + end_time-start_time

            final_stress = _calculate_stress(Dissimilarities, Weights, node_locations)

            stresses = stresses+ final_stress

        print(f"reso: {reso} Time: {times/10} Stress: {final_stress/10}")

if graph == 1:

    start_time = time.time()

    node_locations, node_precursor_loc = hierarchical_smacof_layout(Dissimilarities, Weights)

    end_time = time.time()

    final_stress = _calculate_stress(Dissimilarities, Weights, node_locations)

    print(f"completed in {end_time - start_time:.4f} seconds.")

    print("final stress:", final_stress)

    colors_excel_path = '/Users/eric/Downloads/Party_data.xlsx'
    colors = load_node_colors_from_excel(colors_excel_path)

    percentile_value = np.percentile(Connections, 97)

    Connections_sums = 1.2**(np.sum(Connections, axis=1)/number_of_nodes)*2.5

    fig, ax = plot_network(
        node_locations, 
        Connections, 
        "Node Configuration", 
        connection_threshold=percentile_value,
        node_colors=colors,
        node_size=Connections_sums)
    
    fig2 = ax.scatter(node_precursor_loc[:, 0], node_precursor_loc[:, 1], c='yellow')

    plt.show()

    