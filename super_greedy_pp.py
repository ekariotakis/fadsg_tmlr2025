import networkx as nx
import copy

### based on NetworkX implementation for DSG
def super_greedy_pp(G, iterations, protected_nodes_all, lam, weight=None, formulation=1):
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    loads = dict.fromkeys(G.nodes, 0)  # Load vector for Greedy++.
    lambda_vec = {node: lam if node in protected_nodes_all else 0 for node in G.nodes}
    best_density = 0.0  # Highest density encountered.
    best_subgraph = set()  # Nodes of the best subgraph found.

    for _ in range(iterations):
        # Initialize heap for fast access to minimum weighted degree.
        heap = nx.utils.BinaryHeap()

        # Compute initial weighted degrees and add nodes to the heap.
        for node, degree in G.degree:
            # heap.insert(node, loads[node] + degree)
            heap.insert(node, loads[node] + degree + lambda_vec[node])
        # Set up tracking for current graph state.
        remaining_nodes = set(G.nodes)
        protected_nodes = set(protected_nodes_all)
        num_edges = G.number_of_edges()
        current_degrees = dict(G.degree)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)

            # Current density of the (implicit) graph
            # current_density = num_edges / num_nodes
            current_density = compute_cost_function(num_nodes, num_edges, protected_nodes_all, protected_nodes, lam, weight=weight, formulation=formulation)

            # Update the best density.
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)
                protected_nodes_densest = protected_nodes.copy()

            # Pop the node with the smallest weighted degree.
            node, _ = heap.pop()
            if node not in remaining_nodes:
                continue  # Skip nodes already removed.

            # Update the load of the popped node.
            loads[node] += current_degrees[node] + lambda_vec[node]
            
            # Update neighbors' degrees and the heap.
            for neighbor in G.neighbors(node):
                if neighbor in remaining_nodes:
                    current_degrees[neighbor] -= 1
                    num_edges -= 1
                    heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor] + lambda_vec[neighbor])

            # Remove the node from the remaining nodes.
            remaining_nodes.remove(node)
            protected_nodes.discard(node)

    # return best_density, best_subgraph, protected_nodes_densest
    return best_subgraph, best_density, protected_nodes_densest



def compute_f(num_nodes, num_edges, protected_nodes_all, protected_nodes, lam, weight=None, formulation=1):
    """
    compute F(S_i) 
    """

    if formulation==1:
        # num_edges = S.size(weight)
        num_protected = len(protected_nodes)
        f = num_edges + lam*num_protected
    elif formulation==2:
        # num_edges = S.size(weight)
        # num_nodes = S.number_of_nodes()
        num_protected_all = len(protected_nodes_all)
        num_protected = len(protected_nodes)
        f = num_edges - lam*(num_nodes+num_protected_all-2*num_protected)

    return f

def compute_cost_function(num_nodes, num_edges, protected_nodes_all, protected_nodes_i, lam, weight, formulation=1): 
    return compute_f(num_nodes, num_edges, protected_nodes_all, protected_nodes_i, lam, weight, formulation) / num_nodes
