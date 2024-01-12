import networkx as nx
import time


# If you use the standard louvain, you should make sure to use a filter function before adding edges to the graph in
# partition heuristic. For example, I had decent results when just adding all weights below the respective median.
# i.e. only add edges with weight to graph_bi if the weight is BELOW the median weight for all uni weights.
def louvain_standard(graph):
    # Standard resolution parameter
    resolution = 1.0

    # Compute Louvain communities
    communities = list(nx.algorithms.community.louvain_partitions(graph, seed=time.time_ns(), resolution=resolution, weight="weight"))[0]

    # Sometimes the algorithm fails to find more than one community. The fallback here is to increase
    # the resolution parameter until we actually find more than one community.
    while len(communities) == 1:
        resolution += 0.1
        communities = list(nx.algorithms.community.louvain_partitions(graph, seed=time.time_ns(), resolution=resolution, weight="weight"))[0]

    return communities

def louvain_custom(graph):

    # Calculate general graph average
    ga = graph_average(graph)

    # Create singleton partition
    c = [[x] for x in graph.nodes]

    # Initialize some booleans to check if were caught in a loop
    improve = True
    merged = False
    fail_to_improve = False

    # Copy original graph to
    original_graph = graph.copy()

    # While we improve graph modularity
    while improve:

        # New partition after merging communities
        c_prime = one_step(graph, c, ga)

        # If there is a modularity improving move, we reassign our partition.
        # The approach here is special for louvain in the regard that we only do ONE MOVE for each
        # iteration of louvain.
        if c_prime != False:
            fail_to_improve = False
            c = c_prime

        # If we already acquired a bipartition, and we could not find an improving move, we terminate.
        if len(c) == 2 and fail_to_improve:
            c_prime = []
            for set_comm in c:
                aglom_com = set()
                for fs in set_comm:
                    aglom_com = aglom_com.union(set(fs))
                c_prime += [list(aglom_com)]
            c = c_prime
            break

        else:
            # If we dont have a bipartition, and still could not improve, we terminate.
            if fail_to_improve:
                c_prime = []
                for set_comm in c:
                    aglom_com = set()
                    for fs in set_comm:
                        aglom_com = aglom_com.union(set(fs))
                    c_prime += [list(aglom_com)]
                c = c_prime
                break

            fail_to_improve = True

            # if we improved last step, we merge nodes.
            if not merged:
                graph = create_average_merge_graph(graph, c)
                merged = True
            else:
                c_prime = []
                for set_comm in c:
                    aglom_com = set()
                    for fs in set_comm:
                        aglom_com = aglom_com.union(set(fs))
                    c_prime += [list(aglom_com)]
                graph = create_average_merge_graph(original_graph, c_prime)

            # compute new singleton partition and new graph average.
            c = [[x] for x in graph.nodes]
            ga = graph_average(graph)

    return c

# Compute the modularity of the whole graph with custom modularity.
def modularity_graph(graph: nx.Graph, communities: list, graph_average: float):

    # weights = dict(graph.degree(weight="weight"))
    community_modularity = []

    # Count inside connections, inside weight sum, outside connections and outside weight sum for each community
    for community in communities:
        outside = 0
        outside_connections = 0
        inside_connections = 0
        inside = 0
        for v in community:
            neighbors = dict((graph[v]))
            for n in neighbors:
                if n in community:
                    try:
                        inside += neighbors[n]["weight"]
                        inside_connections += 1
                    except KeyError:
                        for edge in neighbors[n]:
                            inside += neighbors[n][edge]["weight"]
                            inside_connections += 1
                else:
                    try:
                        outside += neighbors[n]["weight"]
                        outside_connections += 1
                    except KeyError:
                        for edge in neighbors[n]:
                            outside += neighbors[n][edge]["weight"]
                            outside_connections += 1

        # Add custom modularity function to full modularity sum
        if inside  != 0 and inside_connections != 0:
            community_modularity += [(outside**2/outside_connections) / (inside**2/inside_connections**2)]
        else:
            community_modularity += [(outside**2 / outside_connections)]

    # We also return the individual contribution for each community in community modularity. This speeds up the
    # Calculation of delta_modularilty in one_step.
    return community_modularity, (sum(community_modularity) / len(community_modularity)) / graph_average

# Returns average edge weight in the graph.
def graph_average(graph: nx.Graph):
    edges = list(graph.edges(data=True))
    edgesum = 0
    for e in edges:
        edgesum += e[2]["weight"]
    return edgesum / len(edges)

# Compute the modularity of a single community.
def modularity_community(graph, community):
    outside = 0
    outside_connections = 0
    inside_connections = 0
    inside = 0

    for v in community:
        neighbors = dict((graph[v]))

        for n in neighbors:
            if n in community:
                try:
                    inside += neighbors[n]["weight"]
                    inside_connections += 1
                except KeyError:
                    for edge in neighbors[n]:
                        inside += neighbors[n][edge]["weight"]
                        inside_connections += 1
            else:
                try:
                    outside += neighbors[n]["weight"]
                    outside_connections += 1
                except KeyError:
                    for edge in neighbors[n]:
                        outside += neighbors[n][edge]["weight"]
                        outside_connections += 1

    if outside_connections == 0:
        return graph_average(graph)

    if inside != 0 and inside_connections != 0:
        return (outside**3 / outside_connections) / ((inside**2 / inside_connections**2))
    else:
        return outside**3 / outside_connections

def one_step(graph, communities, graph_average):
    community_modularities = []

    # Initialize the tuple to find the next best move.
    max_insert_tuple = [None, None, None, 0]

    # Compute individual community modularities
    for community in communities:
        community_modularities += [modularity_community(graph, community)]

    # Modularity of the whole graph.
    baseline_mod = sum(community_modularities) / len(community_modularities)

    # Find the best move.
    for community in communities:

        # Examine each vertex in the community and find a max_delta insert community.
        for v in community:
            for insert_community in communities:

                # Skip same communities
                if insert_community == community:
                    continue

                # Compute the modularity of the new insert community
                insert_modularity = modularity_community(graph, insert_community + [v])

                remove_community = [x for x in community if x != v]

                # We dont want to move everything into one community
                if remove_community == [] and len(communities) == 2:
                    continue
                # If we delete the old community, hence empty it by moving v, we set the remove modularity to 0.
                if remove_community == []:
                    remove_modularity = 0
                else:
                    # Compute the new
                    remove_modularity = modularity_community(graph, [x for x in community if x != v])

                if True:
                    # If we fully remove the community, we only check if we increased the modularity of the community we insert into.
                    if remove_modularity == 0:
                        gain = insert_modularity - community_modularities[communities.index(insert_community)]

                    # If both communities still persist, we compute the new modularity of G using difference of
                    # old communities vs new communities (for insert and remove).
                    else:
                        gain = (insert_modularity - community_modularities[communities.index(insert_community)]) + (remove_modularity - community_modularities[communities.index(community)])
                # If we improve, we check if its higher than the last max_delta improvement we calculated
                if gain > 0.0:
                    if gain > max_insert_tuple[3]:
                        max_insert_tuple = [v, communities.index(community), communities.index(insert_community), gain]

    # Could not find a move to improve modularity.
    if max_insert_tuple == [None, None, None, 0]:
        return False
    # Alter communities based on the max_delta we found.
    else:
        communities[max_insert_tuple[1]] = [x for x in communities[max_insert_tuple[1]] if x != max_insert_tuple[0]]
        communities[max_insert_tuple[2]] = communities[max_insert_tuple[2]] + [max_insert_tuple[0]]
        communities = [x for x in communities if x != []]
        return communities

def create_average_merge_graph(graph, communities):

    # Idea is to maintain the modularity of the graph even when merging.
    # We achieve that by just making a multigraph and adding all the old edges to it.
    merge_graph = nx.MultiGraph()

    for community in communities:
        merge_graph.add_node(frozenset(community))

    for i in range(len(communities)):
        community = communities[i]

        for j in range(i, len(communities)):
            community2 = communities[j]

            insert_weight = 0
            count = 0

            # Same community
            if i == j:
                for k in range(len(community)):
                    u = community[k]
                    for l in range(k+1, len(community2)):
                        v = community[l]
                        try:
                            insert_weight += graph[u][v]["weight"]
                            count += 1
                        except KeyError:
                            pass
                if count != 0:
                    merge_graph.add_edge(frozenset(community), frozenset(community2), weight=insert_weight/count)
            # Different community
            else:
                for u in community:
                    for v in community2:
                        try:
                            insert_weight += graph[u][v]["weight"]
                            count += 1
                        except KeyError:
                            pass
                if count != 0:
                    merge_graph.add_edge(frozenset(community), frozenset(community2), weight=insert_weight/count)

    return merge_graph

# Deprec Merge Graph function.
def create_merge_graph(graph, communities):
    merge_graph = nx.MultiGraph()

    for community in communities:
        merge_graph.add_node(frozenset(community))

    for i in range(len(communities)):
        community = communities[i]

        for j in range(i, len(communities)):
            community2 = communities[j]

            # Same community
            if i == j:
                for k in range(len(community)):
                    u = community[k]
                    for l in range(k+1, len(community2)):
                        v = community[l]
                        try:
                            insert_weight = graph[u][v]["weight"]
                            merge_graph.add_edge(frozenset(community), frozenset(community2), weight=insert_weight)
                        except KeyError:
                            pass
            # Different community
            else:
                for u in community:
                    for v in community2:
                        try:
                            insert_weight = graph[u][v]["weight"]
                            merge_graph.add_edge(frozenset(community), frozenset(community2), weight=insert_weight)
                        except KeyError:
                            pass
    return merge_graph