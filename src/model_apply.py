import json, sys, time, gc
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
from data_generate import get_covariates_forward_looking
def generate_route(linear, route, route_data, travel_times, package_data, features_per_stop):
    stops = []
    station=''
    for st in route_data["stops"].keys():
        if route_data["stops"][st]["type"] == "Station":
            station = st
        else:
            stops.append(st)
    
    yet_to_go = stops    
    nstops = len(stops)
    prev_stop = station
    proposed_route = {"proposed":{station:0}}
        
    G = nx.DiGraph()
    ebunch  = [(i,j,travel_times[i][j]) for i in stops for j in stops if i != j]
    G.add_weighted_edges_from(ebunch)
    ebunch  = [(i,j,1/(travel_times[i][j] + 0.01)) for i in stops for j in stops if i != j]
    G.add_weighted_edges_from(ebunch,weight = "inv_distance")

    betweenness = nx.algorithms.centrality.betweenness_centrality(G,weight='weight')
    eigenvector = nx.algorithms.centrality.eigenvector_centrality(G,weight='weight')
    closeness = nx.algorithms.centrality.closeness_centrality(G,distance='weight')
    pagerank  = nx.algorithms.link_analysis.pagerank_alg.pagerank(G,weight='inv_distance')
    nn= tf.keras.models.load_model("./data/model_build_outputs/nn.h5", compile=False)

    for i in range(nstops-1):

        x = get_covariates_forward_looking(yet_to_go, prev_stop, station, route_data, travel_times,
                                   package_data, features_per_stop, len(yet_to_go), pagerank,closeness,eigenvector,betweenness)
        xshaped = np.reshape(x,(1,len(yet_to_go),features_per_stop))
        utilities = nn(xshaped[:1,:,:])[0,:].numpy() +xshaped.dot(linear).T

        idx = np.argmax(utilities)
        chosen_stop = yet_to_go[idx]
        yet_to_go.remove(chosen_stop)

        pgold = pagerank[chosen_stop]
        weightsum = sum([(1/(travel_times[chosen_stop][j]+ 0.01)) for j in yet_to_go ])
        pg_delta = {j: pgold *(1/(travel_times[chosen_stop][j]+ 0.01))/weightsum for j in yet_to_go }
        for s in yet_to_go:
            pagerank[s] = (pagerank[s] + pg_delta[s])

        proposed_route["proposed"][chosen_stop] = i+1
        prev_stop = chosen_stop
    
    proposed_route["proposed"][yet_to_go[0]] = nstops
    
    return route, proposed_route

if __name__ == "__main__":
    all_travel_times = json.load(open("./data/model_apply_inputs/new_travel_times.json", newline=''))
    all_route_data = json.load(open("./data/model_apply_inputs/new_route_data.json", newline=''))
    all_package_data = json.load(open("./data/model_apply_inputs/new_package_data.json", newline=''))
    routes = list(all_route_data.keys())

    linear = np.load("./data/model_build_outputs/linear.npy")
    features_per_stop = len(linear)

    pool = mp.Pool(min(16,mp.cpu_count()))

    inputs = [(linear, route, all_route_data[route], all_travel_times[route], all_package_data[route], features_per_stop) for route in routes]

    results = pool.starmap(generate_route,tqdm(inputs, total=len(inputs)),chunksize=1)
    pool.close()
    pool.join()

    proposed = {r:p for r,p in results}

    with open('./data/model_apply_outputs/proposed_sequences.json', 'w') as f:
        json.dump(proposed, f)

