import json
import numpy as np
import pickle as pkl
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
import gc


def nearest_diff_zone(st, stop_zone_id, travel_times, yet_to_go_wo_st):
    zone_to_go = list(set(stop_zone_id.values()))
    st_zone = stop_zone_id[st]
    zone_to_go.remove(st_zone)
    avg_dis_to_other_zone = []
    min_dis_to_other_zone = []
    sum_dis_to_other_zone = []
    # avg_dis_to_other_zone.append(np.mean([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] != stop_zone_id[key]]))
    # avg_dis_to_other_zone.append(np.min([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] != stop_zone_id[key]]))
    zone_to_go = [zone for zone in zone_to_go if zone == zone]
    for zone in zone_to_go:
        avg_dis_to_other_zone.append(np.mean([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))
        min_dis_to_other_zone.append(np.min([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))
        sum_dis_to_other_zone.append(np.sum([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))

    avg_zone =  avg_dis_to_other_zone[np.argmin(avg_dis_to_other_zone)] if len(avg_dis_to_other_zone) >  0 else 0    
    min_zone =  min_dis_to_other_zone[np.argmin(min_dis_to_other_zone)] if len(min_dis_to_other_zone) >  0 else 0 
    sum_zone =  sum_dis_to_other_zone[np.argmin(sum_dis_to_other_zone)] if len(sum_dis_to_other_zone) >  0 else 0 
    
    return (avg_zone, min_zone, sum_zone)

from anytree import Node, RenderTree
import anytree
from anytree import Node, RenderTree, AsciiStyle
from anytree.search import find,findall
from copy import deepcopy
def zone_tree(route_data, stops,zone_ids):
    root = Node(".")
    for st,zone_id in zip(stops,zone_ids):
        try:
            a,b,c,d = splitzone(zone_id)
        except:
            continue
        rc = [ch.name for ch in root.children]
        aexists = a not in rc
        if aexists:
            aguy = Node(a,parent=root)
        else:
            aguy = root.children[rc.index(a)]

        ac = [ch.name for ch in aguy.children]
        bexists = b not in ac
        if bexists:
            bguy = Node(b,parent=aguy)
        else:
            bguy = aguy.children[ac.index(b)]

        bc = [ch.name for ch in bguy.children]
        cexists = c not in bc
        if cexists:
            cguy = Node(c,parent=bguy)
        else:
            cguy = bguy.children[bc.index(c)]


        cc = [ch.name for ch in cguy.children]
        dexists = d not in cc
        if dexists:
            dguy = Node(d,parent=cguy)
        else:
            dguy = cguy.children[cc.index(d)]

        Node(st,parent=dguy)
    return root
def splitzone(zone_id):
    a = zone_id.split("-")[0]
    rest = zone_id.split("-")[1]
    b = rest.split(".")[0]
    c = rest[:-1].split(".")[1]
    d = zone_id[-1]
    return a,b,c,d

def zone_covariates(stop,tree,travel_times):
    stopnode = anytree.search.find(tree, lambda node: node.name == stop)
    
    if stopnode is not None:
        stoppath = stopnode.path

    dnode= stopnode.path[-2] if stopnode is not None else None
    same_d = [s.name for s in dnode.leaves if s.name != stop] if stopnode is not None else []
    
    cnode= stopnode.path[-3] if stopnode is not None else None
    same_c = [s.name for s in cnode.leaves if s.name != stop] if stopnode is not None else []
    
    bnode= stopnode.path[-4] if stopnode is not None else None
    same_b = [s.name for s in bnode.leaves if s.name != stop] if stopnode is not None else []
    
    anode= stopnode.path[-5] if stopnode is not None else None
    same_a = [s.name for s in anode.leaves if s.name != stop] if stopnode is not None else []
    
    na = len(same_a)
    nb = len(same_b)
    nc = len(same_c)
    nd = len(same_d)
        

    return na,nb,nc,nd


def nearest_diff_zone(st, stop_zone_id, travel_times, yet_to_go_wo_st):
    zone_to_go = list(set(stop_zone_id.values()))
    st_zone = stop_zone_id[st]
    zone_to_go.remove(st_zone)
    avg_dis_to_other_zone = []
    min_dis_to_other_zone = []
    sum_dis_to_other_zone = []
    # avg_dis_to_other_zone.append(np.mean([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] != stop_zone_id[key]]))
    # avg_dis_to_other_zone.append(np.min([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] != stop_zone_id[key]]))
    zone_to_go = [zone for zone in zone_to_go if zone == zone]
    for zone in zone_to_go:
        avg_dis_to_other_zone.append(np.mean([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))
        min_dis_to_other_zone.append(np.min([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))
        sum_dis_to_other_zone.append(np.sum([travel_times[st][key] for key in yet_to_go_wo_st if stop_zone_id[key] == zone]))

    avg_zone =  avg_dis_to_other_zone[np.argmin(avg_dis_to_other_zone)] if len(avg_dis_to_other_zone) >  0 else 0    
    min_zone =  min_dis_to_other_zone[np.argmin(min_dis_to_other_zone)] if len(min_dis_to_other_zone) >  0 else 0 
    sum_zone =  sum_dis_to_other_zone[np.argmin(sum_dis_to_other_zone)] if len(sum_dis_to_other_zone) >  0 else 0 
    
    return (avg_zone, min_zone, sum_zone)

def get_covariates_forward_looking(yet_to_go, prev_stop, station, route_data, travel_times,
                                   package_data, features_per_stop, maxstops,pagerank,closeness,eigenvector,betweenness):
    
    covariates = np.zeros((1,features_per_stop*maxstops))
    k=0
    stop_zone_id = {key: route_data['stops'][key]['zone_id'] for key in yet_to_go}
    zones_left_to_go = [route_data["stops"][stop]["zone_id"] for stop in yet_to_go]

    nstopsleftafter = len(yet_to_go)-1
    #tree = zone_tree(route_data,yet_to_go,zones_left_to_go)

    for st in yet_to_go:
        st_zone = stop_zone_id[st]
        dist = travel_times[prev_stop][st]
        
        yet_to_go_wo_st = [yet_to_go[i] for i in range(len(yet_to_go)) if yet_to_go[i] != st]
        yet_to_go_same_zone = [x for x in yet_to_go_wo_st if stop_zone_id[x] == st_zone]

        distances = np.array([travel_times[st][stop] for stop in yet_to_go_wo_st])
    
        avg_dis_to_other_stops = np.mean(distances) if len(distances) >  0 else 0
        avg_dis_to_closest_stops = np.mean(np.sort(distances)[:5])  if len(distances) >  0 else 0
    
        avg_dis_to_same_zone = np.mean([travel_times[st][key] for key in yet_to_go_same_zone])  if len(yet_to_go_same_zone) >  0 else 0
        min_distance_to_other_stops = np.min([travel_times[st][key] for key in yet_to_go_wo_st])
        
        ### avg, min, sum distance to nearest diff zone (in terms of avg, min, sum respectively)
        avg_dis_to_ndiff_zone, min_dis_to_ndiff_zone, sum_dis_to_ndiff_zone = nearest_diff_zone(st, stop_zone_id, travel_times, yet_to_go_wo_st)
        
        service_time = 0.0
        same_zone,zone_index,zone_second_index = 0,0,10
        station_index = 1 if station == prev_stop else 0

        same_a,same_b,same_c,same_d = 0,100,100,0
        same_zone = 0 
        try:
            if station_index == 0 and type(st_zone)==str and type(prev_stop_zone)==str:
                a,b,c,d = splitzone(st_zone)
                pa,pb,pc,pd = splitzone(prev_stop_zone)
                same_zone = 1 if st_zone == prev_stop_zone else 0
                same_a = 1 if pa==a else 0
                same_b = abs(float(b)-float(pb)) if (same_a==1) else 100
                same_c = abs(float(c)-float(pc)) if (same_b==0) else 100
                same_d = 1 if (same_c==0  and  pd == d ) else 0
                
        except:
            pass
        
        for pkg  in package_data[st].values():
            service_time = pkg['planned_service_time_seconds']
        # na,nb,nc,nd = zone_covariates(st,tree,travel_times)
        # na,nb,nc,nd = na/nstopsleftafter,nb/nstopsleftafter,nc/nstopsleftafter,nd/nstopsleftafter
        
        covariates[0,k:k+features_per_stop] = np.array([pagerank[st], betweenness[st], closeness[st], eigenvector[st], same_b, same_c, same_d, dist, service_time, avg_dis_to_closest_stops,
                                                        avg_dis_to_other_stops, avg_dis_to_same_zone, min_distance_to_other_stops, avg_dis_to_ndiff_zone, min_dis_to_ndiff_zone, sum_dis_to_ndiff_zone])

        k+=features_per_stop
        
    while k <= ((features_per_stop * maxstops) - features_per_stop):
        covariates[0,k:k+features_per_stop] = covariates[0,0:features_per_stop]
        k+=features_per_stop
        
    return covariates



def parse_route(route_data, actual_sequence, travel_times, package_data,maxstops=256,features_per_stop=4,):
    stops = []
    station = ''
    for st in route_data['stops'].keys():
        if route_data['stops'][st]['type'] == 'Station':
            station = st
        else:
            stops.append(st)
    G = nx.DiGraph()

    ebunch  = [(i,j,travel_times[i][j]) for i in stops for j in stops if i != j]

    G.add_weighted_edges_from(ebunch)
    
    ebunch  = [(i,j,1/(travel_times[i][j] + 0.01)) for i in stops for j in stops if i != j]
    
    G.add_weighted_edges_from(ebunch,weight = "inv_distance")

    betweenness = nx.algorithms.centrality.betweenness_centrality(G,weight='weight')
    eigenvector = nx.algorithms.centrality.eigenvector_centrality(G,weight='weight')
    closeness = nx.algorithms.centrality.closeness_centrality(G,distance='weight')
    pagerank  = nx.algorithms.link_analysis.pagerank_alg.pagerank(G,weight='inv_distance')
    
    prev_stop = station
    seq = actual_sequence['actual']
    rev = {v:k for k,v in seq.items()}
    nstops = len(stops)
    yet_to_go = stops.copy()
    timeoffset = 0
    covariates = np.zeros((nstops-1,maxstops*features_per_stop))
    choices = np.zeros((nstops-1,maxstops))
    for i in range(nstops-1):
        curr_stop = rev[i+1]
        stop_idx = yet_to_go.index(curr_stop)
        stop_cov = get_covariates_forward_looking(yet_to_go, prev_stop, station, route_data, travel_times, package_data,features_per_stop,maxstops,pagerank,closeness,eigenvector,betweenness)
        covariates[i,:stop_cov.shape[1]] = stop_cov
        choices[i,stop_idx] = 1.0
        
        yet_to_go.remove(curr_stop)
        
        pgold = pagerank[curr_stop]
        weightsum = sum([(1/(travel_times[curr_stop][j]+ 0.01)) for j in yet_to_go ])
        pg_delta = {j: pgold *(1/(travel_times[curr_stop][j]+ 0.01))/weightsum for j in yet_to_go }
        for s in yet_to_go:
            pagerank[s] = (pagerank[s] + pg_delta[s])
        
        prev_stop = curr_stop
    return covariates,choices


if __name__ == "__main__":
    all_travel_times = json.load(open("./data/model_build_inputs/travel_times.json", newline=''))
    all_route_data = json.load(open("./data/model_build_inputs/route_data.json", newline=''))
    all_actual_sequences = json.load(open("./data/model_build_inputs/actual_sequences.json", newline=''))
    all_package_data = json.load(open("./data/model_build_inputs/package_data.json", newline=''))
    routes = list(all_route_data.keys())

    maxstops = max([len(all_route_data[r]["stops"].keys())  for r in routes])
    features_per_stop = 16

    hqroutes= [route for route in  routes if all_route_data[route]["route_score"] == "High"][:200]

    pool = mp.Pool(min(16,mp.cpu_count()))
    inputs = [(all_route_data[route],all_actual_sequences[route],all_travel_times[route],all_package_data[route],maxstops,features_per_stop) for route in hqroutes]
    results = pool.starmap(parse_route,tqdm(inputs, total=len(inputs)),chunksize=1)

    pool.close()
    pool.join()

    numberstops = [len(all_route_data[route]["stops"].keys())-2 for route in hqroutes ]

    nobs = np.sum(numberstops)
    X = np.zeros((nobs,maxstops*features_per_stop))
    y = np.zeros((nobs,maxstops))
    i=0
    k=0
    for covariates,choices in tqdm(results):
        X[k:k+numberstops[i],:] = covariates
        y[k:k+numberstops[i],:] = choices 
        k+= numberstops[i]
        i+=1
        del covariates
        del choices
        covariates = None
        choices = None
    del results[:]
    del results


    del all_travel_times 
    del all_route_data 
    del all_actual_sequences
    del all_package_data

    gc.collect()


    X = np.reshape(X,(X.shape[0],maxstops,features_per_stop))

    np.save("./data/model_build_outputs/X.npy",X)
    np.save("./data/model_build_outputs/y.npy",y)





