import argparse
import os
import subprocess
import re

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-knightKing', type=str,
                    help='Path to KnightKing')
parser.add_argument('-runs', type=int, help="Number of Runs")

args = parser.parse_args()

input_dir = "./input"

#Run KnightKing Benchmarks
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data")},
    "LiveJournal": {"v": 4847569, "path": os.path.join(input_dir, "LJ1.data")},
    "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data")},
    "Patents": {"v":6009555,"path":os.path.join(input_dir, "patents.data")},
    "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
}

knightKing = os.path.join(args.knightKing, 'build/bin')

knightKingWalks = {
    "deepwalk": " -l 100 ", "node2vec": " -p 2.0 -q 0.5 -l 100 ", "ppr":" -t 0.001 "
}

nextDoorApps = ["DeepWalk", "PPR", "Node2Vec", "KHop", "MultiRW","MVS"] #["Layer", "subGraphSampling", "mvs"]

results = {"KnightKing": {walk : {graph: -1 for graph in graphInfo} for walk in knightKingWalks},
           "SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "TP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},}

# for walk in knightKingWalks:
#     for graph in graphInfo:
#         times = []
#         for run in range(args.runs):
#             walkBinary = os.path.join(knightKing, walk) + " -w %d "%graphInfo[graph]["v"] + \
#                 " -v %d"%graphInfo[graph]["v"] +\
#                 " -s weighted " + "-g " + graphInfo[graph]["path"] + \
#                  knightKingWalks[walk]   

#             status, output = subprocess.getstatusoutput(walkBinary)
#             t = float(re.findall(r'total time ([\d\.]+)s', output)[0])
#             print (t)
#             times += [t]

#         avg = sum(times)/len(times)

#         results["KnightKing"][walk][graph] = avg

for app in nextDoorApps:
    times = []
    appBinary = os.path.join("build/tests/singleGPU", app.lower())
    print ("Running ", appBinary)
    status, output = subprocess.getstatusoutput(appBinary)
    print (output)
    for technique in results:
        if technique == "KnightKing":
            continue
        for graph in graphInfo:
            out = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)[0]
            end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
            print(app, graph, technique, end2end)

print(results)

