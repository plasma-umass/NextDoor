import argparse
import os
import subprocess
import re
import datetime

#TODO: Store the output as log some where.
logFile = "benchmarking.log"

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-knightKing', type=str,
                    help='Path to KnightKing',required=True)
parser.add_argument('-runs', type=int, help="Number of Runs",required=True)
parser.add_argument('-gpus', type=str, help="CUDA DEVICES",required=False)

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
    "Node2Vec": " -p 2.0 -q 0.5 -l 100 ", "PPR":" -t 0.001 ", "DeepWalk": " -l 100 ",
}

nextDoorApps = ["PPR", "Node2Vec","DeepWalk","KHop","MultiRW","MVS","ClusterGCN"] #"FastGCN", "LADIES"
multiGPUApps = ["PPR", "Node2Vec","DeepWalk","KHop"]

results = {"KnightKing": {walk : {graph: -1 for graph in graphInfo} for walk in knightKingWalks},
           "SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "TP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "InversionTime": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "MultiGPU-LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps}}

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()
    with open(logFile, "r+") as f:
        f.write(s)

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

for walk in knightKingWalks:
    for graph in graphInfo:
        times = []
        for run in range(args.runs):
            walkBinary = os.path.join(knightKing, walk.lower()) + " -w %d "%graphInfo[graph]["v"] + \
                " -v %d"%graphInfo[graph]["v"] +\
                " -s weighted " + "-g " + graphInfo[graph]["path"] + \
                 knightKingWalks[walk]   
            print("Executing " + walkBinary)
            writeToLog("Executing "+walkBinary)
            status, output = subprocess.getstatusoutput(walkBinary)
            writeToLog(output)
            t = float(re.findall(r'total time ([\d\.]+)s', output)[0])
            times += [t]

        avg = sum(times)/len(times)

        results["KnightKing"][walk][graph] = avg


for app in nextDoorApps:
    times = []
    appBinary = os.path.join("build/tests/singleGPU", app.lower())
    print ("Running ", appBinary)
    writeToLog("Executing "+appBinary)
    status, output = subprocess.getstatusoutput(appBinary)
    writeToLog(output)
    print (output)
    for technique in results:
        if technique == "KnightKing" or technique == "InversionTime" or technique == "MultiGPU-LB":
            continue
        for graph in graphInfo:
            print (app, graph, technique)
            out = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)[0]
            end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
            results[technique][app][graph] = float(end2end[0])
            if (technique == "LB"):
                inversionTime = re.findall(r'InversionTime: ([\d\.]+)', out)
                loadbalancingTime = re.findall(r'LoadBalancingTime: ([\d\.]+)', out)
                t = float(inversionTime[0]) + float(loadbalancingTime[0])
                results["InversionTime"][app][graph] = t

if len(args.gpus) > 1:
    #MultiGPU Results
    for app in multiGPUApps:
        times = []
        appBinary = "CUDA_DEVICES="+args.gpus + " " + os.path.join("build/tests/multiGPU", app.lower())
        print ("Running ", appBinary)
        writeToLog("Executing "+appBinary)
        status, output = subprocess.getstatusoutput(appBinary)
        writeToLog(output)
        print (output)
        technique = "LB"
        for graph in graphInfo:
            print (app, graph, technique)
            out = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)[0]
            end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
            results["MultiGPU-LB"][app][graph] = float(end2end[0])
else:
    print ("Not taking MultiGPU results because only one GPU mentioned in 'gpus': ", args.gpus)
    
#Speedup Over KnightKing
print ("\n\nFigure 7 (a): Speedup Over KnightKing")
row_format = "{:>20}" * 3
print (row_format.format("Random Walk", "Graph", "Speedup"))
for walk in knightKingWalks:
    for graph in graphInfo:
        speedup = results["KnightKing"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, speedup))
    
#Speedup Over SP and TP
print ("\n\nFigure 7 (c): Speedup Over SP and TP")
row_format = "{:>30}" * 4
print (row_format.format("Sampling App", "Graph", "Speedup over SP", "Speedup over TP"))
for walk in nextDoorApps:
    for graph in graphInfo:
        speedupSP = results["SP"][walk][graph]/results["LB"][walk][graph]
        speedupTP = results["TP"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, speedupSP, speedupTP))


print ("\n\nFigure 6: %age of Time Spent in Building scheduling index")
row_format = "{:>30}" * 3
print (row_format.format("Sampling App", "Graph", "%age of Time in Index"))
for walk in nextDoorApps:
    for graph in graphInfo:
        t = results["InversionTime"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, t * 100))

#Multi GPU results
print ("\n\nFigure 10: Speedup of sampling using Multiple GPUs over 1 GPU")
row_format = "{:>30}" * 3
print (row_format.format("Sampling App", "Graph", "%age of Time in Index"))
for walk in multiGPUApps:
    for graph in graphInfo:
        speedup = results["LB"][walk][graph]/results["MultiGPU-LB"][walk][graph]
        print (row_format.format(walk, graph, speedup * 100))