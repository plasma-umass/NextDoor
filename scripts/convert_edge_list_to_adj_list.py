import sys
import re

assert len(sys.argv) == 3, "Usage: <edge list file> <adj list file>"

f_edge_list = sys.argv[1]
f_adj_list = sys.argv[2]
edges = 0
adj_list = {}

f = open(f_edge_list, "r+")
s = f.readline ()
edge_list = []
vertex_to_label = {}

n_vertices = 0
while s != "":
  if s[0] != "#":
    try:
      v, u = re.findall(r'(\d+)\s+(\d+)', s.strip())[0]
    except e:
      print (s)
      raise e
    v = int(v)
    u = int(u)
    edge_list += [(v,u)]

    if (v not in vertex_to_label):
      vertex_to_label[v] = len(vertex_to_label)
      
    if (u not in vertex_to_label):
      vertex_to_label[u] = len(vertex_to_label)
    edges +=1

  s = f.readline ()

f.close ()

n_vertices = len(vertex_to_label)

for edge in edge_list:
  v,u = edge
  v = vertex_to_label[v]
  u = vertex_to_label[u]

  if v not in adj_list:
    adj_list[v] = set()
  else:
    adj_list[v].add (u)
  if u not in adj_list:
    adj_list[u] = set()

for v in range(0, max(adj_list)+1):
  if v not in adj_list:
    assert False

print ("graph with %d vertices and %d edges", len(adj_list),edges)

with open(f_adj_list, "w") as f:
  for v in sorted(list(adj_list.keys())):
    adj_list[v] = sorted(list(adj_list[v]))
    s = str(v) + " 0 "
    s += " ".join(map (lambda x: str(x), adj_list[v]))
    f.write(s + "\n")

