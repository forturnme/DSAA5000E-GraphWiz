import gradio
from transformers import pipeline

pipe = pipeline("text-generation", model="graphwiz/LLaMA2-7B-DPO", device="cuda", max_length=2048)
alpaca_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction:\n{query}\n\n### Response:"

def main_fn(input):
    finp = alpaca_template.format(query = input)
    ret = pipe(finp)[0]['generated_text'][len(finp)+2:]
    reasoning, result = ret.split("###")[0], ret.split("###")[1].strip()
    return reasoning, result


# build a gradio interface on port 9999, listen 0.0.0.0
print("Starting Gradio server on port 9999")
iface = gradio.Interface(fn=main_fn, inputs=["text"], outputs=["text", "text"], examples=[
    ["Find the shortest path between two nodes in an undirected graph. In an undirected graph, (i,j,k) means that node i and node j are connected with an undirected edge with weight k. Given a graph and a pair of nodes, you need to output the shortest path between the two nodes. Q: The nodes are numbered from 0 to 8, and the edges are: (0,1,4) (1,2,7) (1,7,1) (1,3,4) (2,6,2) (2,4,8) (2,7,5) (3,6,1) (4,8,3) (5,6,6) (6,8,8) (7,8,7). Give the weight of the shortest path from node 0 to node 8.",
    "All the paths from node 0 to node 8 are:\n0,1,7,8 with a total weight of <<4 + 1 + 7 = 12>>,\n0,1,3,6,8 with a total weight of <<4 + 4 + 1 + 8 = 17>>,\n0,1,2,6,8 with a total weight of <<4 + 7 + 2 + 8 = 21>>,\n0,1,2,4,8 with a total weight of <<4 + 7 + 8 + 3 = 22>>,\n0,1,2,7,8 with a total weight of <<4 + 7 + 5 + 7 = 23>>.\nThe weight of path 0,1,7,8 is the smallest, so the shortest path from node 0 to node 8 is [0,1,7,8] with a total weight of 12. ### 12."],
    ["Determine whether or not a graph is bipartite. In a directed graph, (i->j) means that node i and node j are connected with an directed edge from node i to node j. Given a graph, you need to output Yes or No, indicating whether the graph is bipartite. Q: The nodes are numbered from 0 to 10, and the edges are: (3->5) (0->10) (5->6) (1->10) (0->1) (4->7) (1->3) (1->4) (1->2) (4->5) (0->4) (7->8) (0->5) (6->8) (6->7) (0->8) (4->9) (0->7) (4->8) (1->6) (3->9) (3->4) (1->8) (2->10) (1->7) (2->8) (4->10) (3->7) (0->3) (2->3) (4->6) (0->9) (2->5) (2->7) (5->8) (3->6) (2->6). Is this graph bipartite?",
    "No"],
    ["Determine whether two nodes are connected in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph and a pair of nodes, you need to output Yes or No, indicating whether the node i and node j are connected. Q: The nodes are numbered from 0 to 12, and the edges are: (0, 12) (0, 8) (0, 4) (1, 12) (1, 5) (1, 7) (1, 2) (1, 6) (2, 8) (2, 12) (2, 3) (2, 5) (3, 12) (3, 6) (4, 7) (4, 6) (5, 12) (6, 7) (6, 12) (6, 10) (7, 11) (7, 10) (8, 12) (9, 12) (9, 11) (10, 12). Is there a path between node 8 and node 10?", 
    ""],
    ["Determine whether or not there is a cycle in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph, you need to output Yes or No, indicating whether there is a cycle in the graph. Q: The nodes are numbered from 0 to 4, and the edges are: (0, 3) (0, 4) (1, 3) (1, 4) (2, 3) (2, 4) (3, 4). Is there a cycle in this graph?",
    ""],
    ["Find the maximum flow between two nodes in a directed graph. In a directed graph, (i->j,k) means that node i and node j are connected with an directed edge from node i to node j with weight k. Given a graph and a pair of nodes, you need to output the maximum flow between the two nodes. Q: The nodes are numbered from 0 to 4, and the edges are: (0->1,6) (1->2,2) (2->4,6) (3->4,4). What is the maximum flow from node 0 to node 4?",
    ""],
    ["Determine whether or not there is a Hamiltonian path in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph, you need to output Yes or No, indicating whether there is a Hamiltonian path in the graph. Q: The nodes are numbered from 0 to 4, and the edges are: (0, 1) (0, 4) (1, 3) (2, 3) (2, 4). Is there a Hamiltonian path in this graph?",
    ""],
    ["Determine if a smaller graph is present as an exact match within a larger graph. In a directed graph, (i->j) means that node i and node j are connected with a directed edge from node i to node j. Given a graph G and a subgraph G', you need to output Yes or No, indicating whether subgraph G' is present within the directed graph G. Q: The nodes of graph G are numbered from 0 to 6, and the edges are: (0->1) (0->5) (1->2) (1->4) (2->4) (2->6) (4->5). The nodes of subgraph G' are numbered from a to e, and the edges are: (a->d) (b->e) (b->c) (c->d). Is subgraph G' present within graph G as a direct substructure?",
    ""],
    ["Find one of the topology sorting paths of the given graph. In a directed graph, (i->j) means that node i and node j are connected with a directed edge from node i to node j. Given a graph, you need to output one of the topology sorting paths of the graph. Q: The nodes are numbered from 0 to 7, and the edges are: (6->1) (6->2) (6->7) (6->4) (6->5) (1->4) (1->3) (1->7) (4->7) (4->2) (4->5) (4->3) (4->0) (7->3) (7->0) (7->2) (3->5) (5->2) (2->0). Give one topology sorting path of this graph.",
    ""],
    ["Find the maximum sum of the weights of three interconnected nodes. In an undirected graph, [i, k] means that node i has the weight k. (i,j) means that node i and node j are connected with an undirected edge. Given a graph, you need to output the maximum sum of the weights of three interconnected nodes. Q: The nodes are numbered from 0 to 4, weights of nodes are: [0, 8] [1, 5] [2, 3] [3, 6] [4, 3], and the edges are: (0, 4) (0, 3) (0, 1) (1, 3) (1, 2) (3, 4). What is the maximum sum of the weights of three nodes?",
    ""]
], title="Inferencing with Graphwiz")
iface.launch(server_name="0.0.0.0", server_port=9999, ssl_verify=False)
