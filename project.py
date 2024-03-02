import networkx as nx
import matplotlib.pyplot as plt
import math


def pagerank(graph, alpha=0.85, max_iterations=100, tol=1e-6):
    G = nx.DiGraph(graph)
    
    node_scores = {node: 1 / G.number_of_nodes() for node in G.nodes()}
    
    iteration_scores = []
    
    for _ in range(max_iterations):
        prev_scores = node_scores.copy()
        iteration_scores.append(prev_scores.copy()) 
        
        for node in G.nodes():
            score = sum(prev_scores[neighbor] / G.out_degree(neighbor) for neighbor in G.predecessors(node))
            node_scores[node] = alpha * score + (1 - alpha) / G.number_of_nodes()
        
        if sum(abs(node_scores[node] - prev_scores[node]) for node in G.nodes()) < tol:
            break
    
    return node_scores, iteration_scores

# graph = {
#     'A': ['B', 'C', 'D'],
#     'B': ['A', 'E', 'F'],
#     'C': ['A', 'B'],
#     'D': ['C', 'B'],
#     'E': ['A', 'G', 'F', 'B'],
#     'F': ['A'],
#     'G': ['A', 'B', 'C', 'D']
# }
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['C', 'D'],
    'C': ['A'],
    'D': ['C', 'A']
}
# graph = {
#     'A': ['B', 'C'],
#     'B': ['C'],
#     'C': ['A']
# }

scores, iteration_scores = pagerank(graph)
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # Sort scores in descending order

for rank, (node, score) in enumerate(sorted_scores, start=1):
    print(f"Rank {rank}: {node} (PageRank: {score})")

G = nx.DiGraph(graph)

node_sizes = [12000 * scores[node] for node in G.nodes()]

labels = {}
for rank, (node, score) in enumerate(sorted_scores, start=1):
    score = round(score, 2)
    labels[node] = f'{node}\nRank: {rank}\nPageRank: {score}'

nx.draw(
    G,
    with_labels=True,
    labels=labels,
    node_color='yellow',
    node_size=node_sizes,
    font_size=5,
    arrowsize=16,
)
plt.title('Graph with Node Sizes Proportional to PageRank Scores and Rankings')
plt.savefig('pagerank_graph.png')

plt.show()

x = range(len(iteration_scores))

for node_idx, label in enumerate(labels):
    y = [scores[label] for scores in iteration_scores]
    plt.plot(x, y, label=f'{label}')

plt.xlabel('Iteration')
plt.ylabel('PageRank')
plt.title('PageRank Changes with each iteration')

plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('pagerank_changes.png')
plt.show()
