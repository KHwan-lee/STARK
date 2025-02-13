import matplotlib.pyplot as plt
import networkx as nx

# Define a directed graph to represent function relationships
G = nx.DiGraph()

# Add main.py and its related functions
G.add_node("main.py")
G.add_nodes_from([
    "parse_args.py (parse_args)", 
    "KnowledgeGraph.py (KnowledgeGraph)",
    "LoraKGE_Layers.py (TransE)",
    "model_process.py (TrainProcessor)",
    "model_process.py (TestProcessor)",
    "LoraKGE_Layers.py (switch_snapshot)",
    "LoraKGE_Layers.py (get_new_ordered_entities)",
    "LoraKGE_Layers.py (get_new_ordered_edges)",
    "model_process.py (run_epoch)",
    "model_process.py (margin_loss)",
    "LoraKGE_Layers.py (predict)",
    "save_logs"
])

# Add edges to represent function dependencies
edges = [
    ("main.py", "parse_args.py (parse_args)"),
    ("main.py", "KnowledgeGraph.py (KnowledgeGraph)"),
    ("main.py", "LoraKGE_Layers.py (TransE)"),
    ("main.py", "model_process.py (TrainProcessor)"),
    ("main.py", "model_process.py (TestProcessor)"),
    ("main.py", "model_process.py (run_epoch)"),
    ("model_process.py (run_epoch)", "LoraKGE_Layers.py (predict)"),
    ("model_process.py (run_epoch)", "model_process.py (margin_loss)"),
    ("main.py", "LoraKGE_Layers.py (switch_snapshot)"),
    ("LoraKGE_Layers.py (switch_snapshot)", "LoraKGE_Layers.py (get_new_ordered_entities)"),
    ("LoraKGE_Layers.py (switch_snapshot)", "LoraKGE_Layers.py (get_new_ordered_edges)"),
    ("main.py", "save_logs")
]

# Add edges to the graph
G.add_edges_from(edges)

# Visualize the graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)  # Position for better layout

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="black")
nx.draw_networkx_labels(G, pos, font_size=9, font_color="black")

# Add a title
plt.title("Function Call Relationships in main.py", fontsize=14)
plt.axis("off")
plt.show()
