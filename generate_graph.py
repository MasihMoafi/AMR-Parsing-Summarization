
import amrlib
import penman
from graphviz import Digraph

def penman_to_dot(penman_string: str, sentence: str) -> Digraph:
    """
    Converts a PENMAN string to a styled Graphviz Digraph object.
    """
    g = penman.decode(penman_string)
    dot = Digraph(comment=sentence)

    # Graph attributes
    dot.attr('graph', 
             label=f'AMR Graph for: "{sentence}"',
             labelloc='t',
             fontsize='20',
             fontname='Helvetica',
             rankdir='TB',
             splines='spline')

    # Node attributes
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fontname='Helvetica',
             color='black')

    # Edge attributes
    dot.attr('edge', 
             fontname='Helvetica',
             fontsize='12',
             color='#444444')

    # Add nodes with specific styling
    for instance in g.instances():
        if instance.source == g.top:
            # Style for the top node
            dot.node(instance.source, label=instance.target, 
                     fillcolor='#a3d9a5', shape='ellipse')
        else:
            # Style for other concept nodes
            dot.node(instance.source, label=instance.target, 
                     fillcolor='#c9d9f2')

    # Add edges
    for edge in g.edges():
        dot.edge(edge.source, edge.target, label=edge.role)
        
    return dot

# --- Main Execution ---
# Load the pre-trained model
stog = amrlib.load_stog_model('model_parse_xfm_bart_large-v0_1_0', device='cpu')

# The sentence to parse
sentence = "An official with the ministry of education said that more than one million Chinese have studied abroad."

# Parse the sentence into an AMR graph string
graphs = stog.parse_sents([sentence])
graph_string = graphs[0]

# Convert the AMR string to a styled Digraph object
dot_graph = penman_to_dot(graph_string, sentence)

# Render the graph to a file
output_path = '/home/masih/Desktop/AMR/amr_graph_showcase'
dot_graph.render(output_path, format='png', view=False, cleanup=True)

print(f"Showcase AMR graph generated and saved as {output_path}.png")
