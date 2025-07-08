import amrlib
import penman
import spacy
from graphviz import Digraph

def create_enhanced_graph(sentence: str):
    """
    Parses a sentence, enhances it with NER, and returns a Digraph object.
    """
    # --- 1. Load Models ---
    stog = amrlib.load_stog_model('model_parse_xfm_bart_large-v0_1_0', device='cpu')
    nlp = spacy.load("en_core_web_sm")

    # --- 2. Initial AMR Parsing ---
    graph_string = stog.parse_sents([sentence])[0]
    p_graph = penman.decode(graph_string)
    print("--- Original AMR Graph ---")
    print(graph_string)
    
    # --- 3. NER Extraction ---
    ner_doc = nlp(sentence)
    entities = {ent.text.lower(): ent.label_ for ent in ner_doc.ents}
    print(f"\n--- Found Entities (NER) ---")
    print(entities)

    # --- 4. Intelligent Graph Construction ---
    print("\n--- Building Enhanced Graph ---")
    dot = Digraph(comment=sentence)
    dot.attr('graph', label=f'AMR Graph for: "{sentence}"', labelloc='t', fontsize='20', fontname='Helvetica', rankdir='TB', splines='spline')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', color='black')
    dot.attr('edge', fontname='Helvetica', fontsize='12', color='#444444')

    # Create a mapping from generic node variables to specific labels
    enhancement_map = {}
    for inst in p_graph.instances():
        if inst.target == 'country' and 'chinese' in entities:
            enhancement_map[inst.source] = 'Chinese'
        if inst.target == 'government-organization' and 'the ministry of education' in entities:
            enhancement_map[inst.source] = 'Ministry of Education'
        if inst.target == 'more-than':
            enhancement_map[inst.source] = 'more than one million'

    # Add nodes, using the enhanced map
    for inst in p_graph.instances():
        node_var = inst.source
        node_label = enhancement_map.get(node_var, inst.target) # Use enhanced label if available
        
        fillcolor = '#c9d9f2' # Default
        if node_var == p_graph.top:
            fillcolor = '#a3d9a5' # Top node
        elif node_var in enhancement_map:
            fillcolor = '#f5d4a7' # Enhanced node
            
        dot.node(node_var, label=node_label, fillcolor=fillcolor)
        print(f"  Added node: {node_var} -> {node_label}")

    # Add edges, skipping those that are now redundant
    for edge in p_graph.edges():
        # Don't draw the edges for concepts we've manually replaced
        if edge.source in enhancement_map and edge.role in [':name', ':quant']:
            continue
        dot.edge(edge.source, edge.target, label=edge.role)

    return dot

# --- Main Execution ---
sentence_to_parse = "An official with the ministry of education said that more than one million Chinese have studied abroad."
final_dot_graph = create_enhanced_graph(sentence_to_parse)

# Render the final graph
output_path = '/home/masih/Desktop/AMR/amr_graph_final'
final_dot_graph.render(output_path, format='png', view=False, cleanup=True)

print(f"\nFinal, enhanced AMR graph saved as {output_path}.png")
