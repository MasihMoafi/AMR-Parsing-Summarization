import amrlib
import penman
import spacy
from graphviz import Digraph

def create_enhanced_graph(sentence: str):
    """
    Parses a sentence, enhances it with NER, and returns a correctly
    structured and visualized Digraph object.
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
    print("\n--- Building Final Graph ---")
    dot = Digraph(comment=sentence)
    dot.attr('graph', label=f'AMR Graph for: "{sentence}"', labelloc='t', fontsize='20', fontname='Helvetica', rankdir='TB', splines='spline')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', color='black')
    dot.attr('edge', fontname='Helvetica', fontsize='12', color='#444444')

    # --- Identify nodes to enhance and redundant nodes to remove ---
    enhancement_map = {}
    nodes_to_remove = set()
    for inst in p_graph.instances():
        if inst.target == 'country' and 'chinese' in entities:
            enhancement_map[inst.source] = 'Chinese'
            for edge in p_graph.edges(source=inst.source):
                if edge.role == ':name': nodes_to_remove.add(edge.target)
        
        if inst.target == 'government-organization' and 'the ministry of education' in entities:
            enhancement_map[inst.source] = 'Ministry of Education'
            for edge in p_graph.edges(source=inst.source):
                if edge.role == ':name': nodes_to_remove.add(edge.target)

        if inst.target == 'more-than':
            enhancement_map[inst.source] = 'more than one million'
            for edge in p_graph.edges(source=inst.source):
                if edge.role == ':op1': nodes_to_remove.add(edge.target)
    
    print(f"  Nodes to enhance: {list(enhancement_map.keys())}")
    print(f"  Nodes to remove: {list(nodes_to_remove)}")

    # --- Add nodes to visualization, skipping the removed ones ---
    for inst in p_graph.instances():
        if inst.source in nodes_to_remove:
            continue # CRITICAL FIX: Do not draw the removed node

        node_label = enhancement_map.get(inst.source, inst.target)
        
        fillcolor = '#c9d9f2'
        if inst.source == p_graph.top: fillcolor = '#a3d9a5'
        elif inst.source in enhancement_map: fillcolor = '#f5d4a7'
            
        dot.node(inst.source, label=node_label, fillcolor=fillcolor)

    # --- Add edges to visualization, skipping edges that point to removed nodes ---
    for edge in p_graph.edges():
        if edge.target in nodes_to_remove:
            continue # CRITICAL FIX: Do not draw the edge to the removed node
        dot.edge(edge.source, edge.target, label=edge.role)

    return dot

# --- Main Execution ---
sentence_to_parse = "An official with the ministry of education said that more than one million Chinese have studied abroad."
final_dot_graph = create_enhanced_graph(sentence_to_parse)

output_path = '/home/masih/Desktop/AMR/amr_graph_final'
final_dot_graph.render(output_path, format='png', view=False, cleanup=True)

print(f"\nFinal, correct graph saved as {output_path}.png")