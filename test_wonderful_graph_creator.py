import json
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI
import os

# Configuration de l'API NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def load_triplets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['triplets']

def analyze_triplets_with_ai(triplets):
    prompt = f"""Analyse ces triplets de connaissances et identifie les relations explicites et implicites.\nPour chaque triplet, indique s'il y a une relation valide et si oui, quelle est sa nature.\nFormat de réponse attendu: liste de dictionnaires avec 'valid': bool, 'relation_type': str, 'confidence': float\n\nTriplets à analyser:\n{json.dumps(triplets, indent=2)}\n"""
    completion = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"thinking": True}},
        stream=True
    )
    
    # On va collecter tout le texte retourné
    full_response = ""
    for chunk in completion:
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="")
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            full_response += chunk.choices[0].delta.content
    
    # Recherche de la première liste JSON dans la réponse
    import re
    match = re.search(r'(\[\s*{.*?}\s*\])', full_response, re.DOTALL)
    if match:
        try:
            analysis = json.loads(match.group(1))
        except Exception as e:
            print("Erreur de parsing JSON:", e)
            analysis = []
    else:
        print("Aucune liste JSON trouvée dans la réponse de l'API.")
        analysis = []
    # Sauvegarde pour debug
    with open('analysis_debug.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    return analysis

def create_knowledge_graph(triplets, analysis):
    G = nx.DiGraph()
    for triplet, analysis_item in zip(triplets, analysis):
        if analysis_item.get('valid', False):
            subject = triplet['subject']
            object_ = triplet['object']
            relation = analysis_item.get('relation_type', triplet['relation'])
            G.add_node(subject, type='entity')
            G.add_node(object_, type='entity')
            G.add_edge(subject, object_, relation=relation, confidence=analysis_item.get('confidence', 0.5))
    return G

def plot_graph(G):
    print(f"Nombre de nœuds dans le graphe : {G.number_of_nodes()}")
    print(f"Nombre d'arêtes dans le graphe : {G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        print("Le graphe est vide, rien à afficher.")
        return
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Graphe de Connaissances")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('knowledge_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    triplets = load_triplets('test_triplets_20250519_150237.json')
    print("Analyse des triplets en cours...")
    analysis = analyze_triplets_with_ai(triplets)
    print("\nCréation du graphe de connaissances...")
    G = create_knowledge_graph(triplets, analysis)
    print("Génération de la visualisation...")
    plot_graph(G)
    print("Graphe sauvegardé dans 'knowledge_graph.png'")

if __name__ == "__main__":
    main() 