import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Set, Union
import re

from modules.doc_extraction.triplet_extractor import TripletExtractor


class KnowledgeGraph:
    """
    Classe pour créer et manipuler un graphe de connaissances à partir de triplets.
    """
    
    def __init__(self):
        """
        Initialise un graphe de connaissances vide.
        """
        # Graphe principal
        self.graph = nx.DiGraph()
        
        # Extracteur de triplets pour générer des triplets à partir de texte
        self.triplet_extractor = TripletExtractor()
        
        # Statistiques du graphe
        self.stats = {
            'node_count': 0,
            'edge_count': 0,
            'subject_count': 0,
            'object_count': 0,
            'relation_types': set()
        }
    
    def add_triplet(self, subject: str, relation: str, obj: str, properties: Dict[str, Any] = None) -> None:
        """
        Ajoute un triplet au graphe.
        
        Args:
            subject (str): Le sujet du triplet.
            relation (str): La relation entre le sujet et l'objet.
            obj (str): L'objet du triplet.
            properties (Dict[str, Any], optional): Propriétés additionnelles à ajouter à l'arête.
        """
        # Ajouter les nœuds s'ils n'existent pas
        if subject not in self.graph.nodes:
            self.graph.add_node(subject, type='subject')
            self.stats['subject_count'] += 1
        
        if obj not in self.graph.nodes:
            self.graph.add_node(obj, type='object')
            self.stats['object_count'] += 1
        
        # Préparer les propriétés de l'arête
        edge_props = {'relation': relation}
        if properties:
            edge_props.update(properties)
        
        # Ajouter l'arête
        self.graph.add_edge(subject, obj, **edge_props)
        
        # Mettre à jour les statistiques
        self.stats['node_count'] = self.graph.number_of_nodes()
        self.stats['edge_count'] = self.graph.number_of_edges()
        self.stats['relation_types'].add(relation)
    
    def add_triplets_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extrait et ajoute des triplets à partir d'un texte.
        
        Args:
            text (str): Le texte source.
            
        Returns:
            List[Tuple[str, str, str]]: Liste des triplets extraits et ajoutés.
        """
        # Extraire les triplets
        triplets = self.triplet_extractor.extract_triplets(text)
        
        # Ajouter chaque triplet au graphe
        for subj, verb, obj in triplets:
            self.add_triplet(subj, verb, obj)
        
        return triplets
    
    def add_triplets_from_list(self, triplets: List[Tuple[str, str, str]]) -> None:
        """
        Ajoute une liste de triplets au graphe.
        
        Args:
            triplets (List[Tuple[str, str, str]]): Liste de triplets (sujet, relation, objet).
        """
        for subj, rel, obj in triplets:
            self.add_triplet(subj, rel, obj)
    
    def visualize(self, 
                 title: str = "Knowledge Graph", 
                 node_size: int = 1500, 
                 font_size: int = 10,
                 figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualise le graphe de connaissances.
        
        Args:
            title (str): Titre du graphique.
            node_size (int): Taille des nœuds.
            font_size (int): Taille de la police.
            figsize (Tuple[int, int]): Taille de la figure (largeur, hauteur).
        """
        plt.figure(figsize=figsize)
        
        # Créer une disposition en utilisant l'algorithme spring layout
        pos = nx.spring_layout(self.graph, k=0.5, seed=42)
        
        # Dessiner les nœuds
        nx.draw(self.graph, pos, with_labels=True, 
                node_size=node_size, node_color='lightblue', 
                font_size=font_size, font_weight='bold')
        
        # Récupérer les étiquettes des arêtes (relations)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        
        # Dessiner les étiquettes des arêtes
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title(title)
        plt.axis('off')  # Désactiver les axes
        plt.tight_layout()
        plt.show()
    
    def get_subgraph_for_node(self, node: str, distance: int = 1) -> nx.DiGraph:
        """
        Récupère un sous-graphe centré sur un nœud spécifique.
        
        Args:
            node (str): Nœud central.
            distance (int): Distance maximale depuis le nœud central.
            
        Returns:
            nx.DiGraph: Sous-graphe.
        """
        if node not in self.graph.nodes:
            return nx.DiGraph()
        
        # Commencer avec le nœud central
        nodes_to_include = {node}
        
        # Parcourir les voisins jusqu'à la distance spécifiée
        for _ in range(distance):
            new_nodes = set()
            for n in nodes_to_include:
                # Ajouter les successeurs (arêtes sortantes)
                new_nodes.update(self.graph.successors(n))
                # Ajouter les prédécesseurs (arêtes entrantes)
                new_nodes.update(self.graph.predecessors(n))
            nodes_to_include.update(new_nodes)
        
        # Créer le sous-graphe avec les nœuds collectés
        return self.graph.subgraph(nodes_to_include).copy()
    
    def search_nodes(self, query: str) -> List[str]:
        """
        Recherche des nœuds dans le graphe par une correspondance textuelle.
        
        Args:
            query (str): Texte à rechercher.
            
        Returns:
            List[str]: Liste des nœuds correspondants.
        """
        matching_nodes = []
        query_lower = query.lower()
        
        for node in self.graph.nodes:
            if query_lower in node.lower():
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_relations_between(self, node1: str, node2: str) -> List[Dict[str, Any]]:
        """
        Récupère toutes les relations directes entre deux nœuds.
        
        Args:
            node1 (str): Premier nœud.
            node2 (str): Second nœud.
            
        Returns:
            List[Dict[str, Any]]: Liste des relations avec leurs propriétés.
        """
        relations = []
        
        # Vérifier si les nœuds existent
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return relations
        
        # Vérifier les arêtes directes de node1 à node2
        if self.graph.has_edge(node1, node2):
            edge_data = self.graph.get_edge_data(node1, node2)
            for key, attrs in edge_data.items():
                relation_info = {'direction': 'outgoing', 'other_node': node2}
                relation_info.update(attrs)
                relations.append(relation_info)
        
        # Vérifier les arêtes directes de node2 à node1
        if self.graph.has_edge(node2, node1):
            edge_data = self.graph.get_edge_data(node2, node1)
            for key, attrs in edge_data.items():
                relation_info = {'direction': 'incoming', 'other_node': node2}
                relation_info.update(attrs)
                relations.append(relation_info)
        
        return relations
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques actuelles du graphe.
        
        Returns:
            Dict[str, Any]: Dictionnaire des statistiques.
        """
        # Mettre à jour les statistiques
        self.stats['node_count'] = self.graph.number_of_nodes()
        self.stats['edge_count'] = self.graph.number_of_edges()
        
        # Convertir le set en liste pour le retour
        stats_copy = self.stats.copy()
        stats_copy['relation_types'] = list(self.stats['relation_types'])
        
        return stats_copy
    
    def clear(self) -> None:
        """
        Efface le graphe.
        """
        self.graph.clear()
        self.stats = {
            'node_count': 0,
            'edge_count': 0,
            'subject_count': 0,
            'object_count': 0,
            'relation_types': set()
        }
    
    def save_to_graphml(self, filename: str) -> None:
        """
        Enregistre le graphe au format GraphML.
        
        Args:
            filename (str): Nom du fichier de sortie.
        """
        nx.write_graphml(self.graph, filename)
    
    def load_from_graphml(self, filename: str) -> None:
        """
        Charge un graphe depuis un fichier GraphML.
        
        Args:
            filename (str): Nom du fichier à charger.
        """
        self.graph = nx.read_graphml(filename)
        
        # Recalculer les statistiques
        self.stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'subject_count': 0,
            'object_count': 0,
            'relation_types': set()
        }
        
        # Reconstruire les statistiques des types de nœuds et de relations
        for _, attrs in self.graph.nodes(data=True):
            if 'type' in attrs:
                if attrs['type'] == 'subject':
                    self.stats['subject_count'] += 1
                elif attrs['type'] == 'object':
                    self.stats['object_count'] += 1
        
        for _, _, attrs in self.graph.edges(data=True):
            if 'relation' in attrs:
                self.stats['relation_types'].add(attrs['relation'])
    

    
    def compare_to_graph(self, other_graph: nx.DiGraph) -> Dict[str, float]:
        """
        Compare ce graphe avec un autre graphe.
        
        Args:
            other_graph (nx.DiGraph): L'autre graphe à comparer.
            
        Returns:
            Dict[str, float]: Métriques de similarité entre les graphes.
        """
        # Ensembles de nœuds et d'arêtes
        nodes1 = set(self.graph.nodes())
        nodes2 = set(other_graph.nodes())
        
        # Créer des tuples (source, cible, relation) pour les arêtes
        edges1 = set((u, v, d.get('relation', '')) for u, v, d in self.graph.edges(data=True))
        edges2 = set((u, v, d.get('relation', '')) for u, v, d in other_graph.edges(data=True))
        
        # Calcul des similarités de Jaccard
        jaccard_nodes = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 | nodes2 else 0
        jaccard_edges = len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 0
        
        return {
            'node_similarity': jaccard_nodes,
            'edge_similarity': jaccard_edges,
            'overall_similarity': (jaccard_nodes + jaccard_edges) / 2
        } 