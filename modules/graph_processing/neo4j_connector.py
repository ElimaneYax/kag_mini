from py2neo import Graph, Node, Relationship, NodeMatcher
from typing import List, Dict, Tuple, Any, Optional, Union
import networkx as nx
import os

from modules.graph_processing.knowledge_graph import KnowledgeGraph


class Neo4jConnector:
    """
    Classe pour interagir avec une base de données Neo4j et y stocker des graphes de connaissances.
    """
    
    def __init__(self, 
                uri: str = "bolt://localhost:7687", 
                username: str = "neo4j", 
                password: str = "password"):
        """
        Initialise la connexion à Neo4j.
        
        Args:
            uri (str): URI du serveur Neo4j.
            username (str): Nom d'utilisateur.
            password (str): Mot de passe.
        """
        # Définir les paramètres de connexion
        self.uri = uri
        self.username = username
        self.password = password
        
        # Utiliser les variables d'environnement si disponibles
        self.uri = os.environ.get("NEO4J_URI", self.uri)
        self.username = os.environ.get("NEO4J_USERNAME", self.username)
        self.password = os.environ.get("NEO4J_PASSWORD", self.password)
        
        # Connexion à Neo4j (établie à la demande)
        self._graph = None
        self._connected = False
    
    def connect(self) -> bool:
        """
        Établit la connexion avec Neo4j.
        
        Returns:
            bool: True si connexion établie, False sinon.
        """
        try:
            self._graph = Graph(self.uri, auth=(self.username, self.password))
            self._connected = True
            return True
        except Exception as e:
            print(f"Erreur de connexion à Neo4j: {e}")
            self._connected = False
            return False
    
    @property
    def graph(self) -> Optional[Graph]:
        """
        Récupère l'objet Graph py2neo.
        
        Returns:
            Graph: L'objet Graph Neo4j.
        """
        if not self._connected:
            self.connect()
        return self._graph
    
    def test_connection(self) -> bool:
        """
        Teste la connexion à Neo4j.
        
        Returns:
            bool: True si connexion fonctionne, False sinon.
        """
        if not self._connected:
            return self.connect()
        
        try:
            # Simple requête pour vérifier la connexion
            self._graph.run("MATCH (n) RETURN count(n) LIMIT 1")
            return True
        except Exception as e:
            print(f"La connexion à Neo4j a échoué: {e}")
            self._connected = False
            return False
    
    def clear_database(self, confirm: bool = False) -> bool:
        """
        Supprime tous les nœuds et relations de la base de données.
        
        Args:
            confirm (bool): Confirmation de suppression.
            
        Returns:
            bool: True si suppression réussie, False sinon.
        """
        if not confirm:
            print("Opération annulée. Pour confirmer la suppression de toutes les données, "
                 "appelez cette méthode avec confirm=True")
            return False
        
        if not self.test_connection():
            return False
        
        try:
            self._graph.run("MATCH (n) DETACH DELETE n")
            print("Base de données effacée avec succès.")
            return True
        except Exception as e:
            print(f"Erreur lors de l'effacement de la base de données: {e}")
            return False
    
    def export_knowledge_graph(self, kg: KnowledgeGraph, label: str = "Document") -> bool:
        """
        Exporte un graphe de connaissances dans Neo4j.
        
        Args:
            kg (KnowledgeGraph): Le graphe de connaissances à exporter.
            label (str): Étiquette à attribuer aux nœuds.
            
        Returns:
            bool: True si l'export a réussi, False sinon.
        """
        if not self.test_connection():
            return False
        
        try:
            # Créer une transaction
            tx = self._graph.begin()
            
            # Dictionnaire pour garder une trace des nœuds déjà créés
            created_nodes = {}
            
            # Créer les nœuds et les relations
            for subject, obj, attrs in kg.graph.edges(data=True):
                # Créer le nœud sujet s'il n'existe pas déjà
                if subject not in created_nodes:
                    subj_node = Node(label, name=subject, type="subject")
                    tx.create(subj_node)
                    created_nodes[subject] = subj_node
                else:
                    subj_node = created_nodes[subject]
                
                # Créer le nœud objet s'il n'existe pas déjà
                if obj not in created_nodes:
                    obj_node = Node(label, name=obj, type="object")
                    tx.create(obj_node)
                    created_nodes[obj] = obj_node
                else:
                    obj_node = created_nodes[obj]
                
                # Créer la relation
                relation = attrs.get('relation', 'RELATED_TO')
                rel = Relationship(subj_node, relation, obj_node)
                
                # Ajouter d'autres propriétés à la relation
                for key, value in attrs.items():
                    if key != 'relation':
                        rel[key] = value
                
                tx.create(rel)
            
            # Valider la transaction
            tx.commit()
            
            print(f"Graphe exporté avec succès avec {len(created_nodes)} nœuds.")
            return True
        
        except Exception as e:
            print(f"Erreur lors de l'export du graphe: {e}")
            return False
    
    def import_to_knowledge_graph(self, query: str = None, label: str = None) -> KnowledgeGraph:
        """
        Importe un graphe depuis Neo4j dans un objet KnowledgeGraph.
        
        Args:
            query (str, optional): Requête Cypher pour filtrer les données.
            label (str, optional): Étiquette des nœuds à importer.
            
        Returns:
            KnowledgeGraph: Graphe de connaissances importé.
        """
        if not self.test_connection():
            return KnowledgeGraph()
        
        kg = KnowledgeGraph()
        
        try:
            if query:
                # Utiliser la requête personnalisée
                result = self._graph.run(query)
            elif label:
                # Requête basée sur l'étiquette
                result = self._graph.run(
                    f"MATCH (s:{label})-[r]->(o:{label}) "
                    "RETURN s.name AS subject, type(r) AS relation, o.name AS object"
                )
            else:
                # Requête par défaut pour tous les nœuds
                result = self._graph.run(
                    "MATCH (s)-[r]->(o) "
                    "RETURN s.name AS subject, type(r) AS relation, o.name AS object"
                )
            
            # Parcourir les résultats et construire le graphe
            for record in result:
                subject = record["subject"]
                relation = record["relation"]
                obj = record["object"]
                
                if subject and relation and obj:
                    kg.add_triplet(subject, relation, obj)
            
            print(f"Graphe importé avec succès: {kg.get_stats()['node_count']} nœuds, "
                 f"{kg.get_stats()['edge_count']} relations.")
            
            return kg
        
        except Exception as e:
            print(f"Erreur lors de l'import du graphe: {e}")
            return kg
    
    def run_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Exécute une requête Cypher personnalisée.
        
        Args:
            query (str): Requête Cypher.
            params (Dict[str, Any], optional): Paramètres pour la requête.
            
        Returns:
            List[Dict[str, Any]]: Résultats de la requête.
        """
        if not self.test_connection():
            return []
        
        try:
            result = self._graph.run(query, parameters=params or {})
            return [dict(record) for record in result]
        
        except Exception as e:
            print(f"Erreur lors de l'exécution de la requête: {e}")
            return []
    
    def get_node_by_name(self, name: str, label: str = None) -> Optional[Dict[str, Any]]:
        """
        Récupère un nœud par son nom.
        
        Args:
            name (str): Nom du nœud.
            label (str, optional): Étiquette du nœud.
            
        Returns:
            Optional[Dict[str, Any]]: Propriétés du nœud ou None si pas trouvé.
        """
        if not self.test_connection():
            return None
        
        try:
            matcher = NodeMatcher(self._graph)
            
            if label:
                node = matcher.match(label, name=name).first()
            else:
                node = matcher.match(name=name).first()
            
            if node:
                return dict(node)
            return None
        
        except Exception as e:
            print(f"Erreur lors de la recherche du nœud: {e}")
            return None
    
    def get_related_nodes(self, 
                         node_name: str, 
                         relation_type: str = None, 
                         direction: str = "both") -> List[Dict[str, Any]]:
        """
        Récupère les nœuds liés à un nœud donné.
        
        Args:
            node_name (str): Nom du nœud de référence.
            relation_type (str, optional): Type de relation à filtrer.
            direction (str): Direction des relations: "outgoing", "incoming" ou "both".
            
        Returns:
            List[Dict[str, Any]]: Liste des nœuds liés avec leurs relations.
        """
        if not self.test_connection():
            return []
        
        try:
            if direction == "outgoing":
                direction_query = "-[r]->"
            elif direction == "incoming":
                direction_query = "<-[r]-"
            else:  # both
                direction_query = "-[r]-"
            
            rel_type = f":{relation_type}" if relation_type else ""
            
            query = (
                f"MATCH (s {{name: $name}}){direction_query}{rel_type}(o) "
                f"RETURN o.name AS related_node, type(r) AS relation, "
                f"CASE WHEN s-->o THEN 'outgoing' ELSE 'incoming' END AS direction"
            )
            
            return self.run_query(query, {"name": node_name})
        
        except Exception as e:
            print(f"Erreur lors de la récupération des nœuds liés: {e}")
            return [] 