#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour tester la connexion à Neo4j et exporter un graphe de connaissances
"""

import os
import sys
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

from modules.doc_extraction.text_loader import TextLoader
from modules.doc_extraction.triplet_extractor import TripletExtractor
from modules.graph_processing.knowledge_graph import KnowledgeGraph
from modules.graph_processing.neo4j_connector import Neo4jConnector


def main():
    """
    Fonction principale pour tester la connexion et l'exportation vers Neo4j
    """
    # Chemin vers le fichier texte à traiter
    input_file = "test.txt"
    
    # Si un fichier est spécifié en argument, l'utiliser à la place
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier '{input_file}' n'existe pas.")
        return
    
    print(f"Configuration Neo4j:")
    print(f"URI: {NEO4J_URI}")
    print(f"Utilisateur: {NEO4J_USER}")
    print(f"Mot de passe: {'*' * len(NEO4J_PASSWORD)}")
    
    # Initialisation du connecteur Neo4j
    neo4j = Neo4jConnector(
        uri=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    # Test de la connexion
    print("\nTest de la connexion à Neo4j...")
    if neo4j.test_connection():
        print("✅ Connexion réussie à Neo4j!")
    else:
        print("❌ Échec de la connexion à Neo4j!")
        return
    
    # Charger le fichier texte
    print(f"\nChargement du fichier texte '{input_file}'...")
    text_loader = TextLoader()
    text = text_loader.load_text(input_file)
    print(f"Texte chargé: {len(text)} caractères")
    
    # Extraire les triplets et créer le graphe
    print("\nExtraction des triplets et création du graphe...")
    kg = KnowledgeGraph()
    triplets = kg.add_triplets_from_text(text)
    
    print(f"Graphe créé avec {kg.get_stats()['node_count']} nœuds et {kg.get_stats()['edge_count']} relations:")
    for i, triplet in enumerate(triplets[:10]):  # Afficher les 10 premiers triplets
        print(f"  {i+1}. {triplet[0]} -> {triplet[1]} -> {triplet[2]}")
    
    if len(triplets) > 10:
        print(f"  ... et {len(triplets) - 10} autres triplets")
    
    # Exporter le graphe vers Neo4j
    print("\nExportation du graphe vers Neo4j...")
    success = neo4j.export_knowledge_graph(kg, label="TestDoc")
    
    if success:
        print("\n✅ Graphe exporté avec succès vers Neo4j!")
        print("\nRequête Cypher pour visualiser le graphe dans Neo4j Browser:")
        print("MATCH (n:TestDoc)-[r]->(m:TestDoc) RETURN n, r, m LIMIT 25")
    else:
        print("\n❌ Échec de l'exportation du graphe vers Neo4j!")


if __name__ == "__main__":
    main() 