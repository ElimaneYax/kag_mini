#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour l'extraction de triplets avec le LLM NVIDIA
"""

import json
from modules.doc_extraction.triplet_extractor import TripletExtractor
from modules.graph_processing.knowledge_graph import KnowledgeGraph
from modules.graph_processing.neo4j_connector import Neo4jConnector
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def main():
    # Texte de test
    test_text = """
    L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de simuler l'intelligence humaine.
    Le deep learning est une sous-catégorie du machine learning qui utilise des réseaux de neurones artificiels.
    Les réseaux de neurones sont inspirés du fonctionnement du cerveau humain.
    Le NLP (Natural Language Processing) permet aux machines de comprendre et de générer du langage naturel.
    Les modèles de langage comme GPT-3 peuvent générer du texte de manière autonome.
    """
    
    print("Initialisation de l'extracteur de triplets...")
    extractor = TripletExtractor()
    
    print("\nExtraction des triplets avec contexte...")
    triplets_with_context = extractor.extract_triplets_with_context(test_text)
    
    print("\nTriplets extraits avec contexte:")
    print(json.dumps(triplets_with_context, indent=2, ensure_ascii=False))
    
    # Créer le graphe de connaissances
    print("\nCréation du graphe de connaissances...")
    kg = KnowledgeGraph()
    
    # Ajouter les triplets au graphe
    for triplet_data in triplets_with_context:
        kg.add_triplet(
            subject=triplet_data['subject'],
            relation=triplet_data['relation'],
            obj=triplet_data['object'],
            properties={
                'confidence': triplet_data['confidence'],
                'sentence': triplet_data['sentence']
            }
        )
    
    # Afficher les statistiques du graphe
    stats = kg.get_stats()
    print(f"\nStatistiques du graphe:")
    print(f"Nombre de nœuds: {stats['node_count']}")
    print(f"Nombre de relations: {stats['edge_count']}")
    print(f"Types de relations: {stats['relation_types']}")
    
    # Visualiser le graphe
    print("\nVisualisation du graphe...")
    kg.visualize(title="Graphe de connaissances extrait avec LLM")
    
    # Exporter vers Neo4j
    print("\nConnexion à Neo4j...")
    neo4j = Neo4jConnector(
        uri=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    if neo4j.test_connection():
        print("Connexion à Neo4j réussie!")
        print("\nExportation du graphe vers Neo4j...")
        success = neo4j.export_knowledge_graph(kg, label="LLMExtracted")
        
        if success:
            print("✅ Graphe exporté avec succès vers Neo4j!")
            print("\nRequête Cypher pour visualiser le graphe dans Neo4j Browser:")
            print("MATCH (n:LLMExtracted)-[r]->(m:LLMExtracted) RETURN n, r, m LIMIT 25")
        else:
            print("❌ Échec de l'exportation du graphe vers Neo4j!")
    else:
        print("❌ Échec de la connexion à Neo4j!")

if __name__ == "__main__":
    main() 