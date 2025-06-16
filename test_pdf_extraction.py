#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script générique pour l'extraction de triplets depuis un fichier PDF ou TXT avec le LLM NVIDIA,
avec possibilité d'extraction multi-niveaux.
"""

import json
import argparse
import os
from typing import List, Dict, Any
from modules.doc_extraction.pdf_loader import PDFLoader
from modules.doc_extraction.text_loader import TextLoader
from modules.doc_extraction.triplet_extractor import TripletExtractor
from modules.graph_processing.knowledge_graph import KnowledgeGraph
from modules.graph_processing.neo4j_connector import Neo4jConnector
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def split_text(text: str, max_tokens: int = 2000) -> list:
    """
    Divise le texte en morceaux de taille maximale (tokens estimés).
    Utilise les points comme séparateurs pour garder des phrases cohérentes.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Estimation grossière : 1 token ≈ 4 caractères
        sentence_length = len(sentence) // 4
        
        # Ajoute 1 pour le ". " perdu lors du split
        if current_length + sentence_length + 1 > max_tokens:
            if current_chunk:
                # Joindre les phrases pour former le chunk, ajouter le ". " sauf pour le dernier
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1 # Ajoute 1 pour le ". " entre les phrases
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def format_triplets_for_next_level(triplets: List[Dict[str, Any]], current_level: int) -> str:
    """
    Formate une liste de triplets en texte pour l'extraction du niveau suivant.
    """
    formatted_text = f"Voici une liste de faits extraits (Niveau {current_level}) :\n\n"
    for i, t in enumerate(triplets, 1):
        formatted_text += f"{i}. {t['subject']} --[{t['relation']}]--> {t['object']} (Confidence: {t['confidence']:.2f}, Sentence: \"{t['sentence'].strip()}\")\n"
    formatted_text += "\nExtrayez des relations de niveau supérieur entre ces faits.\n"
    return formatted_text

def main():
    parser = argparse.ArgumentParser(description="Extraction de triplets depuis un fichier PDF ou TXT avec LLM")
    parser.add_argument('--file', type=str, required=True, help='Chemin du fichier à traiter (.pdf ou .txt)')
    parser.add_argument('--label', type=str, default='KnowledgeGraph', help='Label Neo4j pour les nœuds du graphe')
    parser.add_argument('--level', type=int, default=1, help='Niveau d\'extraction des relations (1 pour extraction directe, >1 pour extraction itérative)')
    args = parser.parse_args()

    file_path = args.file
    label = args.label
    max_level = args.level

    if not os.path.isfile(file_path):
        print(f"❌ Fichier non trouvé : {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        print(f"Chargement du PDF: {file_path}")
        loader = PDFLoader()
        text = loader.load_pdf(file_path)
    elif ext == '.txt':
        print(f"Chargement du fichier texte: {file_path}")
        loader = TextLoader()
        text = loader.load_text(file_path)
    else:
        print("❌ Format de fichier non supporté. Utilisez un PDF ou un TXT.")
        return

    print(f"Fichier chargé: {len(text)} caractères")
    
    print("\nInitialisation de l'extracteur de triplets...")
    extractor = TripletExtractor()
    
    all_triplets_across_levels = []
    current_input = text # Pour le niveau 1, l'entrée est le texte
    previous_level_triplets = []

    for current_level in range(1, max_level + 1):
        print(f"\n=== Extraction de Triplet - Niveau {current_level} ===")
        
        if not current_input:
            print(f"Aucune entrée pour le niveau {current_level}, arrêt de l'extraction.")
            break
            
        if current_level > 1:
             # Pour les niveaux > 1, diviser l'entrée (qui est un texte formaté de triplets) en morceaux si nécessaire
             # NOTE: La division par points peut ne pas être idéale ici. Une division plus simple par taille pourrait être préférable pour des listes de triplets.
             # Pour l'instant, on suppose que la liste formatée n'est pas trop grande ou on l'envoie en un seul morceau.
             text_chunks = [current_input] # Envoie tout le texte formaté en un seul morceau pour l'exemple
             print(f"Entrée du niveau {current_level}: {len(current_input)} caractères")
        else:
             # Niveau 1 : Utilise la fonction split_text existante pour le texte brut
             text_chunks = split_text(current_input)
             print(f"Texte divisé en {len(text_chunks)} morceaux pour le niveau {current_level}")
             
        current_level_triplets = []
        
        # Traiter chaque morceau
        for i, chunk in enumerate(text_chunks, 1):
            print(f"Traitement du morceau {i}/{len(text_chunks)} pour le niveau {current_level}...")
            
            # La méthode extract_triplets_with_context dans TripletExtractor devra être capable de gérer
            # un prompt différent pour les niveaux > 1 (basé sur une liste de triplets formatée).
            # Pour l'instant, elle utilise le même prompt, ce qui devra être adapté dans TripletExtractor.
            triplets_from_chunk = extractor.extract_triplets_with_context(chunk)
            
            # Ajouter la propriété de niveau aux triplets extraits
            for t in triplets_from_chunk:
                t['level'] = current_level
                current_level_triplets.append(t)
                
            print(f"Triplets extraits de ce morceau: {len(triplets_from_chunk)}")
        
        print(f"Triplets extraits au niveau {current_level}: {len(current_level_triplets)}")
        all_triplets_across_levels.extend(current_level_triplets)
        previous_level_triplets = current_level_triplets # Les triplets extraits à ce niveau deviennent l'entrée pour le suivant
        
        if not previous_level_triplets and current_level < max_level:
             print(f"Aucun triplet extrait au niveau {current_level}. Arrêt de l'extraction des niveaux supérieurs.")
             break

        # Préparer l'entrée pour le prochain niveau (si ce n'est pas le dernier)
        if current_level < max_level:
            # Convertir les triplets extraits à ce niveau en texte pour le niveau suivant
            current_input = format_triplets_for_next_level(previous_level_triplets, current_level)
            # NOTE: Il faudra adapter le prompt dans TripletExtractor pour qu'il analyse ce format.
        else:
            current_input = None # Pas de prochain niveau

    print(f"\nTotal des triplets extraits sur tous les niveaux: {len(all_triplets_across_levels)}")
    
    # Sauvegarder les triplets dans un fichier JSON pour analyse
    with open('triplets_extracted_multilevel.json', 'w', encoding='utf-8') as f:
        json.dump(all_triplets_across_levels, f, ensure_ascii=False, indent=2)
    print("\nTriplets sauvegardés dans 'triplets_extracted_multilevel.json'")
    
    # Créer le graphe de connaissances
    print("\nCréation du graphe de connaissances...")
    kg = KnowledgeGraph()
    
    # Ajouter les triplets au graphe (avec la propriété 'level')
    for triplet_data in all_triplets_across_levels:
        kg.add_triplet(
            subject=triplet_data['subject'],
            relation=triplet_data['relation'],
            obj=triplet_data['object'],
            properties={
                'confidence': triplet_data.get('confidence'), # Utilise get pour gérer les triplets sans confidence si besoin
                'sentence': triplet_data.get('sentence'),     # Utilise get pour gérer les triplets sans sentence si besoin
                'level': triplet_data.get('level', 1)         # Ajoute la propriété level
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
    kg.visualize(title=f"Graphe de connaissances - {label} (Niveaux {max_level})")
    
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
        # Le label dans Neo4j sera celui spécifié, la propriété 'level' sera sur les relations ou les nœuds (ici sur les relations par défaut)
        success = neo4j.export_knowledge_graph(kg, label=label)
        
        if success:
            print(f"✅ Graphe exporté avec succès vers Neo4j avec le label '{label}'!")
            print("\nRequête Cypher pour visualiser le graphe dans Neo4j Browser (incluant la propriété 'level'):")
            print(f"MATCH (n:{label})-[r]->(m:{label}) RETURN n, r, m LIMIT 50 // Inspectez les propriétés des relations pour voir 'level'")
        else:
            print("❌ Échec de l'exportation du graphe vers Neo4j!")
    else:
        print("❌ Échec de la connexion à Neo4j!")

if __name__ == "__main__":
    main() 