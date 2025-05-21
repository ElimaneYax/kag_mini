#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Any, List, Optional

from modules.doc_extraction.pdf_loader import PDFLoader
from modules.doc_extraction.text_loader import TextLoader
from modules.doc_extraction.triplet_extractor import TripletExtractor
from modules.semantic_processing.semantic_chunker import SemanticChunker
from modules.semantic_processing.prompt_enhancer import PromptEnhancer
from modules.graph_processing.knowledge_graph import KnowledgeGraph
from modules.graph_processing.neo4j_connector import Neo4jConnector
from modules.llm.nvidia_api import NvidiaLLMClient


class KAGSystem:
    """
    Système intégré pour l'extraction de connaissance, la création de graphe,
    et la génération de réponses basées sur KAG et RAG.
    """
    
    def __init__(self,
                neo4j_uri: str = "bolt://localhost:7687",
                neo4j_user: str = "neo4j",
                neo4j_password: str = "password",
                use_neo4j: bool = True):
        """
        Initialise le système KAG.
        
        Args:
            neo4j_uri (str): URI pour la connexion Neo4j.
            neo4j_user (str): Nom d'utilisateur Neo4j.
            neo4j_password (str): Mot de passe Neo4j.
            use_neo4j (bool): Si True, utilise Neo4j pour le stockage des graphes.
        """
        # Charger les modules
        self.pdf_loader = PDFLoader()
        self.text_loader = TextLoader()
        self.triplet_extractor = TripletExtractor()
        self.semantic_chunker = SemanticChunker()
        self.prompt_enhancer = PromptEnhancer()
        self.knowledge_graph = KnowledgeGraph()
        self.llm_client = NvidiaLLMClient()
        
        # Configuration Neo4j
        self.use_neo4j = use_neo4j
        if use_neo4j:
            self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
            self.neo4j_connected = self.neo4j.test_connection()
            if not self.neo4j_connected:
                print("⚠️ Connexion à Neo4j impossible. Le graphe sera uniquement en mémoire.")
                self.use_neo4j = False
        else:
            self.neo4j = None
            self.neo4j_connected = False
    
    def process_document(self, file_path: str, document_label: str = "Document") -> str:
        """
        Traite un document (PDF ou texte) et génère un graphe de connaissances.
        
        Args:
            file_path (str): Chemin vers le fichier.
            document_label (str): Étiquette à utiliser pour les nœuds dans Neo4j.
            
        Returns:
            str: Message de statut.
        """
        # Déterminer le type de fichier
        _, ext = os.path.splitext(file_path)
        
        # Extraire le texte
        if ext.lower() == '.pdf':
            print(f"Chargement du PDF: {file_path}")
            text = self.pdf_loader.load_pdf(file_path)
        else:
            print(f"Chargement du fichier texte: {file_path}")
            text = self.text_loader.load_text(file_path)
        
        print(f"Document chargé: {len(text)} caractères")
        
        # Extraire les triplets et construire le graphe
        print("Extraction des triplets et construction du graphe...")
        triplets = self.knowledge_graph.add_triplets_from_text(text)
        
        print(f"Graphe construit avec {self.knowledge_graph.get_stats()['node_count']} nœuds "
              f"et {self.knowledge_graph.get_stats()['edge_count']} relations.")
        #############################
            # Affichage des triplets extraits
        print("\n📎 Triplets extraits :")
        for triplet in triplets:
          print(f"  - {triplet[0]} ---[{triplet[1]}]---> {triplet[2]}")

        import json
        from datetime import datetime

        # Nom de fichier basé sur le nom du document
        basename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(basename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{name_without_ext}_triplets_{timestamp}.json"
        
        # Structure à sauvegarder
        triplet_data = {
            "document": basename,
            "triplets": [
                {"subject": s, "relation": r, "object": o}
                for s, r, o in triplets
            ]
        }
        
        # Écriture dans un fichier JSON
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(triplet_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Triplets sauvegardés dans le fichier : {json_filename}")


        # Exporter vers Neo4j si disponible
        if self.use_neo4j and self.neo4j_connected:
            print(f"Exportation du graphe vers Neo4j...")
            success = self.neo4j.export_knowledge_graph(self.knowledge_graph, label=document_label)
            if success:
                return f"Document traité et graphe exporté vers Neo4j avec succès: {len(triplets)} triplets extraits."
            else:
                return f"Document traité mais l'export vers Neo4j a échoué: {len(triplets)} triplets extraits."
        
        return f"Document traité avec succès: {len(triplets)} triplets extraits."
    
    def answer_question(self, 
                       question: str, 
                       enhancement_method: str = "kag_rag",
                       temperature: float = 0.7,
                       model: str = None) -> Dict[str, Any]:
        """
        Répond à une question en utilisant le graphe de connaissances et le modèle LLM.
        
        Args:
            question (str): La question posée.
            enhancement_method (str): Méthode d'amélioration du prompt: "vanilla", "rag", "kag", ou "kag_rag".
            temperature (float): Température pour la génération de texte.
            model (str, optional): Modèle LLM à utiliser (si None, utilise le modèle par défaut).
            
        Returns:
            Dict[str, Any]: Résultat contenant la réponse et des informations supplémentaires.
        """
        # Vérifier si nous avons un graphe
        if self.knowledge_graph.get_stats()['node_count'] == 0:
            return {
                "answer": "Aucun document n'a été chargé. Veuillez d'abord traiter un document.",
                "enhanced_prompt": question,
                "method": "vanilla"
            }
        
        # Récupérer le texte complet (pour RAG)
        if hasattr(self.pdf_loader, 'text') and self.pdf_loader.text:
            document_text = self.pdf_loader.text
        elif hasattr(self.text_loader, 'text') and self.text_loader.text:
            document_text = self.text_loader.text
        else:
            document_text = ""
            print("⚠️ Aucun texte de document disponible pour RAG.")
        
        # Préparer le prompt en fonction de la méthode choisie
        if enhancement_method == "vanilla":
            enhanced_prompt = question
            method = "vanilla"
        elif enhancement_method == "rag":
            enhanced_prompt = self.prompt_enhancer.enhance_with_rag(question, document_text)['enhanced_prompt']
            method = "rag"
        elif enhancement_method == "kag":
            enhanced_prompt = self.prompt_enhancer.enhance_with_kag(question, document_text)['enhanced_prompt']
            method = "kag"
        else:  # kag_rag
            enhanced_prompt = self.prompt_enhancer.enhance_with_kag_rag(question, document_text)['enhanced_prompt']
            method = "kag_rag"
        
        # Interroger le modèle LLM
        answer = self.llm_client.query(
            prompt=enhanced_prompt,
            model=model,
            temperature=temperature
        )
        
        return {
            "answer": answer,
            "enhanced_prompt": enhanced_prompt,
            "method": method
        }
    
    def visualize_graph(self) -> None:
        """
        Visualise le graphe de connaissances.
        """
        if self.knowledge_graph.get_stats()['node_count'] == 0:
            print("Aucun graphe à visualiser. Veuillez d'abord traiter un document.")
            return
        
        self.knowledge_graph.visualize(title="Graphe de connaissances du document")
    
    def get_neo4j_status(self) -> str:
        """
        Renvoie le statut de la connexion Neo4j.
        
        Returns:
            str: Message de statut.
        """
        if not self.use_neo4j:
            return "Neo4j n'est pas activé."
        
        if self.neo4j_connected:
            return "Connecté à Neo4j."
        else:
            connected = self.neo4j.test_connection()
            if connected:
                self.neo4j_connected = True
                return "Connecté à Neo4j."
            else:
                return "Non connecté à Neo4j. Vérifiez les paramètres de connexion."
    
    def clear_knowledge_graph(self) -> str:
        """
        Efface le graphe de connaissances en mémoire et dans Neo4j.
        
        Returns:
            str: Message de statut.
        """
        self.knowledge_graph.clear()
        
        if self.use_neo4j and self.neo4j_connected:
            success = self.neo4j.clear_database(confirm=True)
            if success:
                return "Graphe de connaissances effacé en mémoire et dans Neo4j."
            else:
                return "Graphe de connaissances effacé en mémoire, mais l'effacement dans Neo4j a échoué."
        
        return "Graphe de connaissances effacé en mémoire."


def main():
    """
    Fonction principale pour l'exécution du programme en ligne de commande.
    """
    parser = argparse.ArgumentParser(description='KAG System - Knowledge-Augmented Generation')
    parser.add_argument('--document', '-d', type=str, help='Chemin vers le document à traiter')
    parser.add_argument('--question', '-q', type=str, help='Question à poser au système')
    parser.add_argument('--method', '-m', type=str, default='kag_rag', 
                        choices=['vanilla', 'rag', 'kag', 'kag_rag'],
                        help='Méthode d\'amélioration du prompt')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687',
                        help='URI du serveur Neo4j')
    parser.add_argument('--neo4j_user', type=str, default='neo4j',
                        help='Nom d\'utilisateur Neo4j')
    parser.add_argument('--neo4j_password', type=str, default='password',
                        help='Mot de passe Neo4j')
    parser.add_argument('--no_neo4j', action='store_true',
                        help='Désactiver l\'utilisation de Neo4j')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualiser le graphe de connaissances')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Effacer le graphe de connaissances')
    
    args = parser.parse_args()
    
    # Initialiser le système
    system = KAGSystem(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        use_neo4j=not args.no_neo4j
    )
    
    # Traiter les commandes
    if args.clear:
        result = system.clear_knowledge_graph()
        print(result)
    
    if args.document:
        result = system.process_document(args.document)
        print(result)
    
    if args.visualize:
        system.visualize_graph()
    
    if args.question:
        result = system.answer_question(args.question, enhancement_method=args.method)
        print("\nQuestion: ", args.question)
        print("\nRéponse: ", result["answer"])
        print("\nMéthode utilisée: ", result["method"])
    
    # Si aucune commande spécifiée, afficher l'aide
    if not any([args.document, args.question, args.visualize, args.clear]):
        parser.print_help()


if __name__ == "__main__":
    main() 