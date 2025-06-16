from typing import List, Tuple, Dict, Any, Optional, Union
import json
from modules.llm.nvidia_api import NvidiaLLMClient

class TripletExtractor:
    """
    Classe pour extraire des triplets (sujet, verbe, objet) à partir de textes
    en utilisant un modèle de langage NVIDIA.
    Adapte l'extraction en fonction du type d'entrée (texte brut ou liste de triplets).
    """
    
    # Clé API par défaut (à remplacer par la vôtre ou à définir via variable d'environnement)
    DEFAULT_API_KEY = "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"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise l'extracteur de triplets avec le client NVIDIA LLM.
        """
        self.llm_client = NvidiaLLMClient(api_key=api_key)

    def _get_prompt(self, text: str) -> str:
        """
        Détermine le prompt approprié en fonction du contenu du texte d'entrée.
        """
        # Simple heuristic: if text starts with the formatting used for triplets, it's level > 1 input
        if text.strip().startswith("Voici une liste de faits extraits (Niveau"):
            # Prompt pour l'extraction de niveau supérieur (à partir de triplets existants)
            prompt = f"""Tu es un expert en analyse de connaissances et en synthèse. Ton rôle est d'analyser une liste de triplets de connaissances et d'en extraire des relations de niveau supérieur ou des synthèses importantes, également sous forme de triplets (sujet, relation, objet) avec leur contexte.

Instructions spécifiques :
1. Analyse les triplets fournis et identifie les concepts récurrents ou les relations générales qui émergent.
2. Crée de nouveaux triplets qui synthétisent ou relient ces concepts à un niveau plus élevé.
3. Utilise des relations précises et significatives pour ces relations de haut niveau.
4. Évite de simplement reformuler les triplets de niveau inférieur.
5. Assure-toi que chaque triplet représente une information factuelle claire et de niveau supérieur.
6. Pour chaque nouveau triplet, indique les triplets sources (ou une référence générale au texte source) et un score de confiance pour la synthèse.

Réponds UNIQUEMENT avec une liste JSON au format suivant:
[
    {{
        "subject": "concept_synthèse1",
        "relation": "relation_haut_niveau",
        "object": "concept_synthèse2",
        "sentence": "Synthèse basée sur les faits...", # Ou référence aux triplets sources
        "confidence": 0.8
    }}
]

Liste de faits (triplets de niveau inférieur) à analyser:
{text}
"""
        else:
            # Prompt pour l'extraction de niveau 1 (à partir de texte brut)
            prompt = f"""Tu es un expert en extraction de connaissances. Ton rôle est d'analyser un texte et d'en extraire les faits importants sous forme de triplets (sujet, relation, objet) avec leur contexte.

Instructions spécifiques :\n1. Identifie les concepts clés et leurs relations\n2. Utilise des relations précises et significatives\n3. Évite les relations trop génériques comme "est" ou "a"\n4. Préfère les relations comme "est_composé_de", "permet_de", "utilise", "définit", etc.\n5. Assure-toi que chaque triplet représente une information factuelle claire\n6. Pour chaque triplet, indique la phrase source et un score de confiance\n\nRéponds UNIQUEMENT avec une liste JSON au format suivant:
[
    {{
        "subject": "concept1",
        "relation": "relation_précise",
        "object": "concept2",
        "sentence": "phrase source complète",
        "confidence": 0.95
    }}
]

Texte à analyser:
{text}
"""

        return prompt
        
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extrait les triplets (sujet, verbe, objet) d'un texte en utilisant le LLM.
        (Cette méthode n'est plus la principale, préférer extract_triplets_with_context pour les niveaux).
        """
        # Cette méthode pourrait potentiellement utiliser _get_prompt(text) aussi
        prompt = self._get_prompt(text) # Utilise le même prompt que _with_context pour la cohérence
        
        response = self.llm_client.query(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Gérer le cas où la réponse est None
        if response is None:
            print("Erreur: La requête LLM a retourné None.")
            return []

        # Le parsing ici est plus simple car on attend juste [{}, {}]
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                triplets_data = json.loads(json_str)
                return [(t.get('subject'), t.get('relation'), t.get('object')) 
                        for t in triplets_data if 'subject' in t and 'relation' in t and 'object' in t]
            else:
                print("Aucun JSON valide trouvé dans la réponse du LLM")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON: {e}")
            # Afficher la réponse brute pour debug en cas d'erreur de parsing
            print("\n--- Réponse brute du LLM qui a causé l'erreur ---")
            print(response)
            print("--- Fin de la réponse brute ---\n")
            return []

    def extract_triplets_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les triplets avec des informations de contexte supplémentaires, en adaptant le prompt au niveau.
        """
        prompt = self._get_prompt(text) # Détermine le prompt basé sur l'entrée

        response = self.llm_client.query(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Gérer le cas où la réponse est None
        if response is None:
             print("Erreur: La requête LLM a retourné None dans extract_triplets_with_context.")
             return []

        # Afficher la réponse brute pour debug (déjà présent, le garde pour l'instant)
        print("\n--- Réponse brute du LLM ---")
        print(response)
        print("--- Fin de la réponse brute ---\n")
        
        # Sauvegarder la réponse brute dans un fichier (optionnel, déjà présent, le garde)
        with open('llm_raw_response.txt', 'a', encoding='utf-8') as f:
            f.write("\n--- Réponse brute du LLM ---\n")
            f.write(response)
            f.write("\n--- Fin de la réponse brute ---\n")
        
        try:
            # Extraire la partie JSON de la réponse
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                triplets_data = json.loads(json_str)
                
                # S'assurer que chaque élément est un dictionnaire et contient les clés nécessaires
                valid_triplets = []
                for t in triplets_data:
                    if isinstance(t, dict) and 'subject' in t and 'relation' in t and 'object' in t:
                         # Utiliser .get() pour les clés optionnelles comme 'sentence' et 'confidence'
                         valid_triplets.append({
                             'subject': t['subject'],
                             'relation': t['relation'],
                             'object': t['object'],
                             'sentence': t.get('sentence', ''), # Fournit une chaîne vide par défaut
                             'confidence': t.get('confidence', None) # Fournit None par défaut
                         })
                    else:
                         print(f"Avertissement: Format de triplet inattendu ou clé(s) manquante(s): {t}")
                         
                return valid_triplets
                
            else:
                print("Aucun JSON valide (liste de dictionnaires) trouvé dans la réponse du LLM")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON: {e}")
            # Afficher la réponse brute pour debug en cas d'erreur de parsing
            print("\n--- Réponse brute du LLM qui a causé l'erreur ---")
            print(response)
            print("--- Fin de la réponse brute ---\n")
            return []
    
    def format_triplet_natural(self, triplet: Tuple[str, str, str]) -> str:
        """
        Formate un triplet en une phrase naturelle.
        """
        subj, verb, obj = triplet
        return f"{subj} {verb} {obj}" 