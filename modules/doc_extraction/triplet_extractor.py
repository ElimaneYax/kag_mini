import re
import spacy
from typing import List, Tuple, Dict, Any, Optional, Union

# Chargement du modèle spaCy (à faire à l'initialisation)
# Le modèle peut être changé selon la langue et la qualité nécessaire
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Si le modèle n'est pas disponible, proposer de l'installer
    print("Le modèle spaCy 'en_core_web_sm' n'est pas installé.")
    print("Pour l'installer, exécutez: python -m spacy download en_core_web_sm")


class TripletExtractor:
    """
    Classe pour extraire des triplets (sujet, verbe, objet) à partir de textes.
    """
    
    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialise l'extracteur de triplets avec un modèle de langue.
        
        Args:
            language_model (str): Le modèle de langue spaCy à utiliser.
                                  Par défaut "en_core_web_sm".
        """
        try:
            self.nlp = spacy.load(language_model)
        except OSError:
            raise ValueError(f"Le modèle spaCy '{language_model}' n'est pas installé. "
                            f"Pour l'installer, exécutez: python -m spacy download {language_model}")
    
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extrait les triplets (sujet, verbe, objet) d'un texte.
        
        Args:
            text (str): Le texte à analyser.
            
        Returns:
            List[Tuple[str, str, str]]: Liste de triplets (sujet, verbe, objet).
        """
        triplets = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_triplets = self._extract_triplets_from_sentence(sent)
            triplets.extend(sent_triplets)
        
        return triplets
    
    def _extract_triplets_from_sentence(self, sentence) -> List[Tuple[str, str, str]]:
        """
        Extrait les triplets d'une phrase parsée par spaCy.
        
        Args:
            sentence: La phrase parsée par spaCy.
            
        Returns:
            List[Tuple[str, str, str]]: Liste de triplets extraits.
        """
        triplets = []
        subject = None
        verb = None
        obj = None
        
        # Parcourir les tokens pour identifier les sujets, verbes et objets
        for token in sentence:
            # Identification du sujet
            if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
                subject = self._extract_span_text(token)
            
            # Identification du verbe (prédicat)
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX") and verb is None:
                verb = token.lemma_
            
            # Identification de l'objet
            if token.dep_ in ("dobj", "pobj", "attr") and obj is None:
                obj = self._extract_span_text(token)
        
        # Si nous avons tous les éléments du triplet, l'ajouter à la liste
        if subject and verb and obj:
            triplets.append((subject, verb, obj))
        
        return triplets
    
    def _extract_span_text(self, token) -> str:
        """
        Extrait le texte complet d'un span à partir d'un token.
        Par exemple, pour un sujet composé, extrait le groupe nominal complet.
        
        Args:
            token: Le token spaCy.
            
        Returns:
            str: Le texte du span.
        """
        # Si le token est la tête d'un groupe nominal, extraire le groupe entier
        if token.pos_ in ("NOUN", "PROPN") and list(token.children):
            # Extraire tous les mots dépendants (déterminants, adjectifs, etc.)
            span_tokens = [token]
            for child in token.children:
                if child.dep_ in ("det", "amod", "compound", "nummod") and child.i < token.i:
                    span_tokens.append(child)
            
            # Trier les tokens par position dans la phrase
            span_tokens.sort(key=lambda t: t.i)
            return " ".join([t.text for t in span_tokens])
        
        return token.text
    
    def extract_triplets_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les triplets avec des informations de contexte supplémentaires.
        
        Args:
            text (str): Le texte à analyser.
            
        Returns:
            List[Dict[str, Any]]: Liste de dictionnaires contenant les triplets et leur contexte.
                                  Chaque dictionnaire contient les clés:
                                  - 'subject': Le sujet
                                  - 'verb': Le verbe
                                  - 'object': L'objet
                                  - 'sentence': La phrase complète
                                  - 'confidence': Score de confiance de l'extraction
        """
        results = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_str = sent.text
            triplets = self._extract_triplets_from_sentence(sent)
            
            for subj, verb, obj in triplets:
                # Calcul d'un score de confiance simple basé sur la longueur des éléments
                # Un calcul plus sophistiqué pourrait être implémenté
                confidence = min(1.0, (len(subj) + len(verb) + len(obj)) / len(sent_str) if len(sent_str) > 0 else 0)
                
                results.append({
                    'subject': subj,
                    'verb': verb,
                    'object': obj,
                    'sentence': sent_str,
                    'confidence': confidence
                })
        
        return results
    
    @staticmethod
    def split_text_into_sentences(text: str) -> List[str]:
        """
        Divise un texte en phrases.
        
        Args:
            text (str): Le texte à diviser.
            
        Returns:
            List[str]: Liste des phrases.
        """
        # Expression régulière pour diviser le texte en phrases
        # Recherche un point, un point d'exclamation ou d'interrogation suivi d'un espace
        return re.split(r'(?<=[.!?]) +', text)
    
    def format_triplet_natural(self, triplet: Tuple[str, str, str]) -> str:
        """
        Formate un triplet en une phrase naturelle.
        
        Args:
            triplet (Tuple[str, str, str]): Le triplet (sujet, verbe, objet).
            
        Returns:
            str: Phrase formatée naturellement.
        """
        subj, verb, obj = triplet
        
        # Adapter la conjugaison du verbe si nécessaire
        if not verb.endswith('s') and subj.lower() not in ['i', 'you', 'we', 'they']:
            verb += 's'
        
        return f"{subj} {verb} {obj}" 