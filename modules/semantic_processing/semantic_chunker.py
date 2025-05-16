import spacy
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import torch


class SemanticChunker:
    """
    Classe pour découper un texte en chunks sémantiquement cohérents.
    """
    
    def __init__(self, 
                language_model: str = "en_core_web_sm", 
                embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialise le chunker sémantique.
        
        Args:
            language_model (str): Modèle spaCy à utiliser pour l'analyse linguistique.
            embedding_model (str): Modèle Sentence Transformer pour les embeddings.
        """
        # Charger le modèle spaCy
        try:
            self.nlp = spacy.load(language_model)
        except OSError:
            raise ValueError(f"Le modèle spaCy '{language_model}' n'est pas installé. "
                            f"Pour l'installer, exécutez: python -m spacy download {language_model}")
        
        # Charger le modèle d'embedding
        try:
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du modèle d'embedding '{embedding_model}': {e}")
    
    def chunk_text(self, text: str, max_tokens: int = 300) -> List[str]:
        """
        Découpe un texte en chunks de taille maximale spécifiée.
        
        Args:
            text (str): Le texte à découper.
            max_tokens (int): Nombre maximum de tokens par chunk.
            
        Returns:
            List[str]: Liste des chunks.
        """
        doc = self.nlp(text)
        chunks, current_chunk = [], ""
        current_len = 0
        
        for sent in doc.sents:
            token_count = len(sent)
            
            if current_len + token_count > max_tokens:
                # Si le chunk courant dépasse la taille maximale, commencer un nouveau
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = str(sent)
                current_len = token_count
            else:
                # Sinon, ajouter la phrase au chunk courant
                current_chunk += " " + str(sent)
                current_len += token_count
        
        # Ajouter le dernier chunk s'il existe
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def semantic_chunk_text(self, text: str, max_tokens: int = 300, similarity_threshold: float = 0.5) -> List[str]:
        """
        Découpe un texte en chunks sémantiquement cohérents.
        
        Args:
            text (str): Le texte à découper.
            max_tokens (int): Nombre maximum de tokens par chunk.
            similarity_threshold (float): Seuil de similarité pour considérer qu'une phrase appartient au chunk courant.
            
        Returns:
            List[str]: Liste des chunks sémantiquement cohérents.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return []
        
        # Calculer les embeddings des phrases
        sentence_texts = [str(sent) for sent in sentences]
        embeddings = self.embedder.encode(sentence_texts, convert_to_tensor=True)
        
        chunks = []
        current_chunk_sentences = [sentence_texts[0]]
        current_chunk_embedding = embeddings[0].unsqueeze(0)
        current_len = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentence_texts[i]
            token_count = len(sentences[i])
            sentence_embedding = embeddings[i].unsqueeze(0)
            
            # Calculer la similarité avec le chunk courant
            similarity = torch.nn.functional.cosine_similarity(
                sentence_embedding, 
                current_chunk_embedding.mean(dim=0, keepdim=True)
            ).item()
            
            # Conditions pour ajouter au chunk courant ou commencer un nouveau
            if (current_len + token_count <= max_tokens and similarity >= similarity_threshold):
                # Ajouter au chunk courant si la similarité est suffisante et qu'on ne dépasse pas la taille
                current_chunk_sentences.append(sentence)
                current_chunk_embedding = torch.cat([current_chunk_embedding, sentence_embedding])
                current_len += token_count
            else:
                # Sinon, commencer un nouveau chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_embedding
                current_len = token_count
        
        # Ajouter le dernier chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return chunks
    
    def chunk_by_sections(self, text: str, section_markers: List[str] = None) -> Dict[str, str]:
        """
        Découpe un texte en sections en utilisant des marqueurs spécifiques.
        
        Args:
            text (str): Le texte à découper.
            section_markers (List[str], optional): Liste des marqueurs de section.
                Si None, utilise des marqueurs par défaut comme 'Introduction', 'Conclusion', etc.
            
        Returns:
            Dict[str, str]: Dictionnaire {nom_section: contenu}.
        """
        if section_markers is None:
            section_markers = [
                "Abstract", "Introduction", "Related Work", "Methodology", 
                "Results", "Discussion", "Conclusion", "References"
            ]
        
        sections = {}
        current_section = "Preamble"
        current_content = []
        
        for line in text.split('\n'):
            matched = False
            for marker in section_markers:
                # Vérifier si la ligne contient un marqueur de section
                if marker.lower() in line.lower() and len(line) < 100:  # Éviter les faux positifs dans les phrases longues
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = marker
                    current_content = []
                    matched = True
                    break
            
            if not matched:
                current_content.append(line)
        
        # Ajouter la dernière section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Récupère les chunks les plus pertinents pour une requête donnée.
        
        Args:
            query (str): La requête.
            chunks (List[str]): Liste des chunks de texte.
            top_k (int): Nombre de chunks à récupérer.
            
        Returns:
            List[Tuple[str, float]]: Liste de tuples (chunk, score de similarité).
        """
        # Encoder la requête et les chunks
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.embedder.encode(chunks, convert_to_tensor=True)
        
        # Calculer les similarités
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            chunk_embeddings
        )
        
        # Récupérer les chunks les plus pertinents
        top_k = min(top_k, len(chunks))
        top_indices = torch.topk(similarities, k=top_k).indices
        
        return [(chunks[i], similarities[i].item()) for i in top_indices]
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extrait les mots-clés les plus importants d'un texte.
        
        Args:
            text (str): Le texte à analyser.
            max_keywords (int): Nombre maximum de mots-clés à extraire.
            
        Returns:
            List[str]: Liste des mots-clés extraits.
        """
        doc = self.nlp(text)
        keywords = []
        
        # Extraire les noms et adjectifs comme candidats mots-clés
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop and len(token.text) > 2:
                keywords.append(token.text.lower())
        
        # Compter les occurrences
        keyword_freq = {}
        for kw in keywords:
            if kw in keyword_freq:
                keyword_freq[kw] += 1
            else:
                keyword_freq[kw] = 1
        
        # Trier par fréquence et prendre les top N
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:max_keywords]] 