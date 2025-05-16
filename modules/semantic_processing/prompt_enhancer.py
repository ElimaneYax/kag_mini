from typing import List, Dict, Tuple, Any, Optional
import re
from sentence_transformers import SentenceTransformer, util
import torch

from modules.doc_extraction.triplet_extractor import TripletExtractor
from modules.semantic_processing.semantic_chunker import SemanticChunker


class PromptEnhancer:
    """
    Classe pour améliorer les prompts en utilisant les techniques RAG et KAG.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialise le PromptEnhancer.
        
        Args:
            embedding_model (str): Modèle d'embedding à utiliser.
        """
        self.triplet_extractor = TripletExtractor()
        self.semantic_chunker = SemanticChunker(embedding_model=embedding_model)
        self.embedder = SentenceTransformer(embedding_model)
    
    def extract_acronyms(self, text: str) -> List[str]:
        """
        Extrait les acronymes d'un texte.
        
        Args:
            text (str): Texte à analyser.
            
        Returns:
            List[str]: Liste des acronymes trouvés.
        """
        return list(set(re.findall(r'\b[A-Z]{2,10}\b', text)))
    
    def format_triplet_natural(self, triplet: Tuple[str, str, str]) -> str:
        """
        Formate un triplet en langage naturel.
        
        Args:
            triplet (Tuple[str, str, str]): Triplet (sujet, verbe, objet).
            
        Returns:
            str: Triplet formaté en langage naturel.
        """
        subj, verb, obj = triplet
        
        # Conjuguer le verbe correctement
        if not verb.endswith('s') and subj.lower() not in ['i', 'you', 'we', 'they']:
            verb += 's'
        
        return f"{subj} {verb} {obj}"
    
    def enhance_with_rag(self, 
                         prompt: str, 
                         document_text: str, 
                         top_k: int = 3, 
                         max_tokens: int = 300) -> Dict[str, Any]:
        """
        Améliore un prompt en utilisant RAG (Retrieval-Augmented Generation).
        
        Args:
            prompt (str): Le prompt original.
            document_text (str): Le texte du document.
            top_k (int): Nombre de chunks à récupérer.
            max_tokens (int): Taille maximale des chunks.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant:
                - 'enhanced_prompt': Prompt amélioré avec RAG
                - 'retrieved_chunks': Chunks récupérés
                - 'chunk_scores': Scores de similarité des chunks
        """
        # Découper le document en chunks sémantiques
        text_chunks = self.semantic_chunker.semantic_chunk_text(document_text, max_tokens=max_tokens)
        
        # Encoder les chunks et le prompt
        encoded_chunks = self.embedder.encode(text_chunks, convert_to_tensor=True)
        encoded_question = self.embedder.encode(prompt, convert_to_tensor=True)
        
        # Calculer les similarités
        chunk_scores = util.pytorch_cos_sim(encoded_question, encoded_chunks)[0]
        top_indices = torch.topk(chunk_scores, k=min(top_k, len(text_chunks))).indices.tolist()
        
        # Récupérer les chunks les plus pertinents
        retrieved_chunks = [text_chunks[i] for i in top_indices]
        retrieved_scores = [chunk_scores[i].item() for i in top_indices]
        
        # Formater le contexte RAG
        rag_context = "\n\n".join([f"\"{chunk.strip()}\"" for chunk in retrieved_chunks])
        
        # Construire le prompt amélioré
        enhanced_prompt = f"""Vous êtes un assistant répondant uniquement sur la base des informations fournies dans le document.

Preuves contextuelles du document:
{rag_context}

Maintenant, répondez à la question: {prompt}
"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'retrieved_chunks': retrieved_chunks,
            'chunk_scores': retrieved_scores
        }
    
    def enhance_with_kag(self, 
                        prompt: str, 
                        document_text: str, 
                        max_triplets: int = 5) -> Dict[str, Any]:
        """
        Améliore un prompt en utilisant KAG (Knowledge-Augmented Generation).
        
        Args:
            prompt (str): Le prompt original.
            document_text (str): Le texte du document.
            max_triplets (int): Nombre maximum de triplets à inclure.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant:
                - 'enhanced_prompt': Prompt amélioré avec KAG
                - 'selected_triplets': Triplets sélectionnés
                - 'triplet_scores': Scores de similarité des triplets
        """
        # Extraire tous les triplets du document
        all_triplets = self.triplet_extractor.extract_triplets(document_text)
        
        if not all_triplets:
            return {
                'enhanced_prompt': prompt,
                'selected_triplets': [],
                'triplet_scores': []
            }
        
        # Créer des phrases à partir des triplets
        triplet_sentences = [f"{s} {v} {o}" for (s, v, o) in all_triplets]
        
        # Encoder les phrases et le prompt
        encoded_triplets = self.embedder.encode(triplet_sentences, convert_to_tensor=True)
        encoded_question = self.embedder.encode(prompt, convert_to_tensor=True)
        
        # Calculer les similarités
        cos_scores = util.pytorch_cos_sim(encoded_question, encoded_triplets)[0]
        top_k = min(max_triplets, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()
        
        # Sélectionner les triplets les plus pertinents
        selected_triplets = [all_triplets[idx] for idx in top_indices]
        triplet_scores = [cos_scores[idx].item() for idx in top_indices]
        
        # Formater les triplets en langage naturel
        triplet_text = "\n".join([f"- Le document indique que {self.format_triplet_natural(t)}." for t in selected_triplets])
        
        # Construire le prompt amélioré
        enhanced_prompt = f"""Vous êtes un assistant répondant uniquement sur la base des informations fournies dans le document.

Faits structurés extraits du document:
{triplet_text}

Maintenant, répondez à la question: {prompt}
"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'selected_triplets': selected_triplets,
            'triplet_scores': triplet_scores
        }
    
    def enhance_with_kag_rag(self, 
                            prompt: str, 
                            document_text: str, 
                            max_triplets: int = 5, 
                            top_k_chunks: int = 3, 
                            max_tokens: int = 300) -> Dict[str, Any]:
        """
        Améliore un prompt en combinant KAG et RAG.
        
        Args:
            prompt (str): Le prompt original.
            document_text (str): Le texte du document.
            max_triplets (int): Nombre maximum de triplets à inclure.
            top_k_chunks (int): Nombre de chunks à récupérer.
            max_tokens (int): Taille maximale des chunks.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant le prompt amélioré et les informations des approches KAG et RAG.
        """
        # Récupérer les améliorations individuelles
        kag_result = self.enhance_with_kag(prompt, document_text, max_triplets)
        rag_result = self.enhance_with_rag(prompt, document_text, top_k_chunks, max_tokens)
        
        # Extraire les triplets et les chunks
        triplet_text = "\n".join([f"- Le document indique que {self.format_triplet_natural(t)}." 
                                for t in kag_result.get('selected_triplets', [])])
        
        rag_context = "\n\n".join([f"\"{chunk.strip()}\"" 
                                  for chunk in rag_result.get('retrieved_chunks', [])])
        
        # Combiner les deux approches
        enhanced_prompt = f"""Vous êtes un assistant répondant uniquement sur la base des informations fournies dans le document.

Faits structurés extraits du document:
{triplet_text}

Preuves contextuelles du document:
{rag_context}

Maintenant, répondez à la question: {prompt}
"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'kag_info': kag_result,
            'rag_info': rag_result
        }
    
    def compare_enhancement_approaches(self, 
                                      prompt: str, 
                                      document_text: str) -> Dict[str, str]:
        """
        Compare les différentes approches d'amélioration de prompt.
        
        Args:
            prompt (str): Le prompt original.
            document_text (str): Le texte du document.
            
        Returns:
            Dict[str, str]: Dictionnaire des prompts {nom_approche: prompt_amélioré}.
        """
        # Récupérer les différentes versions
        vanilla_prompt = prompt
        rag_prompt = self.enhance_with_rag(prompt, document_text)['enhanced_prompt']
        kag_prompt = self.enhance_with_kag(prompt, document_text)['enhanced_prompt']
        kag_rag_prompt = self.enhance_with_kag_rag(prompt, document_text)['enhanced_prompt']
        
        return {
            'Vanilla': vanilla_prompt,
            'RAG': rag_prompt,
            'KAG': kag_prompt,
            'KAG+RAG': kag_rag_prompt
        } 