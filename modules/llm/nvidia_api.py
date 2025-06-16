from openai import OpenAI
from typing import Dict, List, Any, Optional, Union
import os
import json


class NvidiaLLMClient:
    """
    Client pour interagir avec les modèles LLM via l'API NVIDIA.
    Utilise l'interface compatible OpenAI.
    """
    
    # Clé API par défaut (à remplacer par la vôtre ou à définir via variable d'environnement)
    DEFAULT_API_KEY = "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"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client NVIDIA LLM.
        
        Args:
            api_key (str, optional): Clé API NVIDIA. Si non fournie, utilise la valeur par défaut
                                    ou la variable d'environnement NVIDIA_API_KEY.
        """
        # Ordre de priorité pour la clé API: paramètre > variable d'environnement > défaut
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", self.DEFAULT_API_KEY)
        
        # Initialiser le client OpenAI avec l'URL de base NVIDIA
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        
        # Nouveau modèle par défaut
        self.default_model = "qwen/qwen3-235b-a22b"
    
    def query(self, 
              prompt: str, 
              model: Optional[str] = None, 
              temperature: float = 0.2, 
              max_tokens: int = 2000,
              top_p: float = 0.7) -> str:
        """
        Envoie une requête au modèle LLM Qwen-3 235B (NVIDIA API).
        Retourne uniquement la réponse finale (pas de reasoning/thinking).
        """
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        # On retourne uniquement la réponse finale
        # Ajout d'une vérification si content est None
        if response.choices and response.choices[0].message.content is not None:
            return response.choices[0].message.content
        else:
            print("Avertissement: La réponse du LLM est vide ou None.")
            return None # Retourne explicitement None si la réponse est vide
    
    def generate_responses(self, 
                          prompts: List[str], 
                          model: Optional[str] = None,
                          temperature: float = 0.2, 
                          max_tokens: int = 8192) -> List[str]:
        """
        Génère des réponses pour plusieurs prompts.
        
        Args:
            prompts (List[str]): Liste des prompts à envoyer.
            model (str, optional): Le modèle à utiliser.
            temperature (float): Contrôle de la créativité.
            max_tokens (int): Nombre maximum de tokens dans chaque réponse.
            
        Returns:
            List[str]: Liste des réponses.
        """
        responses = []
        
        for prompt in prompts:
            response = self.query(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response)
        
        return responses
    
    def compare_responses(self, 
                         prompt: str, 
                         models: List[str], 
                         temperature: float = 0.2,
                         max_tokens: int = 8192) -> Dict[str, str]:
        """
        Compare les réponses de différents modèles pour un même prompt.
        
        Args:
            prompt (str): Le prompt à envoyer.
            models (List[str]): Liste des modèles à comparer.
            temperature (float): Contrôle de la créativité.
            max_tokens (int): Nombre maximum de tokens dans chaque réponse.
            
        Returns:
            Dict[str, str]: Dictionnaire {nom_modèle: réponse}.
        """
        results = {}
        
        for model in models:
            try:
                response = self.query(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                results[model] = response
            except Exception as e:
                results[model] = f"Erreur: {str(e)}"
        
        return results
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des modèles disponibles via l'API NVIDIA.
        
        Returns:
            List[Dict[str, Any]]: Liste des modèles disponibles avec leurs détails.
        """
        try:
            # Pour les besoins de l'API Nvidia, utiliser une approche spécifique
            # Cette méthode peut varier selon l'API exacte
            response = self.client.models.list()
            return response.data
        except Exception as e:
            print(f"Erreur lors de la récupération des modèles: {e}")
            # Liste par défaut des modèles connus
            return [
                {"id": "qwen/qwen3-235b-a22b", "name": "Qwen 3 235B"},
                {"id": "meta/llama3-8b-instruct", "name": "Llama 3 8B Instruct"},
                {"id": "meta/llama3-70b-instruct", "name": "Llama 3 70B Instruct"}
            ]
    
    def save_responses_to_file(self, responses: Dict[str, str], filename: str) -> None:
        """
        Enregistre les réponses dans un fichier JSON.
        
        Args:
            responses (Dict[str, str]): Dictionnaire des réponses.
            filename (str): Nom du fichier de sortie.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
    
    def set_default_model(self, model: str) -> None:
        """
        Définit le modèle par défaut.
        
        Args:
            model (str): Identifiant du modèle.
        """
        self.default_model = model 