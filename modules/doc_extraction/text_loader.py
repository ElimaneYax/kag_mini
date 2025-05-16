import os
from typing import Optional, List, Tuple


class TextLoader:
    """
    Classe pour charger et traiter des fichiers texte.
    """
    
    def __init__(self, file_path: str = None):
        """
        Initialise le TextLoader avec un chemin de fichier optionnel.
        
        Args:
            file_path (str, optional): Chemin vers le fichier texte.
        """
        self.file_path = file_path
        self.text = ""
        self.lines = []
        
    def load_text(self, file_path: Optional[str] = None, encoding: str = 'utf-8') -> str:
        """
        Charge un fichier texte.
        
        Args:
            file_path (str, optional): Chemin vers le fichier texte.
                                       Utilisé si aucun chemin n'a été fourni lors de l'initialisation.
            encoding (str): L'encodage du fichier texte.
        
        Returns:
            str: Le contenu textuel du fichier.
            
        Raises:
            FileNotFoundError: Si le fichier texte n'existe pas.
            ValueError: Si aucun chemin de fichier n'est fourni.
        """
        # Utiliser le chemin fourni en argument ou celui de l'initialisation
        path = file_path or self.file_path
        
        if not path:
            raise ValueError("Aucun chemin de fichier texte fourni.")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier texte '{path}' n'existe pas.")
        
        self.file_path = path
        
        with open(path, 'r', encoding=encoding) as file:
            self.text = file.read()
            self.lines = self.text.splitlines()
        
        return self.text
    
    def get_text(self) -> str:
        """
        Renvoie le texte du dernier fichier chargé.
        
        Returns:
            str: Le contenu textuel du fichier.
        """
        return self.text
    
    def get_lines(self) -> List[str]:
        """
        Renvoie le texte sous forme de liste de lignes.
        
        Returns:
            List[str]: Liste des lignes du fichier texte.
        """
        return self.lines
    
    def save_text(self, text: str, file_path: Optional[str] = None, encoding: str = 'utf-8') -> None:
        """
        Enregistre un texte dans un fichier.
        
        Args:
            text (str): Texte à enregistrer.
            file_path (str, optional): Chemin du fichier de sortie.
                                       Si non fourni, utilise le chemin du fichier d'entrée.
            encoding (str): Encodage à utiliser pour l'enregistrement.
            
        Raises:
            ValueError: Si aucun chemin de fichier n'est fourni et qu'aucun n'est défini.
        """
        output_path = file_path or self.file_path
        
        if not output_path:
            raise ValueError("Aucun chemin de fichier de sortie fourni.")
        
        with open(output_path, 'w', encoding=encoding) as file:
            file.write(text)
    
    @staticmethod
    def load_multiple_texts(file_paths: List[str], encoding: str = 'utf-8') -> List[Tuple[str, str]]:
        """
        Charge plusieurs fichiers texte.
        
        Args:
            file_paths (List[str]): Liste des chemins de fichiers à charger.
            encoding (str): Encodage des fichiers.
            
        Returns:
            List[Tuple[str, str]]: Liste de tuples (nom_fichier, contenu).
        """
        results = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    results.append((os.path.basename(file_path), content))
                except Exception as e:
                    print(f"Erreur lors du chargement du fichier {file_path}: {e}")
        
        return results
    
    @staticmethod
    def read_text_from_string(text_content: str) -> List[str]:
        """
        Transforme une chaîne de texte en liste de lignes.
        
        Args:
            text_content (str): Contenu textuel.
            
        Returns:
            List[str]: Liste des lignes du texte.
        """
        return text_content.splitlines() 