import PyPDF2
import os
from typing import Union, List, Optional


class PDFLoader:
    """
    Classe pour charger et extraire du texte à partir de fichiers PDF.
    """
    
    def __init__(self, pdf_path: str = None):
        """
        Initialise le PDFLoader avec un chemin de fichier optionnel.
        
        Args:
            pdf_path (str, optional): Chemin vers le fichier PDF.
        """
        self.pdf_path = pdf_path
        self.text = ""
        
    def load_pdf(self, pdf_path: Optional[str] = None) -> str:
        """
        Charge un fichier PDF et extrait son contenu textuel.
        
        Args:
            pdf_path (str, optional): Chemin vers le fichier PDF.
                                      Utilisé si aucun chemin n'a été fourni lors de l'initialisation.
        
        Returns:
            str: Le texte extrait du PDF.
            
        Raises:
            FileNotFoundError: Si le fichier PDF n'existe pas.
            ValueError: Si aucun chemin de fichier n'est fourni.
        """
        # Utiliser le chemin fourni en argument ou celui de l'initialisation
        path = pdf_path or self.pdf_path
        
        if not path:
            raise ValueError("Aucun chemin de fichier PDF fourni.")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier PDF '{path}' n'existe pas.")
        
        self.pdf_path = path
        self.text = ""
        
        with open(path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    self.text += page_text + "\n"
        
        return self.text
    
    def get_text(self) -> str:
        """
        Renvoie le texte extrait du dernier PDF chargé.
        
        Returns:
            str: Le texte extrait du PDF.
        """
        return self.text
    
    @staticmethod
    def extract_pages(pdf_path: str, page_numbers: List[int]) -> str:
        """
        Extrait le texte de pages spécifiques d'un PDF.
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF.
            page_numbers (List[int]): Liste des numéros de page à extraire (base 0).
            
        Returns:
            str: Texte extrait des pages spécifiées.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF '{pdf_path}' n'existe pas.")
        
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            max_page = len(pdf_reader.pages) - 1
            
            for page_num in page_numbers:
                if 0 <= page_num <= max_page:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"--- Page {page_num + 1} ---\n"
                        text += page_text + "\n\n"
        
        return text
    
    def get_metadata(self) -> dict:
        """
        Extrait les métadonnées du PDF chargé.
        
        Returns:
            dict: Métadonnées du PDF.
            
        Raises:
            ValueError: Si aucun PDF n'a été chargé.
        """
        if not self.pdf_path or not self.text:
            raise ValueError("Aucun PDF n'a été chargé. Utilisez d'abord load_pdf().")
            
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return pdf_reader.metadata
    
    def get_page_count(self) -> int:
        """
        Renvoie le nombre de pages du PDF chargé.
        
        Returns:
            int: Nombre de pages.
            
        Raises:
            ValueError: Si aucun PDF n'a été chargé.
        """
        if not self.pdf_path:
            raise ValueError("Aucun PDF n'a été chargé. Utilisez d'abord load_pdf().")
            
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return len(pdf_reader.pages) 