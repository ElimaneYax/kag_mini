# KAG System - Knowledge-Augmented Generation

Un système d'extraction de connaissances et de génération augmentée par graphes pour traiter des documents, extraire des triplets de connaissances, et fournir des réponses augmentées par RAG et KAG.

## Architecture

Le système est organisé en modules :

- **doc_extraction**: Extraction de texte à partir de documents.
  - `pdf_loader.py`: Charge et extrait du texte de fichiers PDF.
  - `text_loader.py`: Charge et traite des fichiers texte.
  - `triplet_extractor.py`: Extrait des triplets (sujet, verbe, objet) à partir de textes.

- **semantic_processing**: Traitement sémantique du texte.
  - `semantic_chunker.py`: Découpe des textes en chunks sémantiquement cohérents.
  - `prompt_enhancer.py`: Améliore les prompts en utilisant RAG et KAG.

- **graph_processing**: Gestion des graphes de connaissances.
  - `knowledge_graph.py`: Crée et manipule des graphes de connaissances.
  - `neo4j_connector.py`: Connecte le système à une base de données Neo4j.

- **llm**: Interaction avec les modèles de langage.
  - `nvidia_api.py`: Client pour interagir avec l'API NVIDIA pour accéder à des LLMs.

- **main.py**: Intègre tous les modules et fournit une interface en ligne de commande.

## Installation

1. Cloner ce dépôt :
```bash
git clone [repo-url]
cd kag_mini
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Télécharger le modèle spaCy :
```bash
python -m spacy download en_core_web_sm
```

4. (Optionnel) Configurer Neo4j :
   - Installer [Neo4j Desktop](https://neo4j.com/download/) ou utiliser Neo4j Cloud.
   - Créer une base de données avec un mot de passe.
   - Mettre à jour les informations de connexion dans les commandes.

## Utilisation

### Traiter un document

Traiter un document PDF ou texte pour extraire des triplets et construire un graphe de connaissances :

```bash
python main.py --document chemin/vers/document.pdf
```

### Poser une question

Poser une question au système en utilisant différentes méthodes d'amélioration de prompt :

```bash
# Utiliser KAG+RAG (par défaut)
python main.py --question "Quelle est la principale conclusion du document?"

# Utiliser seulement RAG
python main.py --question "Quels sont les résultats principaux?" --method rag

# Utiliser seulement KAG
python main.py --question "Comment les auteurs ont-ils procédé?" --method kag

# Utiliser le prompt vanilla (sans amélioration)
python main.py --question "Qui sont les auteurs?" --method vanilla
```

### Visualiser le graphe

Générer une visualisation du graphe de connaissances :

```bash
python main.py --visualize
```

### Configurer Neo4j

Spécifier les paramètres de connexion à Neo4j :

```bash
python main.py --document chemin/vers/document.pdf --neo4j_uri bolt://localhost:7687 --neo4j_user neo4j --neo4j_password votremotdepasse
```

Désactiver l'utilisation de Neo4j :

```bash
python main.py --document chemin/vers/document.pdf --no_neo4j
```

### Effacer le graphe

Effacer le graphe de connaissances de la mémoire et de Neo4j :

```bash
python main.py --clear
```

## Configuration

Vous pouvez configurer les paramètres de connexion Neo4j via des variables d'environnement :

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=votremotdepasse
```

## Exemple d'utilisation dans un script

```python
from main import KAGSystem

# Initialiser le système
system = KAGSystem()

# Traiter un document
system.process_document("mon_document.pdf")

# Poser une question avec KAG+RAG
result = system.answer_question(
    "Quelle est la principale méthode utilisée dans ce document?", 
    enhancement_method="kag_rag"
)

# Afficher la réponse
print(result["answer"])

# Visualiser le graphe
system.visualize_graph()
```

## Amélioration des prompts

Le système offre trois approches pour améliorer les prompts :

1. **RAG (Retrieval-Augmented Generation)** : Récupère des chunks sémantiquement pertinents du document en fonction de la question.

2. **KAG (Knowledge-Augmented Generation)** : Sélectionne des triplets (faits structurés) pertinents pour la question.

3. **KAG+RAG** : Combine les deux approches pour une meilleure contextualisation et structuration de l'information.

## Extensibilité

Le système est conçu pour être facilement extensible :

- Ajoutez de nouveaux extracteurs de documents dans le module `doc_extraction`.
- Implémentez de nouvelles techniques d'extraction dans `triplet_extractor.py`.
- Ajoutez d'autres backends de graphes dans le module `graph_processing`.
- Intégrez d'autres APIs LLM dans le module `llm`.

## License



---