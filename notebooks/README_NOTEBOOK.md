# Projet de Détection de Spam (SMS & Email)

Ce projet est réalisé dans le cadre de l'atelier de NLP (Traitement du Langage Naturel). L'objectif est de développer un classifieur capable de distinguer les messages légitimes (*ham*) des messages indésirables (*spam*) en utilisant des techniques d'apprentissage automatique supervisé.

Nous comparons l'efficacité de la détection sur deux canaux différents (SMS et Emails) et explorons les limites du transfert d'apprentissage entre ces deux domaines.

## 🛠 Méthodologie du Pipeline

Le projet suit un pipeline de Data Science rigoureux :

1.  **Préparation des données :**
    * Chargement des fichiers CSV (SMS et Email).
    * Nettoyage : Suppression des doublons.
    * **Stratégie de Split :** Utilisation de `stratified split` pour maintenir la distribution des classes.
2.  **Gestion du déséquilibre (Balancing) :**
    * Les datasets d'origine sont déséquilibrés (majorité de messages légitimes).
    * Solution appliquée : **Oversampling** (sur-échantillonnage) de la classe minoritaire (Spam) pour atteindre un ratio 50/50 dans le jeu d'entraînement.
3.  **Traitement du texte (NLP) :**
    * Tokenisation personnalisée (Regex).
    * Suppression des *stop-words* anglais.
    * Vectorisation via **CountVectorizer** (Bag-of-Words) limité aux 5000 mots les plus fréquents.
4.  **Modélisation :**
    * Algorithme : **Régression Logistique**.
    * Métriques d'évaluation : Accuracy, Précision, Rappel (Recall).

## 📊 Résultats Expérimentaux

Voici les performances obtenues selon les différents scénarios d'entraînement :

| Scénario | Accuracy | Précision | Rappel |
| :--- | :---: | :---: | :---: |
| **Email Seul** (Baseline) | **0.9811** | 0.9680 | **0.9680** |
| **SMS Seul** (Baseline) | 0.9749 | **0.9906** | 0.8077 |
| **Combiné** (Email + SMS) | 0.9645 | 0.9480 | 0.8840 |
| **Transfert** (Train SMS -> Test Email) | 0.7385 | 0.5444 | 0.7095 |

## 🧠 Analyse et Conclusions

### 1. Comparaison SMS vs Email
Le modèle **Email Seul** est le plus performant (98% d'accuracy et 96% de rappel). Les emails contiennent généralement plus de texte et de métadonnées que les SMS, offrant ainsi plus de "signaux" au modèle pour identifier un spam.
Le modèle **SMS Seul** a une excellente précision (99%), ce qui signifie qu'il fait très peu de fausses alertes, mais son rappel est plus faible (80%), indiquant qu'il rate environ 20% des spams (probablement à cause de la brièveté des messages et de l'argot).

### 2. Échec du Transfert de Domaine
Le scénario de **Transfert** (apprendre sur SMS pour prédire sur Email) montre une chute drastique des performances (Accuracy de 73%).
* La **précision chute à 54%**, ce qui est à peine mieux que le hasard.
* **Conclusion :** Le vocabulaire utilisé dans les spams SMS (ex: "URGENT", "FREE", numéros courts) est très différent de celui des spams Email (ex: Phishing, HTML, narration longue). Un modèle ne peut pas généraliser efficacement d'un domaine à l'autre sans réentraînement.

### 3. Approche Combinée
L'entraînement sur les données fusionnées (**Combiné**) offre un bon compromis (96.4% d'accuracy). Bien qu'il soit légèrement moins performant que le spécialiste "Email seul", il est beaucoup plus robuste et généraliste. C'est la stratégie recommandée pour un système de production devant gérer plusieurs canaux.

## 👤 Auteurs
* Heroguer Marin