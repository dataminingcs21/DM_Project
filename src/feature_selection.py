import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def compute_rf_feature_importances(X_train, y_train, n_estimators=50, random_state=0, save_fig_path=None):
    """
    Entraîne un modèle Random Forest pour évaluer l'importance relative de chaque caractéristique.
    
    Paramètres :
        X_train (pd.DataFrame) : Caractéristiques d'entraînement standardisées.
        y_train (pd.Series/np.ndarray) : Cibles d'entraînement.
        n_estimators (int) : Nombre d'arbres dans la forêt (par défaut: 50).
        random_state (int) : Graine aléatoire pour la reproductibilité.
        save_fig_path (str, optionnel) : Chemin pour sauvegarder le graphique d'importance.
        
    Retourne :
        pd.DataFrame : Tableau classé des caractéristiques et de leurs scores d'importance.
    """
    print("--- Évaluation de l'importance des variables (Random Forest) ---")
    rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    rfc.fit(X_train, y_train)
    
    score = np.round(rfc.feature_importances_, 3)
    importances = pd.DataFrame({'feature': X_train.columns, 'importance': score})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    
    # Génération et sauvegarde automatique du graphique si demandé
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.figure(figsize=(12, 5))
        # Afficher les 25 caractéristiques les plus discriminantes pour une meilleure lisibilité
        top_n = min(25, len(importances))
        importances.head(top_n).plot(kind='bar', legend=False)
        plt.title(f"Top {top_n} des variables les plus importantes (Random Forest)")
        plt.ylabel("Importance normalisée")
        plt.xlabel("Variables réseau")
        plt.xticks(rotation=75, ha='right')
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300)
        plt.close()
        print(f"Graphique sauvegardé dans : {save_fig_path}")
        
    return importances

def apply_pca(X_train, X_test, n_components=15, save_variance_fig_path=None):
    """
    Applique la réduction de dimension par Analyse en Composantes Principales (PCA).
    Le modèle est ajusté UNIQUEMENT sur X_train pour éviter le data leakage.
    
    Paramètres :
        X_train (pd.DataFrame) : Données d'entraînement normalisées.
        X_test (pd.DataFrame) : Données de test normalisées.
        n_components (int) : Nombre de composantes à retenir (par défaut: 15).
        save_variance_fig_path (str, optionnel) : Chemin pour enregistrer la variance expliquée.
        
    Retourne :
        train_df (pd.DataFrame) : X_train projeté dans l'espace PCA.
        test_df (pd.DataFrame) : X_test projeté dans l'espace PCA.
        pca (PCA) : L'objet PCA ajusté.
    """
    print(f"--- Application de la PCA ({n_components} composantes) ---")
    pca = PCA(n_components=n_components)
    
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)
    
    pc_cols = [f'PC{i}' for i in range(1, n_components + 1)]
    train_df = pd.DataFrame(data=train_pca, columns=pc_cols)
    test_df = pd.DataFrame(data=test_pca, columns=pc_cols)
    
    print(f"Dimensions avant PCA : {X_train.shape}")
    print(f"Dimensions après PCA : {train_df.shape}")
    
    # Graphique de la variance expliquée
    if save_variance_fig_path:
        os.makedirs(os.path.dirname(save_variance_fig_path), exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100)
        plt.xlabel("Composantes principales")
        plt.ylabel("Variance expliquée (%)")
        plt.title("Variance expliquée par les composantes PCA")
        plt.tight_layout()
        plt.savefig(save_variance_fig_path, dpi=300)
        plt.close()
        print(f"Graphique de variance PCA sauvegardé dans : {save_variance_fig_path}")
        
    return train_df, test_df, pca

def apply_lda(X_train, y_train, X_test, n_components=1):
    """
    Applique l'Analyse Discriminante Linéaire (LDA) pour projeter les données
    de manière supervisée vers l'axe maximisant la séparation binaire (Attaque vs Normal).
    
    Paramètres :
        X_train (pd.DataFrame) : Données d'entraînement (idéalement post-PCA).
        y_train (pd.Series/np.ndarray) : Labels cibles binaires.
        X_test (pd.DataFrame) : Données de test.
        n_components (int) : Nombre de composantes discriminantes (1 pour un problème binaire).
        
    Retourne :
        train_lda (np.ndarray) : X_train projeté.
        test_lda (np.ndarray) : X_test projeté.
        lda (LDA) : L'objet LDA ajusté.
    """
    print(f"--- Application de la LDA ({n_components} dimension discriminante) ---")
    lda = LDA(n_components=n_components)
    train_lda = lda.fit_transform(X_train, y_train)
    test_lda = lda.transform(X_test)
    
    print(f"Dimensions finales projetées (LDA) : {train_lda.shape}")
    return train_lda, test_lda, lda