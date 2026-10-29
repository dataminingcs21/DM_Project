import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_merge_datasets(dataset_dir='./dataset/'):
    """
    Charge les 8 fichiers CSV du CIC-IDS-2017 et les fusionne en un seul DataFrame.
    """
    files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    ]
    
    dfs = []
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            print(f"Chargement : {file}")
            dfs.append(pd.read_csv(file_path))
        else:
            print(f"Attention : Fichier non trouvé : {file_path}")
            
    if not dfs:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {dataset_dir}")
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def clean_data(df, zero_threshold=99.0):
    """
    Remplace les infinis par NaN, supprime les colonnes constantes/quasi-nulles
    et retire les lignes contenant des valeurs manquantes.
    """
    # Remplacer les infinis générés par les divisions par 0 dans le trafic réseau
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    
    # Supprimer les colonnes avec >= 99% de zéros (colonnes sans variance)
    cols_to_drop = []
    total_rows = df_clean.shape[0]
    for col in df_clean.columns:
        if (df_clean[col] == 0).sum() / total_rows * 100 >= zero_threshold:
            cols_to_drop.append(col)
            
    df_clean = df_clean.drop(columns=cols_to_drop)
    # Suppression des lignes contenant NaN
    df_clean = df_clean.dropna().copy()
    return df_clean

def encode_labels(df, binary=True):
    """
    Encode les libellés texte en valeurs numériques (binaire ou multiclasse).
    """
    def norm(s):
        return ' '.join(re.sub(r'[^A-Za-z\- ]', ' ', str(s)).split())

    label_names = [
        'BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
        'DoS Hulk', 'DoS GoldenEye', 'Heartbleed', 'Web Attack Brute Force',
        'Web Attack XSS', 'Web Attack Sql Injection', 'Infiltration', 'Bot', 'DDoS', 'PortScan'
    ]
    label_map = {norm(name): idx for idx, name in enumerate(label_names)}
    
    encoded_label = df['Label'].map(lambda s: label_map.get(norm(s)))
    df_encoded = df.copy()
    df_encoded['Label'] = encoded_label.astype(int)
    
    X = df_encoded.drop(columns=['Label'])
    y = df_encoded['Label']
    
    if binary:
        # 1 = BENIGN (trafic normal), 0 = Attaque
        y = (y == 0).astype(int)
        
    return X, y

def split_and_scale(X, y, test_size=0.3, random_state=1):
    """
    Sépare les données en Train/Test et applique un StandardScaler.
    Le scaler est fitté UNIQUEMENT sur le train pour éviter le data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include='number').columns
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values