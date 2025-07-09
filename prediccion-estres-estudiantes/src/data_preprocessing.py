import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Ejemplo: Llenar nulos
    df.fillna(df.mean(), inplace=True)

    # Codificar variables categóricas
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    # Separar features y target
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
