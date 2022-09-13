from ast import If
from readline import parse_and_bind
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import time


def load_data():
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def print_results(y_pred, y_test, model_obj, exec_time):
    print(f"""
    {"*" * 50}
    Modelo elegido: {type(model_obj)}
    Dataset: UCI ML Wine Data Set.

    Parametros usados: {model_obj.get_params()}
    Tiempo de entrenamiento: {exec_time} segundos.

    Reporte de clasificacion: 

    {classification_report(y_test, y_pred)}
    {"*" * 50}
    """)


def model_training(X, y):

    # Particionamieto de datos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    # Arreglo de parametros a probar.
    grid = {
        "n_neighbors": range(1, 30, 1),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "leaf_size": range(1, 50, 5)
    }

    # Estrategia de Cross Validation.
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # Modelo elegido. KNN.
    knn = KNeighborsClassifier()

    # Busqueda de mejores parametros posibles.
    grid_search = GridSearchCV(
                        estimator=knn, 
                        param_grid=grid, 
                        n_jobs=1, 
                        cv=cv, 
                        scoring="accuracy", 
                        error_score=0
    )

    # Definicion del modelo final con los parametros encontrados.
    start = time.time()
    results = grid_search.fit(X_train, y_train)
    model = knn.set_params(**results.best_params_)
    model.fit(X_train, y_train)
    exec_time = time.time() - start
    y_pred = model.predict(X_test)

    print_results(y_pred, y_test, model, exec_time)

    
if __name__ == "__main__" :
    X, y = load_data()
    model_training(X, y)

