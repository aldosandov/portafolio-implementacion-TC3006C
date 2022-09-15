from model import model_training, load_data
import pandas as pd


def make_prediction():
    X_train, X_test, y_train, y_test = load_data() 
    model = model_training(X_train, y_train)

    # La prediccion se hace con el 10% de datos reservados para pruebas. 
    # Entrenamiento y validacion con el 90% restante.
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)

    return y_pred, y_test, score


if __name__ == '__main__':
    y_pred, y_test, score = make_prediction()
    print(f'Tamaño de set de pruebas: {y_pred.shape[0]} registros')
    print("\nResultados: ")
    print(y_pred)
    print("Respectivamente para cada registro.")
    print(f'Accuracy de la prediccion: {score*100}%')


