import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
import numpy as np



PROJECT_NAME = "Prediction_survie_titanic"
RUN_NAME     = "Random_forest_v7"
MODEL_TYPE   = "classification"  # "regression" ou "classification"


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(PROJECT_NAME)


def tracker_mlflow(model, params, X_train, X_test, y_train, y_test):
    mlflow.end_run()

    with mlflow.start_run(run_name=RUN_NAME):

        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if MODEL_TYPE == "regression":
            r2   = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mse  = mean_squared_error(y_test, y_pred)

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse",     rmse)
            mlflow.log_metric("mse",      mse)

            print(f"R²   : {r2:.4f}")
            print(f"RMSE : {rmse:.2f}")

        elif MODEL_TYPE == "classification":
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)

            print(f"Accuracy : {acc:.4f}")
            print(classification_report(y_test, y_pred))

        mlflow.sklearn.log_model(model, "model")
        print(f"Experience '{RUN_NAME}' enregistree dans MLflow !")

    return model, y_pred