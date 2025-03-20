import logging
import os
import pickle
import pandas as pd
import numpy as np
import fire
from typing import Optional, Dict, Any, Tuple, List, Union
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import time
from pathlib import Path

try:
    from clearml import Task

    clearml_available = True
except ImportError:
    clearml_available = False


class My_Classifier_Model:
    """
    Classifier model for the Spaceship Anomaly challenge.

    This class provides methods to train an XGBoost model and make predictions.
    It includes extensive feature engineering tailored to the Spaceship Titanic dataset.
    """

    def _init_clearml(self, task_name="Spaceship Titanic Training"):
        """Initialize ClearML tracking if available."""
        if not self.use_clearml or not clearml_available:
            return None

        try:
            task = Task.init(
                project_name="Spaceship Anomaly",
                task_name=task_name,
                auto_connect_frameworks=True
            )

            task.connect_configuration({
                "model_type": "XGBoost",
                "random_state": self.random_state,
                "n_trials": self.n_trials,
                "model_version": "1.0"
            })

            self.logger.info("ClearML task initialized successfully")
            return task
        except Exception as e:
            self.logger.warning(f"ClearML initialization failed: {str(e)}. Continuing without experiment tracking.")
            self.use_clearml = False
            return None

    def __init__(self, random_state: int = 42, n_trials: int = 50, use_clearml: bool = True):
        """
        Initialize the model and set up logging.

        Args:
            random_state: Random seed for reproducibility
            n_trials: Number of Optuna hyperparameter optimization trials
            use_clearml: Whether to use ClearML for experiment tracking
        """
        self._setup_logging()
        self.model = None
        self.best_threshold = 0.5
        self.random_state = random_state
        self.n_trials = n_trials
        self.use_clearml = use_clearml and clearml_available
        self.task = None

        os.makedirs('./data', exist_ok=True)
        os.makedirs('./model', exist_ok=True)

        self.logger.info(
            f"Initialized model with random_state={random_state}, n_trials={n_trials}, use_clearml={self.use_clearml}")

    def _setup_logging(self):
        """Set up logging to file and console as a singleton."""
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        log_path = os.path.join(data_dir, 'log_file.log')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            self.logger.info(f"Logging initialized at {log_path}")

    def _prepare_data(self, data: pd.DataFrame, is_train: bool = True) -> Union[
        Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare and engineer features from the raw dataset.

        Args:
            data: Raw dataset
            is_train: Whether this is training data (with target variable)

        Returns:
            Processed dataset ready for model training or prediction
        """
        self.logger.info(f"Preparing {'training' if is_train else 'test'} data")

        if is_train:
            y = data['Transported'].copy()
            if y.dtype == bool:
                y = y.astype(int)
                self.logger.info("Converted boolean target to integer (0/1)")
            X = data.drop(['Transported'], axis=1)
        else:
            X = data.copy()
            y = None

        if 'PassengerId' in X.columns:
            passenger_ids = X['PassengerId'].copy()
        else:
            passenger_ids = None

        self.logger.info("Performing feature engineering")

        if 'PassengerId' in X.columns:
            X['PassengerGroup'] = X['PassengerId'].str.split('_').str[0].astype(int)
            X['PassengerNumber'] = X['PassengerId'].str.split('_').str[1].astype(int)

        if 'Cabin' in X.columns:
            X['HasCabin'] = X['Cabin'].notna().astype(int)

            X['Deck'] = 'Z'
            X['CabinNum'] = -1
            X['Side'] = 'Z'

            cabin_mask = X['Cabin'].notna()
            if cabin_mask.any():
                cabin_parts = X.loc[cabin_mask, 'Cabin'].str.split('/', expand=True)
                if cabin_parts.shape[1] >= 3:
                    X.loc[cabin_mask, 'Deck'] = cabin_parts[0]
                    X.loc[cabin_mask, 'CabinNum'] = pd.to_numeric(cabin_parts[1], errors='coerce')
                    X.loc[cabin_mask, 'Side'] = cabin_parts[2]

            X['CabinPosition'] = X['CabinNum'] % 2
            X['Deck_is_ABC'] = X['Deck'].isin(['A', 'B', 'C']).astype(int)
            X['Deck_is_FG'] = X['Deck'].isin(['F', 'G']).astype(int)

        expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        available_expenses = [col for col in expense_columns if col in X.columns]

        for col in available_expenses:
            X[col] = X[col].fillna(0)

        if available_expenses:
            X['TotalExpense'] = X[available_expenses].sum(axis=1)

            for col in available_expenses:
                X[f'{col}_Ratio'] = X.apply(
                    lambda x: x[col] / x['TotalExpense'] if x['TotalExpense'] > 0 else 0,
                    axis=1
                )

            X['HasExpense'] = (X['TotalExpense'] > 0).astype(int)
            X['ExpenseCount'] = (X[available_expenses] > 0).sum(axis=1)

            X['ExpensePattern'] = X.apply(
                lambda x: ''.join(['1' if x[col] > 0 else '0' for col in available_expenses]),
                axis=1
            )

            X['LogExpense'] = np.log1p(X['TotalExpense'])
            X['SqrtExpense'] = np.sqrt(X['TotalExpense'])

        if 'PassengerGroup' in X.columns:
            group_sizes = X.groupby('PassengerGroup').size()
            X['GroupSize'] = X['PassengerGroup'].map(group_sizes)

            if 'TotalExpense' in X.columns:
                group_expenses = X.groupby('PassengerGroup')['TotalExpense'].agg(['sum', 'mean', 'std']).fillna(0)
                X['GroupTotalExpense'] = X['PassengerGroup'].map(group_expenses['sum'])
                X['GroupMeanExpense'] = X['PassengerGroup'].map(group_expenses['mean'])

                X['ExpenseVsGroup'] = X.apply(
                    lambda x: x['TotalExpense'] / x['GroupMeanExpense'] if x['GroupMeanExpense'] > 0 else 0,
                    axis=1
                )

        if 'Name' in X.columns:
            X['LastName'] = X['Name'].str.split(' ').str[0]

            family_sizes = X.groupby('LastName').size()
            X['FamilySize'] = X['LastName'].map(family_sizes)

        if 'GroupSize' in X.columns:
            X['IsAlone'] = (X['GroupSize'] == 1).astype(int)

        if 'FamilySize' in X.columns:
            X['HasFamily'] = (X['FamilySize'] > 1).astype(int)

        if 'HomePlanet' in X.columns:
            X['HomePlanet'] = X['HomePlanet'].fillna(X['HomePlanet'].mode()[0])

        if 'CryoSleep' in X.columns:
            X['CryoSleep'] = X['CryoSleep'].fillna(False)

        if 'Destination' in X.columns:
            X['Destination'] = X['Destination'].fillna(X['Destination'].mode()[0])

        if 'VIP' in X.columns:
            X['VIP'] = X['VIP'].fillna(False)

        if 'Age' in X.columns:
            X['Age'] = X['Age'].fillna(X['Age'].median())
            X['AgeBand'] = pd.qcut(X['Age'], 5, labels=False)
            X['Age_squared'] = X['Age'] ** 2
            X['IsChild'] = (X['Age'] < 13).astype(int)
            X['IsAdult'] = ((X['Age'] >= 13) & (X['Age'] < 65)).astype(int)
            X['IsSenior'] = (X['Age'] >= 65).astype(int)

            if 'HomePlanet' in X.columns:
                planets = sorted(X['HomePlanet'].unique())
                for planet in planets:
                    X[f'Age_from_{planet}'] = X['Age'] * (X['HomePlanet'] == planet)

        if all(col in X.columns for col in ['CryoSleep', 'VIP']):
            X['CryoSleep_VIP'] = (X['CryoSleep'] & X['VIP']).astype(int)

        if all(col in X.columns for col in ['CryoSleep', 'TotalExpense']):
            X['CryoSleep_HasExpense'] = (X['CryoSleep'] & (X['TotalExpense'] > 0)).astype(int)
            X['Unusual_Expense'] = ((X['CryoSleep'] & (X['TotalExpense'] > 0)) |
                                    (~X['CryoSleep'] & (X['TotalExpense'] == 0))).astype(int)

        if all(col in X.columns for col in ['HomePlanet', 'Destination']):
            X['Route'] = X['HomePlanet'] + '_to_' + X['Destination']

            route_counts = X['Route'].value_counts()
            X['RoutePopularity'] = X['Route'].map(route_counts)

        if all(col in X.columns for col in ['Deck', 'Side']):
            X['DeckSide'] = X['Deck'] + X['Side']

        if all(col in X.columns for col in ['VIP', 'TotalExpense']):
            X['Expense_by_VIP'] = X['TotalExpense'] * X['VIP'].astype(int)

        if all(col in X.columns for col in ['Age', 'TotalExpense']):
            X['ExpenseRatio_by_Age'] = X['TotalExpense'] / (X['Age'] + 1)

        self.logger.info("Encoding categorical features")

        categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckSide', 'Route']
        categorical_cols = [col for col in categorical_cols if col in X.columns]

        for col in categorical_cols:
            top_categories = sorted(X[col].value_counts().head(5).index.tolist())
            for category in top_categories:
                X[f'{col}_{category}'] = (X[col] == category).astype(int)

        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le.fit(X[col].fillna('Unknown').astype(str))
                X[col + '_encoded'] = le.transform(X[col].fillna('Unknown').astype(str))

                if col + '_encoded' in X.columns and col != 'LastName':
                    X = X.drop(columns=[col])

        drop_cols = ['Name', 'Cabin', 'PassengerId']
        drop_cols = [col for col in drop_cols if col in X.columns]
        X = X.drop(columns=drop_cols)

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes

        if is_train:
            self.feature_names = list(X.columns)

            feature_order_dir = os.path.join(os.getcwd(), 'model')
            os.makedirs(feature_order_dir, exist_ok=True)
            feature_order_path = os.path.join(feature_order_dir, 'feature_order.pkl')

            with open(feature_order_path, 'wb') as f:
                pickle.dump(self.feature_names, f)

            self.logger.info(f"Saved {len(self.feature_names)} feature names for future prediction")
        else:
            try:
                if hasattr(self, 'feature_names') and self.feature_names:
                    feature_order = self.feature_names
                else:
                    model_path = os.path.join(os.getcwd(), 'model', 'model.pkl')
                    feature_order_path = os.path.join(os.getcwd(), 'model', 'feature_order.pkl')

                    if os.path.exists(feature_order_path):
                        with open(feature_order_path, 'rb') as f:
                            feature_order = pickle.load(f)
                    elif os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                            if 'features' in model_data:
                                feature_order = model_data['features']
                            else:
                                self.logger.warning("Model doesn't contain feature names")
                                feature_order = None
                    else:
                        self.logger.warning("Couldn't find feature order information")
                        feature_order = None

                if feature_order:
                    self.logger.info(f"Adjusting features to match training data ({len(feature_order)} features)")

                    missing_cols = set(feature_order) - set(X.columns)
                    for col in missing_cols:
                        X[col] = 0
                        self.logger.warning(f"Added missing column: {col}")

                    extra_cols = set(X.columns) - set(feature_order)
                    if extra_cols:
                        self.logger.warning(f"Removing extra columns not in training: {extra_cols}")
                        X = X.drop(columns=list(extra_cols))

                    X = X[feature_order]
            except Exception as e:
                self.logger.error(f"Error adjusting feature order: {str(e)}")

        self.logger.info(f"Data preparation complete. Shape: {X.shape}")

        if is_train:
            return X, y
        else:
            return X, passenger_ids

    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Best hyperparameters
        """
        self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': trial.suggest_float('eta', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'tree_method': 'hist',
            }

            n_estimators = trial.suggest_int('n_estimators', 300, 1000)

            cv_scores = []

            for train_idx, val_idx in cv.split(X_train_opt, y_train_opt):
                X_fold_train, X_fold_val = X_train_opt.iloc[train_idx], X_train_opt.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_opt.iloc[train_idx], y_train_opt.iloc[val_idx]

                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=n_estimators,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                fold_preds = model.predict(dval)

                thresholds = np.linspace(0.3, 0.7, 21)
                scores = [accuracy_score(y_fold_val, (fold_preds > t).astype(int)) for t in thresholds]
                best_score = max(scores)
                cv_scores.append(best_score)

                if self.use_clearml and self.task is not None:
                    self.task.get_logger().report_scalar(
                        "CV Fold Accuracy",
                        f"Fold {len(cv_scores)}",
                        best_score,
                        iteration=trial.number
                    )

            mean_score = np.mean(cv_scores)

            if self.use_clearml and self.task is not None:
                self.task.get_logger().report_scalar(
                    "Mean CV Accuracy",
                    "Accuracy",
                    mean_score,
                    iteration=trial.number
                )

                numeric_params = {
                    'eta': params['eta'],
                    'max_depth': params['max_depth'],
                    'min_child_weight': params['min_child_weight'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree'],
                    'gamma': params['gamma'],
                    'reg_alpha': params['reg_alpha'],
                    'reg_lambda': params['reg_lambda'],
                    'scale_pos_weight': params['scale_pos_weight'],
                    'max_delta_step': params['max_delta_step'],
                    'n_estimators': n_estimators
                }

                for param_name, param_value in numeric_params.items():
                    self.task.get_logger().report_scalar(
                        "Parameters",
                        param_name,
                        param_value,
                        iteration=trial.number
                    )

                string_params = {
                    'objective': params['objective'],
                    'eval_metric': params['eval_metric'],
                    'grow_policy': params['grow_policy'],
                    'tree_method': params['tree_method']
                }

                for param_name, param_value in string_params.items():
                    self.task.get_logger().report_text(
                        f"{param_name}: {param_value}",
                        iteration=trial.number
                    )

            return mean_score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params.copy()
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'logloss'
        best_params['tree_method'] = 'hist'

        n_estimators = best_params.pop('n_estimators', 500)

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best n_estimators: {n_estimators}")
        self.logger.info(f"Best CV score: {study.best_value:.4f}")

        return best_params, n_estimators

    def _train_model(self, X_train, y_train, params, n_estimators):
        """
        Train the final model with the optimized hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            params: Optimized hyperparameters
            n_estimators: Optimized number of estimators

        Returns:
            cv_score: Cross-validation score of the final model
        """
        self.logger.info("Training final model with optimal hyperparameters")

        dtrain = xgb.DMatrix(X_train, label=y_train)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators
        )

        train_preds = self.model.predict(dtrain)

        thresholds = np.linspace(0.3, 0.7, 21)
        scores = [accuracy_score(y_train, (train_preds > t).astype(int)) for t in thresholds]
        best_idx = np.argmax(scores)
        self.best_threshold = thresholds[best_idx]
        train_accuracy = scores[best_idx]

        self.logger.info(f"Best threshold: {self.best_threshold:.4f}")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")

        try:
            import matplotlib.pyplot as plt

            importance = self.model.get_score(importance_type='gain')
            importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*importance[:20])

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), values, align='center')
            plt.yticks(range(len(features)), features)
            plt.title('Feature Importance (Gain)')
            plt.xlabel('Importance')
            plt.tight_layout()

            plt.savefig('feature_importance.png')

            if self.use_clearml and self.task is not None:
                self.task.upload_artifact("feature_importance", 'feature_importance.png')

            plt.plot([], [], label='Feature Importance')
            plt.legend()

            plt.close()
        except Exception as e:
            self.logger.warning(f"Error plotting feature importance: {str(e)}")

        cv_score = None
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val)

                fold_model = xgb.train(
                    params,
                    dtrain_fold,
                    num_boost_round=n_estimators
                )

                fold_preds = fold_model.predict(dval_fold)
                fold_scores = [accuracy_score(y_fold_val, (fold_preds > t).astype(int))
                               for t in thresholds]
                cv_scores.append(max(fold_scores))

            cv_score = np.mean(cv_scores)

            if self.use_clearml and self.task is not None:
                for i, score in enumerate(cv_scores):
                    self.task.get_logger().report_scalar(
                        "Final CV", f"Fold {i + 1}", score, iteration=0
                    )
        except Exception as e:
            self.logger.warning(f"Error during cross-validation: {str(e)}")
            cv_score = train_accuracy

        return cv_score

    def train(self, dataset: str) -> None:
        """
        Train the model on the provided dataset.

        Args:
            dataset: Path to the training dataset.
        """
        start_time = time.time()

        try:
            if self.use_clearml:
                try:
                    self.task = Task.init(
                        project_name="Spaceship Anomaly",
                        task_name=f"XGBoost Training {time.strftime('%Y-%m-%d %H:%M')}",
                        auto_connect_frameworks=True
                    )

                    self.task.connect_configuration({
                        "model_type": "XGBoost",
                        "random_state": self.random_state,
                        "n_trials": self.n_trials,
                        "model_version": "1.0"
                    })

                    if os.path.exists(dataset):
                        self.task.upload_artifact("dataset", dataset)

                    self.logger.info("ClearML task initialized")
                except Exception as e:
                    self.logger.warning(f"ClearML initialization failed: {str(e)}")
                    self.use_clearml = False
                    self.task = None

            self.logger.info(f"Starting training using dataset: {dataset}")

            try:
                train_data = pd.read_csv(dataset)
                self.logger.info(f"Loaded training data with shape: {train_data.shape}")

                if self.use_clearml and self.task is not None:
                    self.task.get_logger().report_text(
                        f"Dataset shape: {train_data.shape}\n"
                        f"Missing values: {train_data.isna().sum().sum()}\n"
                        f"Target distribution: {train_data['Transported'].value_counts(normalize=True).to_dict()}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to load dataset: {str(e)}")
                raise

            X_train, y_train = self._prepare_data(train_data, is_train=True)
            self.logger.info(f"Prepared features. X shape: {X_train.shape}, y shape: {y_train.shape}")

            best_params, n_estimators = self._optimize_hyperparameters(X_train, y_train)
            self.logger.info(f"Hyperparameter optimization completed. Best n_estimators: {n_estimators}")

            if self.use_clearml and self.task is not None:
                for param_name, param_value in best_params.items():
                    if isinstance(param_value, (int, float)):
                        self.task.get_logger().report_scalar(
                            "Best Parameters",
                            param_name,
                            param_value,
                            iteration=0
                        )

                string_params = {k: v for k, v in best_params.items() if isinstance(v, str)}
                if string_params:
                    self.task.get_logger().report_text(
                        "String parameters: " + str(string_params),
                        iteration=0
                    )

            cv_score = self._train_model(X_train, y_train, best_params, n_estimators)
            self.logger.info(
                f"Final model training completed. CV Score: {cv_score if cv_score is not None else 'None'}")

            model_dir = os.path.join(os.getcwd(), 'model')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'model.pkl')

            with open(model_path, 'wb') as f:
                model_data = {
                    'model': self.model,
                    'best_threshold': self.best_threshold,
                    'params': best_params,
                    'n_estimators': n_estimators,
                    'features': list(X_train.columns),
                    'training_date': time.strftime('%Y-%m-%d %H:%M')
                }
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {model_path}")

            if self.use_clearml and self.task is not None:
                try:
                    self.task.upload_artifact("final_model", model_path)

                    self.task.get_logger().report_scalar(
                        "Final Metrics",
                        "CV Score",
                        cv_score,
                        iteration=0
                    )

                    training_time = time.time() - start_time
                    self.task.get_logger().report_scalar(
                        "Performance",
                        "Training Time (seconds)",
                        training_time,
                        iteration=0
                    )

                    self.logger.info("Model artifacts uploaded to ClearML")
                except Exception as e:
                    self.logger.warning(f"Error uploading artifacts to ClearML: {str(e)}")

            self.logger.info("Model training completed successfully.")

            if self.use_clearml and self.task is not None:
                try:
                    self.logger.info("Closing ClearML task...")
                    self.task.close()
                    self.task = None
                    self.logger.info("ClearML task closed successfully.")
                except Exception as e:
                    self.logger.warning(f"Error closing ClearML task: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")

            if self.use_clearml and self.task is not None:
                try:
                    self.logger.info("Closing ClearML task due to error...")
                    self.task.close()
                    self.task = None
                except:
                    pass

            raise

    def predict(self, dataset: str) -> None:
        """
        Make predictions using the trained model.

        Args:
            dataset: Path to the dataset for predictions.
        """
        try:
            self.logger.info(f"Starting prediction using dataset: {dataset}")

            try:
                test_data = pd.read_csv(dataset)
                self.logger.info(f"Loaded test data with shape: {test_data.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load dataset: {str(e)}")
                raise

            model_dir = os.path.join(os.getcwd(), 'model')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'model.pkl')

            if self.model is None:
                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found at {model_path}. Train the model first.")
                    raise FileNotFoundError(f"Model file not found at {model_path}")

                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.best_threshold = model_data['best_threshold']
                    self.logger.info("Model loaded successfully")

            X_test, passenger_ids = self._prepare_data(test_data, is_train=False)

            self.logger.info("Making predictions")
            test_dmatrix = xgb.DMatrix(X_test)
            predictions_prob = self.model.predict(test_dmatrix)
            predictions = (predictions_prob > self.best_threshold).astype(bool)

            self.logger.info("Preparing submission file")
            submission = pd.DataFrame({
                'PassengerId': passenger_ids,
                'Transported': predictions
            })

            data_dir = os.path.join(os.getcwd(), 'data')
            os.makedirs(data_dir, exist_ok=True)
            results_path = os.path.join(data_dir, 'results.csv')

            submission.to_csv(results_path, index=False)


            transport_rate = submission['Transported'].mean() * 100
            self.logger.info(f"Prediction transport rate: {transport_rate:.2f}%")
            self.logger.info(f"Predictions saved to {results_path}")

            if self.use_clearml and clearml_available:
                try:
                    from clearml import Task
                    task = Task.init(project_name="Spaceship Anomaly",
                                     task_name=f"XGBoost Prediction {time.strftime('%Y-%m-%d %H:%M')}",
                                     auto_connect_frameworks=False)
                    task.upload_artifact("predictions", results_path)
                    task.upload_artifact("test_dataset", dataset)
                    self.logger.info("Predictions logged to ClearML")
                except Exception as e:
                    self.logger.warning(f"Failed to log predictions to ClearML: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            self.logger.error("Model not trained yet")
            raise ValueError("Model not trained yet")

        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        return importance_df


def main():
    """Command line interface entry point."""
    fire.Fire(My_Classifier_Model)


if __name__ == "__main__":
    main()
