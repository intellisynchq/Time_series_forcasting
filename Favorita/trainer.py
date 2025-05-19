import warnings
warnings.filterwarnings('ignore')

# Data manipulation libraries
import numpy as np
import pandas as pd

# Darts time series libraries
from darts import TimeSeries
from darts.metrics import rmsle
from darts.models import LinearRegressionModel, LightGBMModel, XGBModel, CatBoostModel

# Progress bar
from tqdm.notebook import tqdm_notebook

class Trainer:
    def __init__(
        self,
        target_dict,
        pipe_dict,
        id_dict,
        past_dict,
        future_dict,
        forecast_horizon,
        folds,
        zero_fc_window,
        static_covs=None,
        past_covs=None,
        future_covs=None,
    ):
        self.target_dict = target_dict.copy()
        self.pipe_dict = pipe_dict.copy()
        self.id_dict = id_dict.copy()
        self.past_dict = past_dict.copy()
        self.future_dict = future_dict.copy()
        self.forecast_horizon = forecast_horizon
        self.folds = folds
        self.zero_fc_window = zero_fc_window
        self.static_covs = static_covs
        self.past_covs = past_covs
        self.future_covs = future_covs
        
        self.setup()
    
    def setup(self):
        for fam in tqdm_notebook(self.target_dict.keys(), desc="Setting up"):
            if self.static_covs != "keep_all":
                if self.static_covs is not None:
                    target = self.target_dict[fam]
                    keep_static = [col for col in target[0].static_covariates.columns if col.startswith(tuple(self.static_covs))]
                    static_covs_df = [t.static_covariates[keep_static] for t in target]
                    self.target_dict[fam] = [t.with_static_covariates(d) for t, d in zip(target, static_covs_df)]
                else:
                    self.target_dict[fam] = [t.with_static_covariates(None) for t in self.target_dict[fam]]
            
            if self.past_covs != "keep_all":
                if self.past_covs is not None:
                    self.past_dict[fam] = [p[self.past_covs] for p in self.past_dict[fam]]
                else:
                    self.past_dict[fam] = None
                
            if self.future_covs != "keep_all":
                if self.future_covs is not None:
                    self.future_dict[fam] = [p[self.future_covs] for p in self.future_dict[fam]]
                else:
                    self.future_dict[fam] = None
    
    def clip(self, array):
        return np.clip(array, a_min=0., a_max=None)
    
    def train_valid_split(self, target, length):
        train = [t[:-length] for t in target]
        valid_end_idx = -length + self.forecast_horizon
        if valid_end_idx >= 0:
            valid_end_idx = None
        valid = [t[-length:valid_end_idx] for t in target]
        
        return train, valid
    
    def get_models(self, model_names, model_configs):
        models = {
            "lr": LinearRegressionModel,
            "lgbm": LightGBMModel,
            "cat": CatBoostModel,
            "xgb": XGBModel,
        }
        assert isinstance(model_names, list) and isinstance(model_configs, list),\
        "Both the model names and model configurations must be specified in lists."
        assert all(name in models for name in model_names),\
        f"Model names '{model_names}' not recognized."
        assert len(model_names) == len(model_configs),\
        "The number of model names and the number of model configurations do not match."
        
        if "xgb" in model_names:
            xgb_idx = np.where(np.array(model_names)=="xgb")[0]
            for idx in xgb_idx:
                model_configs[idx] = {"tree_method": "hist", **model_configs[idx]}
        
        return [models[name](**model_configs[j]) for j, name in enumerate(model_names)]
    
    def generate_forecasts(self, models, train, pipe, past_covs, future_covs, drop_before):
        """
        Generate forecasts using the trained models.
        
        Parameters:
        -----------
        models : list
            List of models to use for forecasting
        train : list
            List of training time series
        pipe : Pipeline
            Transformation pipeline for the target time series
        past_covs : list or None
            List of past covariates time series
        future_covs : list or None
            List of future covariates time series
        drop_before : str or None
            Date before which to drop data
            
        Returns:
        --------
        pred_list : list
            List of predictions for each model
        ens_pred : list
            Ensemble prediction (average of all models)
        """
        if drop_before is not None:
            date = pd.Timestamp(drop_before) - pd.Timedelta(days=1)
            train = [t.drop_before(date) for t in train]
            if past_covs is not None:
                past_covs = [p.drop_before(date) for p in past_covs]
            if future_covs is not None:
                future_covs = [f.drop_before(date) for f in future_covs]
        
        inputs = {
            "series": train,
            "past_covariates": past_covs,
            "future_covariates": future_covs,
        }
        
        # Create zero prediction template for items with no sales
        zero_pred = pd.DataFrame({
            "date": pd.date_range(train[0].end_time(), periods=self.forecast_horizon+1)[1:],
            "sales": np.zeros(self.forecast_horizon),
        })
        zero_pred = TimeSeries.from_dataframe(
            df=zero_pred,
            time_col="date",
            value_cols="sales",
        )
        
        pred_list = []
        ens_pred = [0 for _ in range(len(train))]
        
        for m in models:
            try:
                m.fit(**inputs)
            except Exception as e:
                print(f"Warning: Error during model fitting: {e}")
                continue

            try:
                pred = m.predict(n=self.forecast_horizon, **inputs)
                pred = pipe.inverse_transform(pred)

                # For time series with only zeros in recent history, predict zeros
                for j in range(len(train)):
                    if train[j][-self.zero_fc_window:].values().sum() == 0:
                        pred[j] = zero_pred
                
                # Clip predictions to ensure non-negative values
                pred = [p.map(self.clip) for p in pred]
                pred_list.append(pred)
                
                # Calculate ensemble prediction as average
                for j in range(len(ens_pred)):
                    ens_pred[j] += pred[j] / len(models)
            except Exception as e:
                print(f"Warning: Error during prediction: {e}")
                continue

        # If no models successfully generated predictions, return empty lists
        if not pred_list:
            print("Warning: No models generated predictions successfully.")
            return [], ens_pred

        return pred_list, ens_pred
    
    def metric(self, valid, pred):
        """Calculate Root Mean Squared Logarithmic Error (RMSLE)"""
        result = rmsle(valid, pred)
        if isinstance(result, list):
            return np.mean(result)
        return result
    
    def validate(self, model_names, model_configs, drop_before=None):
        longest_len = len(max(self.target_dict.keys(), key=len))
        
        model_metrics_history = []
        ens_metric_history = []
        
        for fam in tqdm_notebook(self.target_dict, desc="Performing validation"):
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]
            
            model_metrics = []
            ens_metric = 0
            
            for j in range(self.folds):    
                length = (self.folds - j) * self.forecast_horizon
                train, valid = self.train_valid_split(target, length)
                valid = pipe.inverse_transform(valid)

                models = self.get_models(model_names, model_configs)
                pred_list, ens_pred = self.generate_forecasts(
                    models, train, pipe, past_covs, future_covs, drop_before
                )
                
                if pred_list:  # Check if any predictions were successfully generated
                    metric_list = [self.metric(valid, pred) / self.folds for pred in pred_list]
                    model_metrics.append(metric_list)
                    
                    if len(models) > 1:
                        ens_metric_fold = self.metric(valid, ens_pred) / self.folds
                        ens_metric += ens_metric_fold
                
            if model_metrics:  # Check if any metrics were calculated
                model_metrics = np.sum(model_metrics, axis=0)
                model_metrics_history.append(model_metrics)
                ens_metric_history.append(ens_metric)
                
                print(
                    fam,
                    " " * (longest_len - len(fam)),
                    " | ",
                    " - ".join([f"{model}: {metric:.5f}" for model, metric in zip(model_names, model_metrics)]),
                    f" - ens: {ens_metric:.5f}" if len(models) > 1 else "",
                    sep="",
                )
            
        if model_metrics_history:  # Check if any metrics were collected
            print(
                "Average RMSLE | "
                + " - ".join([f"{model}: {metric:.5f}" 
                              for model, metric in zip(model_names, np.mean(model_metrics_history, axis=0))])
                + (f" - ens: {np.mean(ens_metric_history):.5f}" if len(models) > 1 else ""),
            )
        else:
            print("No valid metrics could be calculated. Check model configurations.")
        
    def ensemble_predict(self, model_names, model_configs, drop_before=None):
        """Generate predictions using ensemble of models"""
        forecasts = []
        for fam in tqdm_notebook(self.target_dict.keys(), desc="Generating forecasts"):
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            target_id = self.id_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]
            
            models = self.get_models(model_names, model_configs)
            
            # Use full dataset for final prediction
            _, ens_pred = self.generate_forecasts(models, target, pipe, past_covs, future_covs, drop_before)
            
            if not any(isinstance(p, TimeSeries) for p in ens_pred):
                print(f"Warning: Failed to generate predictions for {fam}")
                continue
                
            # Convert to dataframe and add identifiers
            ens_pred = [p.to_dataframe().assign(**i) for p, i in zip(ens_pred, target_id)]
            ens_pred = pd.concat(ens_pred, axis=0)
            forecasts.append(ens_pred)
            
        forecasts = pd.concat(forecasts, axis=0)
        forecasts = forecasts.rename_axis(None, axis=1).reset_index(names="date")
        
        return forecasts