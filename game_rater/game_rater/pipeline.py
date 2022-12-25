import logging

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropConstantFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import game_rater.utils.preprocessors as pp
from game_rater.configs.config import config

_logger = logging.getLogger(__name__)

# renaming, dropping, renaming group-wise, extracting the cat features, imputing and dropping constant features
pipe = Pipeline(
    [
        ("pre_renamer", pp.FeatureRenamer(feature_names=config.model_config.var_pre_rename)),
        (
            "feature_dropper",
            pp.FeatureDroper(
                features_to_drop=config.model_config.vars_to_drop,
                feature_groups_to_drop=config.model_config.var_groups_to_drop,
            ),
        ),
        ("post_renamer", pp.FeatureRenamer(feature_group_names=config.model_config.var_groups_post_rename)),
        ("extractor", pp.MainFeatureExtractor(feature_names=config.model_config.vars_to_extract)),
        ("num_imputer", MeanMedianImputer(variables=config.model_config.num_vars)),
        ("cat_imputer", CategoricalImputer(variables=config.model_config.cat_vars)),
        ("cat_encoder", pp.OrdinalEncoderWrapper(variables=config.model_config.cat_vars)),
        ("constant_dropper", DropConstantFeatures(tol=config.model_config.drop_tol)),
        ("xgb_regressor", XGBRegressor(**config.model_config.xgboost_params)),
    ]
)
