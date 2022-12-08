import typing as t

import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from sklearn.base import TransformerMixin


class FeatureDroper(TransformerMixin):
    def __init__(self, features_to_drop: t.Sequence[str] = None, feature_groups_to_drop: t.Sequence[str] = None):
        """Drop the redundant features and features groups (dataset specific)."""

        self.features_to_drop = features_to_drop
        self.feature_groups_to_drop = feature_groups_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Overriding fit. It's not used here.
        """

        return self

    def transform(self, X: pd.DataFrame):
        """There are feature subgroups in the dataset that are redundant and should be dropped.
        To not list them all separately, the feature groups_to_drop is created."""
        X = X.copy()
        X = X.drop(self.features_to_drop, axis=1)
        if self.feature_groups_to_drop is not None:
            for feature_group in self.feature_groups_to_drop:
                cols = X.columns[X.columns.str.startswith(feature_group)]
                X = X.drop(cols, axis=1)
        return X


class FeatureRenamer(TransformerMixin):
    def __init__(self, feature_names: t.Dict[str, str] = None, feature_group_names: t.Sequence[str] = None) -> None:
        """Rename individual features and also features groups, getting rid of redundant name chaining."""
        self.feature_names = feature_names
        self.feature_group_names = feature_group_names

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Overriding fit. It's not used here.
        """

        return self

    def transform(self, X: pd.DataFrame):
        if self.feature_names is not None:
            X = X.rename(self.feature_names, axis=1)
        if self.feature_group_names is not None:
            name_dict = dict()
            for feature_group in self.feature_group_names:
                cols = X.columns[X.columns.str.startswith(feature_group)]
                new_cols = [".".join(col.split(".")[1:]) for col in cols]
                new_names = {cols[i]: new_cols[i] for i in range(len(cols))}
                name_dict.update(new_names)
            X = X.rename(name_dict, axis=1)
        return X


class OrdinalEncoderWrapper(TransformerMixin):
    """Wrapper on ordinal encoder."""

    def __init__(self, variables):
        self.variables = variables
        self.encoder = OrdinalEncoder(encoding_method="arbitrary", variables=self.variables)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        X = self.encoder.transform(X)
        return X


class MainFeatureExtractor(TransformerMixin):
    """This class is needed to extract the main text feature from multi-variant categorical features listing several
    values, i.e. for mechanics it can be strategy / tactics, strategy as the main category is extracted."""

    def __init__(self, feature_names: t.Sequence[str], splitter: str = None):
        self.feature_names = feature_names
        self.splitter = splitter if splitter else ","

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Overriding fit. It's not used here.
        """

        return self

    def transform(self, X: pd.DataFrame):
        for feature in self.feature_names:
            X[feature] = X[feature].apply(lambda x: x.split(self.splitter)[0] if x else x)
        return X
