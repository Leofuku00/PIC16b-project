"""Model registry for clarity rating experiments."""

from typing import List

from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

from clarity_types import ModelSpec, OrdinalLogisticClassifier

from xgboost import XGBClassifier


class ModelRegistry:
    """Build and expose all supported model specifications."""

    def __init__(self, random_state: int):
        """Store random seed used by stochastic model constructors."""
        self.random_state = random_state

    def get_specs(self) -> List[ModelSpec]:
        """Return configured model specs used during CV and test evaluation."""
        specs = [
            ModelSpec(name="multinomial_logreg", family="classifier", build=lambda: LogisticRegression(solver="saga", penalty="l2", max_iter=2000, multi_class="multinomial", random_state=self.random_state, n_jobs=-1)),
            ModelSpec(name="linear_svc", family="classifier", build=lambda: LinearSVC()),
            ModelSpec(name="random_forest_cls", family="classifier", build=lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=self.random_state, class_weight="balanced_subsample")),
            ModelSpec(name="hist_gradient_boosting", family="classifier_dense", build=lambda: HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300, random_state=self.random_state)),
            ModelSpec(name="bagging_linear", family="classifier", build=lambda: BaggingClassifier(estimator=LogisticRegression(solver="saga", penalty="l2", max_iter=1000, multi_class="multinomial", random_state=self.random_state, n_jobs=-1), n_estimators=10, max_samples=0.8, bootstrap=True, n_jobs=-1, random_state=self.random_state)),
            ModelSpec(name="stacking_lsvc_logreg_rf", family="classifier", build=lambda: StackingClassifier(estimators=[("lsvc", LinearSVC()), ("logreg", LogisticRegression(solver="saga", penalty="l2", max_iter=1000, multi_class="multinomial", random_state=self.random_state, n_jobs=1)), ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=1, random_state=self.random_state, class_weight="balanced_subsample"))], final_estimator=LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="multinomial", random_state=self.random_state), cv=2, n_jobs=1)),
            ModelSpec(name="ridge_regression_round", family="regressor_dense", build=lambda: Ridge(alpha=1.0)),
            ModelSpec(name="ordinal_logistic", family="ordinal", build=lambda: OrdinalLogisticClassifier(min_label=1, max_label=5, random_state=self.random_state)),
        ]

        specs.append(ModelSpec(name="xgboost_cls", family="classifier", build=lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="multi:softmax", num_class=5, eval_metric="mlogloss", tree_method="hist", device="cpu", random_state=self.random_state, n_jobs=4)))

        return specs
