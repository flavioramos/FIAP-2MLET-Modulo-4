from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from utils.config_loader import load_parameters
from config import (
    STATUS_MAP,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_JOB_DESCRIPTION_MAX_FEATURES,
    TFIDF_JOB_REQUIREMENTS_MAX_FEATURES,
    TFIDF_CANDIDATE_CV_MAX_FEATURES,
    TFIDF_JOB_DESCRIPTION_NGRAM_RANGE,
    TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE,
    TFIDF_CANDIDATE_CV_NGRAM_RANGE,
    GRID_SEARCH_CV,
    GRID_SEARCH_SCORING,
    GRID_SEARCH_N_JOBS,
    GRID_SEARCH_C_VALUES,
    LOGISTIC_REGRESSION_MAX_ITER,
)


def create_pipeline(params):
    """Create the machine learning pipeline for job matching.

    Args:
        params (dict): Dictionary of parameters for the pipeline

    Returns:
        Pipeline: Configured scikit-learn pipeline
    """
    preprocessor = ColumnTransformer([
        ("tfidf_ativ", TfidfVectorizer(
            max_features=TFIDF_JOB_DESCRIPTION_MAX_FEATURES,
            ngram_range=TFIDF_JOB_DESCRIPTION_NGRAM_RANGE,
            strip_accents="unicode" 
        ), "job_description"),
        ("tfidf_comp", TfidfVectorizer(
            max_features=TFIDF_JOB_REQUIREMENTS_MAX_FEATURES,
            ngram_range=TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE,
            strip_accents="unicode" 
        ), "job_requirements"),
        ("tfidf_cv", TfidfVectorizer(
            max_features=TFIDF_CANDIDATE_CV_MAX_FEATURES,
            ngram_range=TFIDF_CANDIDATE_CV_NGRAM_RANGE,
            strip_accents="unicode" 
        ), "candidate_cv"),
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER))
    ])

    return pipeline


def train_model(df):
    """Train the job matching model using the configured pipeline and parameters.

    Returns:
        tuple: (auc, grid, X_test, y_test) containing the AUC score, trained grid search object,
               and test data for additional metrics
    """
    print("\n=== Starting Training Process ===")
    params = load_parameters()

    print("\n=== Loading and Processing Data ===")

    print("\n=== Preparing Features and Target ===")
    x = df[["job_description", "job_requirements", "candidate_cv"]]
    y = df["status"].map(STATUS_MAP)
    print(f"Feature shape: {x.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\n=== Creating Pipeline ===")
    pipeline = create_pipeline(params)

    print("\n=== Starting Grid Search ===")
    print(f"Testing C values: {GRID_SEARCH_C_VALUES}")
    print(f"Performing {GRID_SEARCH_CV}-fold cross-validation...")
    print("Progress will be shown below:")

    grid = GridSearchCV(
        pipeline,
        {"clf__C": GRID_SEARCH_C_VALUES},
        cv=GRID_SEARCH_CV,
        scoring=GRID_SEARCH_SCORING,
        n_jobs=GRID_SEARCH_N_JOBS,
        verbose=1
    )

    grid.fit(X_train, y_train)
    print(f"\nBest C value: {grid.best_params_['clf__C']}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")

    print("\n=== Evaluating Model ===")
    y_pred = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    return auc, grid, X_test, y_test

