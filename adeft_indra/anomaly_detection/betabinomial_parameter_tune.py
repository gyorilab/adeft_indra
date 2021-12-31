import argparse
import itertools as it
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from opaque.ood.utils import AnyMethodPipeline
from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.stats import binomial_score
from adeft_indra.anomaly_detection.results import ResultsManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune parameters for beta-binomial model of"
        " sensitivity or specificity."
    )
    parser.add_argument('data_path')
    parser.add_argument('run_name')
    parser.add_argument('type', help="One of sensitivity or specificity")
    parser.add_argument('--coeff_prior_type_list', nargs='+', type=str)
    parser.add_argument('--coeff_scale_list', nargs='+', type=float)
    parser.add_argument('--numpy_seed', type=int, default=561)
    parser.add_argument('--pymc3_seed', type=int, default=1729)

    args = parser.parse_args()
    gen = np.random.RandomState(args.numpy_seed)

    df = pd.read_csv(args.data_path, sep=',')
    df.sample(frac=1, random_state=gen)
    if args.type == 'specificity':
        df = df[df.N_inlier > 0]
        y = df[['N_inlier', 'K_inlier']].values.astype(float)
    elif args.type == 'sensitivity':
        df = df[df.N_outlier > 0]
        y = df[['N_outlier', 'K_outlier']].values.astype(float)

    df['log_num_entrez'] = np.log(df.num_entrez + 1)
    df['log_num_mesh'] = np.log(df.num_mesh + 1)

    X = df[
        [
            'nu',
            'max_features',
            'log_num_entrez',
            'log_num_mesh',
            'sens_neg_set',
            'mean_spec',
            'std_spec'
        ]
    ].values

    if args.run_name not in ResultsManager.show_tables():
        ResultsManager.add_table(args.run_name)

    for prior_type, prior_scale in it.product(
            args.coeff_prior_type_list, args.coeff_scale_list
    ):
        key = f"{prior_type}:{prior_scale}"
        if ResultsManager.get(args.run_name, key) is not None:
            print(f"Results already computed for {key}")
            continue
        test_scores = []
        baseline_scores = []
        skill_scores = []
        splits = GroupKFold(n_splits=10).split(df, groups=df.shortform)
        for train_idx, test_idx in splits:
            model = AnyMethodPipeline(
                [
                    ('scale', StandardScaler()),
                    (
                        'betabinom',
                        BetaBinomialRegressor(
                            coefficient_prior_type=prior_type,
                            coefficient_prior_scale=prior_scale,
                            random_seed=args.pymc3_seed,
                        ),
                    ),
                ]
            )
            model.fit(X[train_idx], y[train_idx])
            total_samples = np.sum(y[train_idx][:, 0])
            total_successes = np.sum(y[train_idx][:, 1])
            p = total_successes / total_samples
            baseline_preds = np.vstack(
                [np.ones(len(test_idx)), np.full(len(test_idx), p)]
            ).T
            preds = model.predict(X[test_idx], N=y[test_idx, 0])
            test_score = binomial_score(y[test_idx], preds)
            baseline_score = binomial_score(y[test_idx], baseline_preds)
            skill_score = 1 - test_score / baseline_score
            test_scores.append(test_score)
            baseline_scores.append(baseline_score)
            skill_scores.append(skill_score)
        ResultsManager.insert(
            args.run_name,
            key,
            {
                'test_scores': test_scores,
                'baseline_scores': baseline_scores,
                'skill_scores': skill_scores,
                'mean_skill_score': np.mean(skill_scores),
                'mean_baseline_score': np.mean(baseline_scores),
                'mean_test_score': np.mean(test_scores),
                'std_skill_score': np.std(skill_scores),
                'std_baseline_score': np.std(baseline_scores),
                'std_test_score': np.std(test_scores)
            }
        )
