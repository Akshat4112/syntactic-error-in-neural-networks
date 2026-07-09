import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, stat_fn=np.mean, seed=42):
    rng = np.random.RandomState(seed)
    boot_stats = []
    data = np.asarray(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(stat_fn(data)), float(lower), float(upper)


def mcnemar_test(correct_a, correct_b):
    a_right_b_wrong = ((correct_a == 1) & (correct_b == 0)).sum()
    a_wrong_b_right = ((correct_a == 0) & (correct_b == 1)).sum()
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'a_right_b_wrong': 0, 'a_wrong_b_right': 0}
    # McNemar's test with continuity correction
    stat = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(stat, df=1)
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'a_right_b_wrong': int(a_right_b_wrong),
        'a_wrong_b_right': int(a_wrong_b_right),
    }


class HumanModelComparison:
    def __init__(self):
        human_data_dir = PROJECT_ROOT / 'data' / 'human_behavior_data'
        self.datasets = {}
        for name in ['SPEEDED_RSVP', 'SPEEDED_SPR', 'UNSPEEDED']:
            path = human_data_dir / f'df_{name}.csv'
            if not path.exists():
                path = human_data_dir / f'{name}.csv'
            self.datasets[name] = pd.read_csv(path)
        self.results_dir = PROJECT_ROOT / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_critical_trials(self, df):
        return df[df['Condition'] != 'filler'].copy()

    def _decode_condition(self, condition):
        if condition == 'filler' or condition == '-':
            return None
        subject_num = 'singular' if condition[0] == 's' else 'plural'
        attractor_num = 'singular' if condition[1] == 's' else 'plural'
        phrase_type = condition.split('_')[1] if '_' in condition else 'unknown'
        match = (condition[0] == condition[1])
        return {
            'subject_number': subject_num,
            'attractor_number': attractor_num,
            'phrase_type': phrase_type,
            'number_match': match,
        }

    def accuracy_by_condition(self, model_col):
        all_results = {}
        for ds_name, df in self.datasets.items():
            if model_col not in df.columns:
                print(f"  {ds_name}: model column '{model_col}' not found, skipping")
                continue
            critical = self._get_critical_trials(df)
            if len(critical) == 0:
                continue

            # Normalize model predictions to 0/1
            preds = critical[model_col].copy()
            if preds.dtype == object:
                preds = preds.map(lambda x: int(str(x).replace('LABEL_', '')))

            # Determine expected label from condition
            # s* conditions expect singular (0), p* conditions expect plural (1)
            expected = critical['Condition'].map(lambda c: 0 if c[0] == 's' else 1)
            model_correct = (preds == expected).astype(int)

            results_by_cond = {}
            for cond in sorted(critical['Condition'].unique()):
                mask = critical['Condition'] == cond
                human_acc, h_lo, h_hi = bootstrap_ci(critical.loc[mask, 'Correct'].values)
                model_acc, m_lo, m_hi = bootstrap_ci(model_correct[mask].values)
                results_by_cond[cond] = {
                    'human_accuracy': human_acc,
                    'human_ci': [h_lo, h_hi],
                    'model_accuracy': model_acc,
                    'model_ci': [m_lo, m_hi],
                    'n': int(mask.sum()),
                    'condition_info': self._decode_condition(cond),
                }

            # Overall
            human_overall, ho_lo, ho_hi = bootstrap_ci(critical['Correct'].values)
            model_overall, mo_lo, mo_hi = bootstrap_ci(model_correct.values)

            # By match/mismatch
            match_results = {}
            for m_val in ['0', '1']:
                mask = critical['Match'] == m_val
                if mask.sum() > 0:
                    h_acc, h_lo, h_hi = bootstrap_ci(critical.loc[mask, 'Correct'].values)
                    m_acc, m_lo, m_hi = bootstrap_ci(model_correct[mask].values)
                    label = 'mismatch' if m_val == '0' else 'match'
                    match_results[label] = {
                        'human_accuracy': h_acc,
                        'human_ci': [h_lo, h_hi],
                        'model_accuracy': m_acc,
                        'model_ci': [m_lo, m_hi],
                        'n': int(mask.sum()),
                    }

            # McNemar's test: human vs model
            mcnemar = mcnemar_test(
                critical['Correct'].values,
                model_correct.values,
            )

            # Correlation between human and model accuracy across conditions
            cond_list = sorted(results_by_cond.keys())
            human_accs = [results_by_cond[c]['human_accuracy'] for c in cond_list]
            model_accs = [results_by_cond[c]['model_accuracy'] for c in cond_list]
            if len(cond_list) > 2:
                corr, corr_p = stats.pearsonr(human_accs, model_accs)
            else:
                corr, corr_p = float('nan'), float('nan')

            all_results[ds_name] = {
                'overall': {
                    'human_accuracy': human_overall,
                    'human_ci': [ho_lo, ho_hi],
                    'model_accuracy': model_overall,
                    'model_ci': [mo_lo, mo_hi],
                    'n': len(critical),
                },
                'by_condition': results_by_cond,
                'by_match': match_results,
                'mcnemar_test': mcnemar,
                'correlation': {
                    'pearson_r': float(corr) if not np.isnan(corr) else None,
                    'p_value': float(corr_p) if not np.isnan(corr_p) else None,
                },
            }

        return all_results

    def compare_models(self, model_cols):
        all_comparisons = {}
        for ds_name, df in self.datasets.items():
            available = [m for m in model_cols if m in df.columns]
            if len(available) < 2:
                continue
            critical = self._get_critical_trials(df)
            if len(critical) == 0:
                continue

            expected = critical['Condition'].map(lambda c: 0 if c[0] == 's' else 1)
            pairwise = {}
            for i, m1 in enumerate(available):
                for m2 in available[i + 1:]:
                    p1 = critical[m1].copy()
                    p2 = critical[m2].copy()
                    if p1.dtype == object:
                        p1 = p1.map(lambda x: int(str(x).replace('LABEL_', '')))
                    if p2.dtype == object:
                        p2 = p2.map(lambda x: int(str(x).replace('LABEL_', '')))
                    c1 = (p1 == expected).astype(int)
                    c2 = (p2 == expected).astype(int)
                    mcnemar = mcnemar_test(c1.values, c2.values)
                    pairwise[f'{m1}_vs_{m2}'] = mcnemar
            all_comparisons[ds_name] = pairwise
        return all_comparisons

    def run_full_analysis(self, model_cols=None):
        if model_cols is None:
            model_cols = ['LSTM', 'BERT', 'RNN']

        print("=" * 60)
        print("HUMAN vs MODEL COMPARISON ANALYSIS")
        print("=" * 60)

        all_results = {}
        for model_col in model_cols:
            print(f"\n--- {model_col} ---")
            results = self.accuracy_by_condition(model_col)
            for ds_name, ds_results in results.items():
                overall = ds_results['overall']
                print(f"\n  {ds_name}:")
                print(f"    Human accuracy:  {overall['human_accuracy']:.4f} "
                      f"[{overall['human_ci'][0]:.4f}, {overall['human_ci'][1]:.4f}]")
                print(f"    Model accuracy:  {overall['model_accuracy']:.4f} "
                      f"[{overall['model_ci'][0]:.4f}, {overall['model_ci'][1]:.4f}]")
                corr = ds_results['correlation']
                if corr['pearson_r'] is not None:
                    print(f"    Correlation (r): {corr['pearson_r']:.4f} (p={corr['p_value']:.4f})")
                mcn = ds_results['mcnemar_test']
                sig = "significant" if mcn['p_value'] < 0.05 else "not significant"
                print(f"    McNemar's test:  chi2={mcn['statistic']:.2f}, p={mcn['p_value']:.4f} ({sig})")

                if ds_results['by_match']:
                    print("    By agreement:")
                    for label, mr in ds_results['by_match'].items():
                        print(f"      {label}: human={mr['human_accuracy']:.4f}, "
                              f"model={mr['model_accuracy']:.4f} (n={mr['n']})")
            all_results[model_col] = results

        # Pairwise model comparisons
        available = [m for m in model_cols
                     if any(m in df.columns for df in self.datasets.values())]
        if len(available) >= 2:
            print("\n--- Pairwise Model Comparisons (McNemar's test) ---")
            comparisons = self.compare_models(available)
            for ds_name, pairs in comparisons.items():
                print(f"\n  {ds_name}:")
                for pair, result in pairs.items():
                    sig = "significant" if result['p_value'] < 0.05 else "not significant"
                    print(f"    {pair}: chi2={result['statistic']:.2f}, "
                          f"p={result['p_value']:.4f} ({sig})")
            all_results['pairwise_comparisons'] = comparisons

        # Save structured results
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_path = self.results_dir / 'human_model_comparison.json'
        with open(output_path, 'w') as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"\nFull results saved to {output_path}")
        return all_results
