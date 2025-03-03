import os
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr

from mt_metrics_eval import tau_optimization
from mt_metrics_eval import stats

def read_scores(file_path):
    with open(file_path, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
    return scores

def read_processed_file(file_path):
    with open(file_path, 'r') as file:
        scores = [float(line.strip().split('<ENDSENTENCE>')[-1]) for line in file.readlines()]
        result = []
        for i in scores:
            #result.append(float(i))
            if i <= 33:
                result.append(0)
            elif i <= 66:
                result.append(50)
            else:
                result.append(100)
        return scores, result

def compute_metrics(predicted_scores, observed_scores, M, N):
    X = np.array(predicted_scores).reshape(M, N)
    Y = np.array(observed_scores).reshape(M, N)
    result = tau_optimization.tau_optimization(X, Y, tau_optimization.TauSufficientStats.acc_23)
    metrics = {
        'Best Tau': result.best_tau
    }
    for metric_name, func in [('Kendall Tau', kendalltau), ('Spearman', spearmanr), ('Pearson', pearsonr)]:
        overall, _ = func(X.flatten(), Y.flatten())
        metrics[metric_name] = overall
    return metrics

def get_language_list(directory):
    files = os.listdir(directory)
    return [file.split('-')[1].split('.')[0] for file in files if file.startswith('processed_eng-')]

directory = '.'
languages = get_language_list(directory)
metrics_data = {}
model_accumulator = {}

for lang in languages:
    processed_file_path = os.path.join(directory, f'processed_eng-{lang}.txt')
    _, observed_scores = read_processed_file(processed_file_path)
    model_files = [f'comet22_result/comet22_{lang}_scores.txt', f'ses2_result/ses2_{lang}_scores.txt', f'xcomet_result/xcomet_{lang}_scores.txt']
    metrics_data[lang] = {}

    for model_file in model_files:
        model_name = model_file.split('_')[0]
        predicted_scores = read_scores(os.path.join(directory, model_file))
        N = len(predicted_scores) // 99  # Assuming M=99 from your previous script
        metrics = compute_metrics(predicted_scores, observed_scores, 99, N)
        metrics_data[lang][model_file] = metrics
        if model_name not in model_accumulator:
            model_accumulator[model_name] = {key: [] for key in metrics}
        for key, value in metrics.items():
            model_accumulator[model_name][key].append(value)

# Print LaTeX longtable
print(r"\begin{longtable}{|l|l|l|l|l|l|}")
print(r"\hline")
print(r"Language & Metric & Best Tau & Kendall Tau & Spearman & Pearson \\ \hline")
print(r"\endfirsthead")
print(r"\multicolumn{6}{c}{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\")
print(r"\hline")
print(r"Language & Metric & Best Tau & Kendall Tau & Spearman & Pearson \\ \hline")
print(r"\endhead")
print(r"\hline \multicolumn{6}{|r|}{{Continued on next page}} \\ \hline")
print(r"\endfoot")
print(r"\hline \hline")
print(r"\endlastfoot")

for lang, data in metrics_data.items():
    model_files = list(data.keys())
    last_model = model_files[-1]
    for model_file in model_files:
        model_name = model_file.split('_')[0]
        metrics = data[model_file]
        if model_file == last_model:
            print(f"{lang if model_file == model_files[0] else ''} & {model_name} & {metrics['Best Tau']:.3f} & {metrics['Kendall Tau']:.4f} & {metrics['Spearman']:.4f} & {metrics['Pearson']:.4f} \\\\ \hline")
        else:
            print(f"{lang if model_file == model_files[0] else ''} & {model_name} & {metrics['Best Tau']:.3f} & {metrics['Kendall Tau']:.4f} & {metrics['Spearman']:.4f} & {metrics['Pearson']:.4f} \\\\")

# Average metrics for each model
for model_name, metrics in model_accumulator.items():
    print(r"\hline")
    print(f"Averages & {model_name} & ", end="")
    print(" & ".join(f"{np.mean(values):.4f}" for key, values in metrics.items()) + r" \\\\")

print(r"\end{longtable}")
