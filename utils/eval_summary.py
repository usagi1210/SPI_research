"""
扫描 results/ 下所有算法的测试日志，汇总成 PSNR/SSIM 对比表。

用法：
    python utils/eval_summary.py                        # 所有算法、所有采样率
    python utils/eval_summary.py --test_set BSD68       # 指定测试集
    python utils/eval_summary.py --latex                # 额外输出 LaTeX 表格
    python utils/eval_summary.py --csv results.csv      # 保存 CSV
"""
import os
import re
import argparse
from collections import defaultdict

_HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

RATIOS = [1, 4, 10, 25, 40, 50]


def parse_log(log_path):
    """从单个 log 文件里提取 (psnr, ssim, test_set, ratio)，返回 list。"""
    entries = []
    pattern = re.compile(
        r'Avg PSNR=([0-9.]+)\s+Avg SSIM=([0-9.]+)\s+\|\s+set=(\w+)\s+ratio=(\d+)%'
    )
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries.append({
                    'psnr':     float(m.group(1)),
                    'ssim':     float(m.group(2)),
                    'test_set': m.group(3),
                    'ratio':    int(m.group(4)),
                })
    return entries


def collect_results(test_set_filter=None):
    """遍历 results/*/logs/，收集所有结果。返回 {algo: {ratio: {psnr, ssim}}}"""
    data = defaultdict(dict)

    if not os.path.exists(RESULTS_DIR):
        print(f'[ERROR] 找不到 results 目录：{RESULTS_DIR}')
        return data

    for algo in sorted(os.listdir(RESULTS_DIR)):
        log_dir = os.path.join(RESULTS_DIR, algo, 'logs')
        if not os.path.isdir(log_dir):
            continue
        for fname in os.listdir(log_dir):
            if not fname.startswith('test_') or not fname.endswith('.txt'):
                continue
            entries = parse_log(os.path.join(log_dir, fname))
            for e in entries:
                if test_set_filter and e['test_set'] != test_set_filter:
                    continue
                key = (e['test_set'], e['ratio'])
                # 同一个 (algo, ratio) 有多条时取最后一条（最新 epoch）
                data[algo][key] = (e['psnr'], e['ssim'])

    return data


def print_table(data, ratios=RATIOS, test_set='Set11'):
    algos = sorted(data.keys())
    if not algos:
        print('暂无结果，请先运行 test.py。')
        return

    col_w = 14
    header = f"{'Algorithm':<20}" + ''.join(f"{'CS '+str(r)+'%':>{col_w}}" for r in ratios)
    sep    = '-' * len(header)

    print(f'\n=== PSNR (dB) / SSIM  |  Test set: {test_set} ===\n')
    print(header)
    print(sep)
    for algo in algos:
        row = f'{algo:<20}'
        for r in ratios:
            key = (test_set, r)
            if key in data[algo]:
                p, s = data[algo][key]
                row += f'{f"{p:.2f}/{s:.4f}":>{col_w}}'
            else:
                row += f"{'—':>{col_w}}"
        print(row)
    print(sep)


def to_latex(data, ratios=RATIOS, test_set='Set11'):
    algos = sorted(data.keys())
    cols  = 'l' + 'c' * len(ratios)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{PSNR (dB) / SSIM on ' + test_set + r'}',
        r'\begin{tabular}{' + cols + r'}',
        r'\toprule',
        'Method & ' + ' & '.join(f'CS {r}\\%' for r in ratios) + r' \\',
        r'\midrule',
    ]
    for algo in algos:
        cells = [algo.replace('_', r'\_')]
        for r in ratios:
            key = (test_set, r)
            if key in data[algo]:
                p, s = data[algo][key]
                cells.append(f'{p:.2f}/{s:.4f}')
            else:
                cells.append('—')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def to_csv(data, ratios=RATIOS, test_set='Set11'):
    rows = ['Algorithm,' + ','.join(f'PSNR_CS{r},SSIM_CS{r}' for r in ratios)]
    for algo in sorted(data.keys()):
        cells = [algo]
        for r in ratios:
            key = (test_set, r)
            if key in data[algo]:
                p, s = data[algo][key]
                cells += [f'{p:.4f}', f'{s:.6f}']
            else:
                cells += ['', '']
        rows.append(','.join(cells))
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description='汇总所有算法的测试结果')
    parser.add_argument('--test_set', type=str, default='Set11')
    parser.add_argument('--ratios',   type=int, nargs='+', default=RATIOS)
    parser.add_argument('--latex',    action='store_true', help='输出 LaTeX 表格')
    parser.add_argument('--csv',      type=str, default='', help='保存 CSV 到指定路径')
    args = parser.parse_args()

    data = collect_results(test_set_filter=args.test_set)
    print_table(data, ratios=args.ratios, test_set=args.test_set)

    if args.latex:
        print('\n=== LaTeX ===\n')
        print(to_latex(data, ratios=args.ratios, test_set=args.test_set))

    if args.csv:
        csv_str = to_csv(data, ratios=args.ratios, test_set=args.test_set)
        with open(args.csv, 'w') as f:
            f.write(csv_str)
        print(f'\nCSV 已保存：{args.csv}')


if __name__ == '__main__':
    main()
