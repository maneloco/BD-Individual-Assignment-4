import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    return df[df['repetition'] >= 0].copy()


def compute_summary_stats(df):
    group_cols = ['experiment', 'method', 'n', 'processes', 'threads_per_proc', 'total_workers']
    grouped = df.groupby(group_cols)

    stats = grouped.agg(
        median_total=('end_to_end_sec', 'median'),
        q25_total=('end_to_end_sec', lambda x: np.percentile(x, 25)),
        q75_total=('end_to_end_sec', lambda x: np.percentile(x, 75)),
        median_dist=('distribute_sec', 'median'),
        median_comp=('compute_sec', 'median'),
        median_comp_max=('compute_max_sec', 'median'),
        median_comp_avg=('compute_avg_sec', 'median'),
        median_gather=('gather_sec', 'median'),
        median_rank0_rss=('rank0_rss_kb', 'median'),
        median_worker_rss=('worker_max_rss_kb', 'median'),
        median_total_rss=('total_rss_kb', 'median'),
        bytes_transferred=('bytes_transferred', 'first')
    ).reset_index()

    stats['iqr_total'] = stats['q75_total'] - stats['q25_total']
    return stats


def get_sequential_lookup(stats):
    seq = stats[stats['method'] == 'seq_ikj']
    if not seq.empty:
        return seq.set_index('n')['median_total'].to_dict()
    mpi_p1 = stats[(stats['experiment'] == 'strong_scaling') &
                   (stats['method'] == 'mpi_ikj') &
                   (stats['processes'] == 1)]
    return mpi_p1.set_index('n')['median_total'].to_dict()


def plot_strong_scaling_speedup(stats, output_dir):
    seq_map = get_sequential_lookup(stats)
    strong = stats[stats['experiment'] == 'strong_scaling'].copy()
    if strong.empty or not seq_map:
        return

    strong['t_seq'] = strong['n'].map(seq_map)
    strong['speedup'] = strong['t_seq'] / strong['median_total']

    available_sizes = sorted(strong['n'].unique())
    target_sizes = [128, 256, 512, 1024, 2048, 4096]
    selected_sizes = [s for s in target_sizes if s in available_sizes]
    if not selected_sizes:
        selected_sizes = available_sizes

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    all_procs = sorted(strong['processes'].unique())
    colormap = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_sizes)))

    methods = [('mpi_ikj', 'Pure MPI (1 thread / node)', axes[0], 1),
               ('mpi_hybrid', 'Hybrid MPI + Pthreads (2 threads / node)', axes[1], 2)]

    for method_name, title, ax, threads_factor in methods:
        sub_m = strong[strong['method'] == method_name]
        if sub_m.empty:
            continue

        ax.plot(all_procs, all_procs, 'k--', label='Ideal Node Speedup (S = P)', alpha=0.7)
        if threads_factor > 1:
            ideal_cores = [p * threads_factor for p in all_procs]
            ax.plot(all_procs, ideal_cores, 'r:', label=f'Ideal Core Speedup (S = {threads_factor}P)', alpha=0.7)

        for idx, n in enumerate(selected_sizes):
            sub_n = sub_m[sub_m['n'] == n].sort_values('processes')
            if not sub_n.empty:
                ax.plot(sub_n['processes'], sub_n['speedup'], '-o', label=f'n = {n}', color=colormap[idx])

        ax.set_xlabel('Number of Distributed Nodes (P)')
        ax.set_title(title)
        ax.set_xticks(all_procs)
        ax.grid(True)
        ax.legend(loc='upper left', frameon=True)

    axes[0].set_ylabel('Speedup S(P) = T_seq / T_dist(P)')
    plt.suptitle('Strong Scaling Speedup Across Distributed Nodes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strong_scaling_speedup.png'), dpi=300)
    plt.close()
    print("  [SAVED] strong_scaling_speedup.png")


def plot_strong_scaling_efficiency(stats, output_dir):
    seq_map = get_sequential_lookup(stats)
    strong = stats[stats['experiment'] == 'strong_scaling'].copy()
    if strong.empty or not seq_map:
        return

    strong['t_seq'] = strong['n'].map(seq_map)
    strong['speedup'] = strong['t_seq'] / strong['median_total']
    strong['efficiency'] = strong['speedup'] / strong['total_workers']

    available_sizes = sorted(strong['n'].unique())
    target_sizes = [128, 256, 512, 1024, 2048, 4096]
    selected_sizes = [s for s in target_sizes if s in available_sizes]
    if not selected_sizes:
        selected_sizes = available_sizes

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    all_procs = sorted(strong['processes'].unique())
    colormap = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_sizes)))

    methods = [('mpi_ikj', 'Pure MPI (1 thread / node)', axes[0]),
               ('mpi_hybrid', 'Hybrid MPI + Pthreads (per total worker cores)', axes[1])]

    for method_name, title, ax in methods:
        sub_m = strong[strong['method'] == method_name]
        if sub_m.empty:
            continue

        ax.axhline(y=1.0, color='k', linestyle='--', label='Ideal Efficiency (100%)', alpha=0.7)

        for idx, n in enumerate(selected_sizes):
            sub_n = sub_m[sub_m['n'] == n].sort_values('processes')
            if not sub_n.empty:
                ax.plot(sub_n['processes'], sub_n['efficiency'], '-s', label=f'n = {n}', color=colormap[idx])

        ax.set_xlabel('Number of Distributed Nodes (P)')
        ax.set_title(title)
        ax.set_xticks(all_procs)
        ax.grid(True)
        ax.legend(loc='upper right', frameon=True)

    axes[0].set_ylabel('Parallel Efficiency E = S / (P * threads)')
    plt.suptitle('Parallel Efficiency vs Distributed Node Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strong_scaling_efficiency.png'), dpi=300)
    plt.close()
    print("  [SAVED] strong_scaling_efficiency.png")


def plot_execution_time_comparison(stats, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    seq = stats[stats['method'] == 'seq_ikj'].sort_values('n')
    if not seq.empty:
        ax.plot(seq['n'], seq['median_total'], 'k-o', label='Sequential Baseline (seq_ikj, 1 core)', linewidth=2.5)

    par = stats[(stats['method'] == 'par_ikj') & (stats['threads_per_proc'] > 1)].sort_values('n')
    if not par.empty:
        t_cnt = int(par['threads_per_proc'].iloc[0])
        ax.plot(par['n'], par['median_total'], '--^', color='#377eb8',
                label=f'Shared-Memory Parallel (par_ikj, 1 node x {t_cnt} threads)')

    strong = stats[stats['experiment'] == 'strong_scaling']
    max_p = strong['processes'].max() if not strong.empty else 4

    mpi_pure = strong[(strong['method'] == 'mpi_ikj') & (strong['processes'] == max_p)].sort_values('n')
    if not mpi_pure.empty:
        ax.plot(mpi_pure['n'], mpi_pure['median_total'], '-s', color='#4daf4a',
                label=f'Distributed Pure MPI (mpi_ikj, {max_p} nodes x 1 thread)')

    mpi_hyb = strong[(strong['method'] == 'mpi_hybrid') & (strong['processes'] == max_p)].sort_values('n')
    if not mpi_hyb.empty:
        t_cnt = int(mpi_hyb['threads_per_proc'].iloc[0])
        ax.plot(mpi_hyb['n'], mpi_hyb['median_total'], '-D', color='#e41a1c',
                label=f'Distributed Hybrid MPI+Pthreads ({max_p} nodes x {t_cnt} threads)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension n (log2 scale)')
    ax.set_ylabel('End-to-End Execution Time [s] (log scale)')
    ax.set_title('Execution Time Comparison: Sequential vs Shared-Memory vs Distributed')
    ax.grid(True, which='both')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_vs_n.png'), dpi=300)
    plt.close()
    print("  [SAVED] execution_time_vs_n.png")


def plot_communication_breakdown(stats, output_dir):
    strong = stats[stats['experiment'] == 'strong_scaling']
    if strong.empty:
        return

    max_p = strong['processes'].max()
    sub = strong[(strong['method'] == 'mpi_hybrid') & (strong['processes'] == max_p)].sort_values('n')
    if sub.empty:
        sub = strong[(strong['method'] == 'mpi_ikj') & (strong['processes'] == max_p)].sort_values('n')
    if sub.empty:
        return

    total = sub['median_dist'] + sub['median_comp'] + sub['median_gather']
    dist_pct = (sub['median_dist'] / total) * 100.0
    comp_pct = (sub['median_comp'] / total) * 100.0
    gath_pct = (sub['median_gather'] / total) * 100.0

    labels = [str(int(n)) for n in sub['n']]
    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, dist_pct, width, label='Distribute (MPI_Scatterv A + MPI_Bcast B)', color='#377eb8')
    ax.bar(x, comp_pct, width, bottom=dist_pct, label='Local Compute (ikj kernel)', color='#4daf4a')
    ax.bar(x, gath_pct, width, bottom=dist_pct + comp_pct, label='Collect (MPI_Gatherv C)', color='#ff7f00')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Matrix Dimension n')
    ax.set_ylabel('Share of End-to-End Runtime [%]')
    ax.set_title(f'Communication vs Computation Breakdown (P = {max_p} Distributed Nodes)')
    ax.grid(True, axis='y')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'communication_breakdown.png'), dpi=300)
    plt.close()
    print("  [SAVED] communication_breakdown.png")


def plot_memory_scaling(stats, output_dir):
    strong = stats[stats['experiment'] == 'strong_scaling']
    if strong.empty:
        return

    max_p = strong['processes'].max()
    sub = strong[(strong['method'] == 'mpi_ikj') & (strong['processes'] == max_p)].sort_values('n')
    if sub.empty:
        return

    n_vals = sub['n']
    rank0_mb = sub['median_rank0_rss'] / 1024.0
    worker_mb = sub['median_worker_rss'] / 1024.0
    total_cluster_mb = sub['median_total_rss'] / 1024.0
    theoretical_single_mb = (3.0 * 8.0 * (n_vals ** 2)) / (1024.0 * 1024.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(n_vals, total_cluster_mb, '-s', color='#e41a1c', label=f'Total Cluster RSS (Sum across {max_p} nodes)')
    ax.plot(n_vals, rank0_mb, '-o', color='#377eb8', label='Coordinator Node Peak RSS (Rank 0)')
    ax.plot(n_vals, worker_mb, '-^', color='#4daf4a', label='Worker Node Peak RSS (Ranks 1..P-1)')
    ax.plot(n_vals, theoretical_single_mb, 'k--', alpha=0.7, label='Theoretical Single-Node Matrix Storage (3 * 8n^2)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension n (log2 scale)')
    ax.set_ylabel('Peak Memory [MB] (log scale)')
    ax.set_title(f'Per-Node and Cluster Memory Scaling vs Matrix Size (P = {max_p} Nodes)')
    ax.grid(True, which='both')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_scaling.png'), dpi=300)
    plt.close()
    print("  [SAVED] memory_scaling.png")


def plot_weak_scaling(stats, output_dir):
    weak = stats[stats['experiment'] == 'weak_scaling'].sort_values('processes')
    if weak.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for method_name, marker, color, label in [
        ('mpi_ikj', '-o', '#377eb8', 'Pure MPI (1 thread / node)'),
        ('mpi_hybrid', '-s', '#e41a1c', 'Hybrid MPI + Pthreads (2 threads / node)')
    ]:
        sub = weak[weak['method'] == method_name].sort_values('processes')
        if not sub.empty:
            ax.plot(sub['processes'], sub['median_total'], marker, color=color, label=f'{label} - End-to-End')
            ax.plot(sub['processes'], sub['median_comp'], '--', color=color, alpha=0.65, label=f'{label} - Compute Only')

    x_ticks = sorted(weak['processes'].unique())
    x_labels = []
    for p in x_ticks:
        n_val = int(weak[weak['processes'] == p]['n'].iloc[0])
        x_labels.append(f'P={p}\n(n={n_val})')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Distributed Nodes P and Scaled Matrix Size n = round(n0 * P^(1/3))')
    ax.set_ylabel('Execution Time [s]')
    ax.set_title('Weak Scaling: Constant O(n^3 / P) Arithmetic Work per Node')
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weak_scaling.png'), dpi=300)
    plt.close()
    print("  [SAVED] weak_scaling.png")


def main():
    results_dir = 'results'
    output_dir = os.path.join(results_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    csv_name = sys.argv[1] if len(sys.argv) > 1 else 'distributed_benchmark.csv'
    csv_path = os.path.join(results_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    print(f"Loading benchmark results from: {csv_path}")
    df = load_data(csv_path)
    if df is None or df.empty:
        print("Error: No data to plot.")
        return

    stats = compute_summary_stats(df)

    print("Generating Assignment 4 publication figures...")
    plot_strong_scaling_speedup(stats, output_dir)
    plot_strong_scaling_efficiency(stats, output_dir)
    plot_execution_time_comparison(stats, output_dir)
    plot_communication_breakdown(stats, output_dir)
    plot_memory_scaling(stats, output_dir)
    plot_weak_scaling(stats, output_dir)
    print(f"All figures generated successfully in: {output_dir}")


if __name__ == '__main__':
    main()
