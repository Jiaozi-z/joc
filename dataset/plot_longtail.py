import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_longtail(dataset_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inter_file = os.path.join(base_dir, dataset_name, f'{dataset_name}.inter')

    if not os.path.exists(inter_file):
        print(f"File not found: {inter_file}")
        sys.exit(1)

    df = pd.read_csv(inter_file, sep='\t')
    item_counts = df['item_id:token'].value_counts()
    freq = item_counts.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(freq.index, freq.values, color='steelblue', width=1.0, edgecolor='steelblue')
    plt.xlabel('Interactions per Item', fontsize=14)
    plt.ylabel('Number of Items', fontsize=14)
    plt.title(f'Long-tail Distribution of Item Interactions ({dataset_name})', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(base_dir, dataset_name, 'longtail_distribution.png')
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Total interactions: {len(df)}")
    print(f"Unique items: {len(item_counts)}")
    print(f"Interactions range: {item_counts.min()} - {item_counts.max()}")
    print(f"Saved to: {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dataset_name>")
        print(f"Example: python {sys.argv[0]} hdf-9580")
        sys.exit(1)
    plot_longtail(sys.argv[1])
