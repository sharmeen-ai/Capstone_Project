"""Generate training and evaluation scatter plots from model metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if 'epoch' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'epoch'})

    return df


def plot_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df[x_column],
        df[y_column],
        c=df.get('epoch', None),
        cmap='viridis',
        s=80,
        alpha=0.85,
        edgecolor='k',
    )

    if 'epoch' in df.columns:
        plt.colorbar(scatter, label='epoch')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Wrote {output_path}')


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'}
    if not required_columns.issubset(metrics.columns):
        missing = ', '.join(sorted(required_columns - set(metrics.columns)))
        raise ValueError(
            f'Missing required metric columns: {missing}. '
            'CSV must include train_loss, val_loss, train_accuracy, val_accuracy.'
        )

    plot_scatter(
        metrics,
        x_column='train_loss',
        y_column='val_loss',
        title='Training Loss vs. Validation Loss',
        xlabel='Training Loss',
        ylabel='Validation Loss',
        output_path=output_dir / 'train_vs_val_loss.png',
    )

    plot_scatter(
        metrics,
        x_column='train_accuracy',
        y_column='val_accuracy',
        title='Training Accuracy vs. Validation Accuracy',
        xlabel='Training Accuracy',
        ylabel='Validation Accuracy',
        output_path=output_dir / 'train_vs_val_accuracy.png',
    )

    if {'predictions', 'targets'}.issubset(metrics.columns):
        plot_scatter(
            metrics,
            x_column='targets',
            y_column='predictions',
            title='Predicted vs. Actual Values',
            xlabel='Actual Values',
            ylabel='Predicted Values',
            output_path=output_dir / 'predicted_vs_actual.png',
        )


def create_example_metrics() -> pd.DataFrame:
    epochs = list(range(1, 21))
    train_loss = [1.4 / (1 + 0.12 * e) for e in epochs]
    val_loss = [1.6 / (1 + 0.09 * e) + 0.02 for e in epochs]
    train_accuracy = [0.55 + 0.02 * e for e in epochs]
    val_accuracy = [0.5 + 0.018 * e for e in epochs]

    return pd.DataFrame(
        {
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create scatter plots for model training and evaluation metrics.'
    )
    parser.add_argument(
        '--metrics',
        type=Path,
        help='Path to a CSV file containing model metrics.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('plots'),
        help='Directory where plot PNG files will be saved.',
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Generate example plots using synthetic training/evaluation data.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.example:
        metrics = create_example_metrics()
    elif args.metrics:
        metrics = load_metrics(args.metrics)
    else:
        raise SystemExit('Provide either --metrics <file.csv> or --example.')

    plot_metrics(metrics, args.output_dir)


if __name__ == '__main__':
    main()
