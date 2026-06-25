from datetime import datetime

from .figures import (
    build_val_acc_figure,
    build_train_loss_figure,
    build_best_acc_bar,
    build_throughput_chart,
    build_ranking_table,
)


def generate_report(snapshot: dict, output_path: str) -> None:
    """Generate a standalone HTML report from a metrics snapshot."""

    sections = []

    # Val accuracy
    fig1 = build_val_acc_figure(snapshot)
    sections.append(fig1.to_html(full_html=False, include_plotlyjs=False))

    # Training loss
    fig2 = build_train_loss_figure(snapshot)
    sections.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # Best accuracy bar
    fig3 = build_best_acc_bar(snapshot)
    sections.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # Throughput
    fig4 = build_throughput_chart(snapshot)
    sections.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # Ranking table
    fig5 = build_ranking_table(snapshot)
    sections.append(fig5.to_html(full_html=False, include_plotlyjs=False))

    # Assemble full HTML
    body = "\n".join(f'<div class="chart">{s}</div>' for s in sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIFAR-10 Optimizer Benchmark Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #333; }}
  .chart {{ margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>CIFAR-10 Optimizer Benchmark</h1>
<p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
{body}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
