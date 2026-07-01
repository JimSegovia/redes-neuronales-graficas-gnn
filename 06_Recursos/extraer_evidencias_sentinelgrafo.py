import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "02_Notebooks_y_Codigo" / "SentinelGrafo_Local.ipynb"
OUTPUT_DIR = ROOT / "06_Recursos" / "evidencias_sentinelgrafo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_notebook_cells(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook["cells"]


def safe_display(*args, **kwargs):
    # Fuera de Jupyter no necesitamos renderizar widgets/tablas.
    return None


def patch_matplotlib(namespace, figure_prefix):
    original_show = namespace["plt"].show

    def saving_show(*args, **kwargs):
        figures = [plt.figure(num) for num in plt.get_fignums()]
        if not figures:
            return original_show(*args, **kwargs)
        for idx, fig in enumerate(figures, start=1):
            filename = OUTPUT_DIR / f"{figure_prefix}_{idx}.png"
            fig.savefig(filename, bbox_inches="tight")
            print(f"[FIGURA] {filename}")
        plt.close("all")
        return None

    namespace["plt"].show = saving_show


def to_float(value):
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def serialize_report(y_true, y_pred, target_names):
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def collect_metrics(namespace):
    baseline_c1 = namespace["baseline_c1"]
    baseline_c2 = namespace["baseline_c2"]
    all_results = namespace["all_results"]

    label_names_c1 = ["No desastre", "Desastre real"]
    label_names_c2 = ["World", "Sports", "Business", "Sci/Tech"]

    def bundle_case(case_key_prefix, baseline, label_names):
        sage = all_results[f"{case_key_prefix}_GraphSAGE"]
        gcn = all_results[f"{case_key_prefix}_GCN"]

        _, _, sage_f1, _ = precision_recall_fscore_support(
            sage["y_true"], sage["y_pred"], average="macro", zero_division=0
        )
        _, _, gcn_f1, _ = precision_recall_fscore_support(
            gcn["y_true"], gcn["y_pred"], average="macro", zero_division=0
        )

        return {
            "baseline": {
                "accuracy": to_float(baseline["acc"]),
                "precision": to_float(baseline["prec"]),
                "recall": to_float(baseline["rec"]),
                "f1_macro": to_float(baseline["f1"]),
                "confusion_matrix": confusion_matrix(
                    baseline["y_true"], baseline["y_pred"]
                ).tolist(),
                "classification_report": serialize_report(
                    baseline["y_true"], baseline["y_pred"], label_names
                ),
            },
            "graphsage": {
                "accuracy": to_float(sage["best_acc"]),
                "f1_macro": to_float(sage_f1),
                "epochs_ran": len(sage["history"]["train_loss"]),
                "best_test_accuracy": to_float(max(sage["history"]["test_acc"])),
                "last_test_accuracy": to_float(sage["history"]["test_acc"][-1]),
                "confusion_matrix": confusion_matrix(
                    sage["y_true"], sage["y_pred"]
                ).tolist(),
                "classification_report": serialize_report(
                    sage["y_true"], sage["y_pred"], label_names
                ),
            },
            "gcn": {
                "accuracy": to_float(gcn["best_acc"]),
                "f1_macro": to_float(gcn_f1),
                "epochs_ran": len(gcn["history"]["train_loss"]),
                "best_test_accuracy": to_float(max(gcn["history"]["test_acc"])),
                "last_test_accuracy": to_float(gcn["history"]["test_acc"][-1]),
                "confusion_matrix": confusion_matrix(
                    gcn["y_true"], gcn["y_pred"]
                ).tolist(),
                "classification_report": serialize_report(
                    gcn["y_true"], gcn["y_pred"], label_names
                ),
            },
        }

    return {
        "caso_1_disaster_tweets": bundle_case("C1", baseline_c1, label_names_c1),
        "caso_2_ag_news": bundle_case("C2", baseline_c2, label_names_c2),
    }


def main():
    os.chdir(ROOT)
    cells = load_notebook_cells(NOTEBOOK_PATH)
    namespace = {"__name__": "__main__"}

    setup_cells = [4, 6, 8, 10, 12, 16, 18, 20, 22]
    plot_cells = {
        24: "curvas_entrenamiento",
        26: "matrices_confusion",
        28: "proyeccion_tsne",
        32: "comparativa_global",
    }

    for idx in setup_cells:
        print(f"\n[EXEC] celda {idx}")
        exec("".join(cells[idx]["source"]), namespace)

    namespace["display"] = safe_display
    namespace["clear_output"] = lambda *args, **kwargs: None

    for idx, prefix in plot_cells.items():
        print(f"\n[EXEC] celda {idx} -> {prefix}")
        patch_matplotlib(namespace, prefix)
        exec("".join(cells[idx]["source"]), namespace)

    print("\n[EXEC] celda 36 -> tabla de resultados")
    exec("".join(cells[36]["source"]), namespace)

    metrics = collect_metrics(namespace)
    metrics_path = OUTPUT_DIR / "metricas_resumen.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[METRICAS] {metrics_path}")

    if "df_results" in namespace:
        csv_path = OUTPUT_DIR / "tabla_resultados.csv"
        namespace["df_results"].to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[TABLA] {csv_path}")


if __name__ == "__main__":
    main()
