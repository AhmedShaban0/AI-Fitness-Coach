import os
import sys
import io
import glob
import ast

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Ensure Windows terminal handles UTF-8 emojis cleanly
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def run_self_test():
    """
    Automated pre-flight diagnostic suite.
    Verifies imports, dataset integrity, component APIs, metrics accessors, and CSS assets.
    """
    print("==================================================")
    print("🚀 RUNNING NTI PLATFORM SYSTEM SELF-TEST")
    print("==================================================")
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    def log_result(name, success, message=""):
        if success:
            results["passed"] += 1
            print(f"  ✓ [{name}] {message}")
        else:
            results["failed"] += 1
            results["errors"].append(f"[{name}] {message}")
            print(f"  ❌ [{name}] {message}")

    # 1. AST Syntax & Import Audit
    try:
        py_files = glob.glob("dashboard/**/*.py", recursive=True)
        for f in py_files:
            content = open(f, encoding="utf-8").read()
            ast.parse(content, filename=f)
        log_result("Python AST Syntax", True, f"Validated {len(py_files)} files cleanly.")
    except Exception as e:
        log_result("Python AST Syntax", False, str(e))

    # 2. Asset Verification
    css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "custom.css")
    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "nti_logo.png")
    
    log_result("CSS Custom Styles", os.path.exists(css_path), f"custom.css exists: {os.path.exists(css_path)}")
    log_result("NTI Logo Asset", os.path.exists(logo_path), f"nti_logo.png exists: {os.path.exists(logo_path)}")

    # 3. Core Component API Verification
    try:
        from dashboard.components.cards import (
            render_hero_banner, render_header, render_glass_kpi_card, render_metric_card,
            render_feature_card, render_model_card, render_sidebar_header, render_section_title, render_info_card
        )
        log_result("Cards Component API", True, "Exposes stable component interfaces.")
    except Exception as e:
        log_result("Cards Component API", False, str(e))

    try:
        from dashboard.components.charts import (
            plot_correlation_heatmap, plot_histogram, plot_boxplot, plot_bar, plot_scatter,
            plot_actual_vs_predicted, plot_residuals, plot_feature_importance, plot_gauge_meter, render_business_insight
        )
        log_result("Charts Component API", True, "Exposes stable Plotly charting interfaces.")
    except Exception as e:
        log_result("Charts Component API", False, str(e))

    # 4. Data Ingestion & Preprocessing Verification
    try:
        from dashboard.utils.data_loader import load_and_process_data
        fn_data = getattr(load_and_process_data, "__wrapped__", load_and_process_data)
        df = fn_data()
        log_result("Dataset Ingestion", len(df) > 0, f"Successfully loaded dataset with {len(df):,} records and {len(df.columns)} columns.")
    except Exception as e:
        log_result("Dataset Ingestion", False, str(e))

    # 5. Defensive Metrics Accessor Verification
    try:
        from dashboard.utils.preprocessing import safe_get_metric
        test_metrics = {"test_r2": 0.995, "train_mae": 1.2}
        r2_val = safe_get_metric(test_metrics, "r2")
        missing_val = safe_get_metric(test_metrics, "unknown_key", default="N/A")
        
        valid = (r2_val == 0.995) and (missing_val == "N/A")
        log_result("Defensive Metric Accessor", valid, f"safe_get_metric returned r2={r2_val}, fallback={missing_val}")
    except Exception as e:
        log_result("Defensive Metric Accessor", False, str(e))

    # 6. Model Training & Dynamic Metrics Evaluation
    try:
        from dashboard.models.train import train_all_models
        fn_train = getattr(train_all_models, "__wrapped__", train_all_models)
        res = fn_train(df, sample_size=5000)
        best_b = res["calories_burned"]["best_model_name"]
        best_r = res["required_calories"]["best_model_name"]
        log_result("Model Benchmark Engine", True, f"Dynamic winner (burned): {best_b}, (required): {best_r}")
    except Exception as e:
        log_result("Model Benchmark Engine", False, str(e))

    print("==================================================")
    print(f"📊 SUMMARY: {results['passed']} Passed | {results['failed']} Failed")
    print("==================================================")
    
    return results["failed"] == 0

if __name__ == "__main__":
    success = run_self_test()
    sys.exit(0 if success else 1)
