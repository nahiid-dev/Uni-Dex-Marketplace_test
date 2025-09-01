import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# لطفا فایل predictive_final.csv را در کنار این اسکریپت قرار دهید
CSV_FILE_PATH = "predictive_final.csv"

try:
    df = pd.read_csv(CSV_FILE_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("فایل CSV با موفقیت بارگذاری شد.")

    # --- گام ۱: تصحیح زمانی و آماده‌سازی داده‌ها ---
    df["prediction"] = df["predictedPrice_api"].shift(1)
    df["actual_pool"] = df["actualPrice_pool"]
    df["actual_external"] = df["external_api_eth_price"]
    df_aligned = df[["prediction", "actual_pool", "actual_external"]].iloc[1:].copy()

    # --- گام ۲: محاسبه معیارها برای هر دو مقایسه ---
    metrics = {}
    comparisons = {
        "vs_pool": ("prediction", "actual_pool"),
        "vs_external": ("prediction", "actual_external"),
    }
    for comp_name, (pred_col, actual_col) in comparisons.items():
        errors = df_aligned[pred_col] - df_aligned[actual_col]
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(np.abs(errors / df_aligned[actual_col])) * 100
        actual_change = df_aligned[actual_col].diff()
        predicted_change = df_aligned[pred_col] - df_aligned[actual_col].shift(1)
        correct_direction = np.sign(actual_change) == np.sign(predicted_change)
        directional_accuracy = correct_direction.mean() * 100
        metrics[comp_name] = {
            "MAE (USD)": mae,
            "RMSE (USD)": rmse,
            "MAPE (%)": mape,
            "Directional Accuracy (%)": directional_accuracy,
        }

    # --- گام ۳: ساخت دیتافریم نهایی برای رسم نمودار و جدول ---
    plot_list = []
    for metric_name in metrics["vs_pool"].keys():
        plot_list.append(
            {
                "Metric": metric_name,
                "Comparison": "Prediction vs. Pool Price",
                "Value": metrics["vs_pool"][metric_name],
            }
        )
        plot_list.append(
            {
                "Metric": metric_name,
                "Comparison": "Prediction vs. External Price",
                "Value": metrics["vs_external"][metric_name],
            }
        )
    df_plot = pd.DataFrame(plot_list)

    # --- گام ۴: رسم نمودار ۲x۲ با میله‌های گروه‌بندی شده ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparative Model Accuracy Analysis", fontsize=20, weight="bold")
    axes = axes.flatten()
    metric_names = ["MAE (USD)", "RMSE (USD)", "MAPE (%)", "Directional Accuracy (%)"]
    palette = {
        "Prediction vs. Pool Price": "royalblue",
        "Prediction vs. External Price": "darkorange",
    }
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        subset_data = df_plot[df_plot["Metric"] == metric_name]
        barplot = sns.barplot(
            data=subset_data,
            x="Metric",
            y="Value",
            hue="Comparison",
            palette=palette,
            ax=ax,
            width=0.6,
        )
        ax.set_title(metric_name, fontsize=14, weight="bold")
        ax.set_ylabel("Value")
        ax.set_xlabel("")
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.legend().set_title("")
        for p in barplot.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="top",  # اصلاح شد: چینش از بالا
                xytext=(0, -8),  # اصلاح شد: انتقال متن به داخل میله
                textcoords="offset points",
                fontsize=11,
                weight="bold",
                color="white",  # اصلاح شد: رنگ سفید برای خوانایی بهتر
            )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("model_accuracy_comparison_2x2.png", dpi=300)
    print("نمودار 'model_accuracy_comparison_2x2.png' با موفقیت ذخیره شد.")
    plt.close(fig)  # بستن شکل نمودار

    # --- گام ۵: ایجاد و ذخیره تصویر جدول نتایج ---
    print("در حال ایجاد تصویر جدول نتایج...")

    # تبدیل داده‌ها از فرمت طولانی به فرمت عریض برای جدول
    df_table = df_plot.pivot(index="Metric", columns="Comparison", values="Value")
    df_table = df_table.round(2)  # گرد کردن اعداد
    df_table.reset_index(inplace=True)  # بازگرداندن ایندکس به ستون

    fig_table, ax_table = plt.subplots(figsize=(10, 4))  # اندازه مناسب برای جدول
    ax_table.set_title(
        "Model Accuracy Metrics (Time-Shift Corrected)", weight="bold", size=16, pad=20
    )
    ax_table.axis("tight")
    ax_table.axis("off")

    # ایجاد جدول با استفاده از matplotlib
    the_table = ax_table.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(df_table.columns),  # رنگ خاکستری برای هدر
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.1, 2)  # تنظیم اندازه سلول‌ها

    # ذخیره جدول به عنوان یک فایل تصویری
    plt.savefig("model_accuracy_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table)

    print("تصویر جدول 'model_accuracy_table.png' با موفقیت ذخیره شد.")

except FileNotFoundError:
    print(f"خطا: فایل {CSV_FILE_PATH} پیدا نشد.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")
