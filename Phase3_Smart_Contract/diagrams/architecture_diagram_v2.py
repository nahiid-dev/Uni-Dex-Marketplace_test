# architecture_diagram_final_version.py
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python, Bash
from diagrams.onprem.network import Nginx
from diagrams.onprem.compute import Server
from diagrams.generic.storage import Storage
from diagrams.custom import Custom
import os

# ==========================================================
# **تنظیمات جدید سایز**
# ==========================================================
# اندازه جدید و کوچک برای اکثر آیکن‌ها
SMALL_SIZE = "0.8"
# اندازه‌ای که می‌خواهیم برای GRU و LSTM حفظ شود
LARGE_SIZE = "1.3"
# اندازه فونت
TEXT_FONT_SIZE = "9"
# ==========================================================


# تنظیمات مسیر آیکون‌ها
ICON_DIR = r"D:\Uni-Dex-Marketplace_test\Phase3_Smart_Contract\diagrams\icons"


def load_custom_icon(icon_file, default_icon, label, size):
    """یک تابع ساده‌شده برای ساخت آیکن سفارشی با سایز مشخص"""
    icon_path = os.path.join(ICON_DIR, icon_file)
    node = (
        Custom(label, icon_path) if os.path.exists(icon_path) else default_icon(label)
    )
    # تنظیمات سایز و فونت برای هر آیکن
    node.width = size
    node.height = size
    node.fontsize = TEXT_FONT_SIZE
    return node


with Diagram(
    "Figure 3-1: Project Architecture (Final Layout)",
    show=False,
    direction="LR",
    filename="system_architecture_final",
    graph_attr={
        "splines": "ortho",
        "fontsize": "12",
        "fontname": "Arial",
        "ranksep": "0.5",
        "nodesep": "0.4",
        "pad": "0.3",
        "compound": "true",
    },
    # **تغییر کلیدی: حذف width و height سراسری از اینجا**
    node_attr={
        "fixedsize": "true",  # این مهم است که بماند
        "margin": "0.1",
        "fontname": "Arial",
    },
    edge_attr={"fontsize": "9", "minlen": "1.5"},
):
    # Phase 1: Data Collection
    with Cluster(
        "Phase 1: Data Collection",
        graph_attr={"bgcolor": "#E8F0FE", "style": "rounded"},
    ):
        data_source = load_custom_icon(
            "binance.png", Server, "Historical Data\n(Binance API)", size=SMALL_SIZE
        )

    # Phase 2: Model Training
    with Cluster(
        "Phase 2: Model Training", graph_attr={"bgcolor": "#E6F4EA", "style": "rounded"}
    ):
        # تنظیم سایز جداگانه برای آیکن‌های استاندارد
        feature_eng = Python("Feature Engineering\n(4 new columns)")
        feature_eng.width = SMALL_SIZE
        feature_eng.height = SMALL_SIZE
        feature_eng.fontsize = TEXT_FONT_SIZE

        with Cluster(
            "Model Comparison", graph_attr={"bgcolor": "white", "style": "dashed"}
        ):
            # GRU و LSTM با سایز بزرگتر
            gru_model = load_custom_icon("gru.png", Python, "GRU", size=LARGE_SIZE)
            lstm_model = load_custom_icon("lstm.png", Python, "LSTM", size=LARGE_SIZE)

        # بقیه آیکن‌ها با سایز کوچک
        evaluation = load_custom_icon(
            "chart.png", Server, "Evaluation\n(RMSE/MAE)", size=SMALL_SIZE
        )
        best_model = load_custom_icon(
            "model.png", Python, "Optimized Model\n(GRU.h5)", size=SMALL_SIZE
        )

        feature_eng >> [gru_model, lstm_model] >> evaluation >> best_model

    # Phase 3: VPS Deployment
    with Cluster(
        "VPS Deployment", graph_attr={"bgcolor": "#FFF3E0", "style": "rounded"}
    ):
        # تنظیم سایز جداگانه برای آیکن‌های استاندارد
        api = Nginx("CryptoPredictAPI")
        api.width = SMALL_SIZE
        api.height = SMALL_SIZE
        api.fontsize = TEXT_FONT_SIZE

        with Cluster("Phase 4: Simulation", graph_attr={"bgcolor": "#FCE4EC"}):
            orchestrator = Bash("Orchestrator")
            orchestrator.width = SMALL_SIZE
            orchestrator.height = SMALL_SIZE
            orchestrator.fontsize = TEXT_FONT_SIZE

            tester = Python("Test Runner")
            tester.width = SMALL_SIZE
            tester.height = SMALL_SIZE
            tester.fontsize = TEXT_FONT_SIZE

            results = load_custom_icon("csv.png", Storage, "Results", size=SMALL_SIZE)

            with Cluster("Hardhat", graph_attr={"bgcolor": "#E1F5FE"}):
                pred_contract = load_custom_icon(
                    "contract.png", Server, "Predictive\nContract", size=SMALL_SIZE
                )
                base_contract = load_custom_icon(
                    "contract.png", Server, "Baseline\nContract", size=SMALL_SIZE
                )
                uniswap = load_custom_icon(
                    "uniswap.png", Server, "Uniswap V3", size=SMALL_SIZE
                )
                ([pred_contract, base_contract] >> Edge(style="dashed") >> uniswap)

            # تعریف جریان‌ها
            orchestrator >> tester
            tester >> Edge(style="dashed") >> results
            tester >> Edge(label="Tx", color="darkgreen") >> pred_contract
            tester >> Edge(label="API Call", color="blue") >> api

    # اتصالات اصلی بین فازها
    data_source >> feature_eng
    best_model >> Edge(style="dotted", label="Load") >> api
