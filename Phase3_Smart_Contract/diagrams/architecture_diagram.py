# final_architecture_optimized.py
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python, Bash
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.network import Nginx
from diagrams.onprem.compute import Server
from diagrams.generic.os import Ubuntu
from diagrams.generic.storage import Storage
from diagrams.custom import Custom
import os

# Custom Icons Configuration
ICON_DIR = r"D:\Uni-Dex-Marketplace_test\Phase3_Smart_Contract\diagrams\icons"


def load_custom_icon(icon_file, default_icon, label):
    icon_path = os.path.join(ICON_DIR, icon_file)
    return (
        Custom(label, icon_path) if os.path.exists(icon_path) else default_icon(label)
    )


# Define custom icons
binance_icon = lambda l: load_custom_icon("binance.png", Server, l)
gru_icon = lambda l: load_custom_icon("gru.png", Python, l)
lstm_icon = lambda l: load_custom_icon("lstm.png", Python, l)
model_icon = lambda l: load_custom_icon("model.png", Python, l)
chart_icon = lambda l: load_custom_icon("chart.png", Server, l)
uniswap_icon = lambda l: load_custom_icon("uniswap.png", Server, l)
contract_icon = lambda l: load_custom_icon("contract.png", Server, l)
fork_icon = lambda l: load_custom_icon("fork.png", Server, l)
csv_icon = lambda l: load_custom_icon("csv_icon.png", Storage, l)

with Diagram(
    "Figure 3-1: Project Architecture (4-Phase Workflow)",
    show=False,
    direction="LR",  # Left-to-right flow
    filename="system_architecture_final",
    graph_attr={
        "splines": "ortho",
        "fontsize": "11",
        "fontname": "Arial",
        "ranksep": "0.6",
        "nodesep": "0.5",
        "pad": "0.2",
    },
):
    # Phase 1: Data Collection
    with Cluster(
        "Phase 1: Data Collection",
        graph_attr={"bgcolor": "#E8F0FE", "style": "rounded"},
    ):
        data_source = binance_icon("Historical Data\n(Binance API)")
        phase1_output = data_source

    # Phase 2: Model Development
    with Cluster(
        "Phase 2: Model Training", graph_attr={"bgcolor": "#E6F4EA", "style": "rounded"}
    ):
        feature_eng = Python("Feature Engineering\n(4 new columns)")

        with Cluster(
            "Model Comparison", graph_attr={"bgcolor": "white", "style": "dashed"}
        ):
            gru_model = gru_icon("GRU Model")
            lstm_model = lstm_icon("LSTM Model")
            evaluation = chart_icon("Evaluation\n(RMSE/MAE)")

        best_model = model_icon("Optimized Model\n(GRU.h5)")

        feature_eng >> [gru_model, lstm_model] >> evaluation >> best_model
        phase2_output = best_model

    # Phase 3: API Deployment
    with Cluster(
        "Phase 3: API Deployment", graph_attr={"bgcolor": "#FFF3E0", "style": "rounded"}
    ):
        vps = Ubuntu("VPS Server")
        api = Nginx("CryptoPredictAPI")

        vps - Edge(style="invis") - api
        phase3_output = api

    # Phase 4: Simulation
    with Cluster(
        "Phase 4: Blockchain Simulation",
        graph_attr={"bgcolor": "#FCE4EC", "style": "rounded"},
    ):
        orchestrator = Bash("Orchestrator Script\n(run_tests.sh)")

        tester = Python("Test Runner")
        results = csv_icon("Results\n(CSV)")

        with Cluster("Hardhat Environment", graph_attr={"bgcolor": "#E1F5FE"}):
            fork = fork_icon("Mainnet Fork")
            pred_contract = contract_icon("Predictive\nContract")
            base_contract = contract_icon("Baseline\nContract")
            uniswap = uniswap_icon("Uniswap V3\nProtocol")

            [pred_contract, base_contract] >> Edge(style="dashed") >> uniswap
            fork - Edge(style="invis") - uniswap

        orchestrator >> tester
        tester >> Edge(style="dashed") >> results

    # Phase connections
    phase1_output >> feature_eng
    phase2_output >> Edge(style="dotted") >> api

    # Key interactions
    tester >> Edge(label="API Call", color="blue") >> api
    tester >> Edge(label="Send Tx", color="darkgreen") >> pred_contract
