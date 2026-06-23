import uvicorn
import argparse
from pathlib import Path
from src.api.app import create_app

app = create_app(
    model_path=Path("models/saved/best_model.pkl"),
    preprocessor_path=Path("models/saved/preprocessor.pkl"),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn prediction API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.run:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
