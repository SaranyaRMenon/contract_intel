"""Local entrypoint for the FastAPI server."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

from contract_intel.api import app  # noqa: F401  (imported for uvicorn)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("contract_intel.api:app", host="0.0.0.0", port=port, reload=False)
