import os
from fastapi import FastAPI
import uvicorn
from deployment import app  # Adjust based on your FastAPI app file

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
