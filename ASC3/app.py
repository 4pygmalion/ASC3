import os
import sys
from contextlib import asynccontextmanager

from starlette.types import Message
from omegaconf import OmegaConf
from fastapi import FastAPI, Request, Depends


ASC3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ASC3_DIR)
sys.path.append(ROOT_DIR)

from ASC3.tree_model.router import api_router
from ASC3.tree_model.model import Classifier
from ASC3.mil_model.router import mil_router
from ASC3.mil_model.model import MILPredictor, EnsembleMILPredictor

from ASC3.error_handler import add_exception_handlers
from utils.log_ops import get_logger


MODEL_NAME = os.environ.get("MODEL_NAME")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = OmegaConf.load(os.path.join(ASC3_DIR, "config.yaml"))

    logger = get_logger("router")

    if MODEL_NAME == "tree":
        classifier = Classifier(config=config, logger=logger, delimiter=".")
        classifier.set_model()

        app.state.classifier: Classifier = classifier
        app.state.logger = logger

        yield

        app.state.classifier = None
        app.state.logger = None

        del app.state.classifier
        del app.state.logger

    elif MODEL_NAME == "mil":
        app.state.mil_predictor = MILPredictor(config=config, logger=logger)
        app.state.logger = logger

        yield

        app.state.mil_predictor = None
        app.state.logger = None

        del app.state.mil_predictor
        del app.state.logger

    elif MODEL_NAME == "ensemble":
        app.state.mil_predictor = EnsembleMILPredictor(config=config, logger=logger)
        app.state.logger = logger

        yield

        app.state.mil_predictor = None
        app.state.logger = None

        del app.state.mil_predictor
        del app.state.logger


app = FastAPI(lifespan=lifespan)
if MODEL_NAME == "tree":
    app.include_router(api_router)

elif MODEL_NAME == "mil":
    app.include_router(mil_router)

elif MODEL_NAME == "ensemble":
    app.include_router(mil_router)

else:
    raise NotImplementedError()


async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


@app.middleware("http")
async def log_request_payload(request: Request, call_next):
    payload = await request.body()
    await set_body(request, payload)
    request.app.state.logger.debug(f"Request payload: {payload.decode()}")
    response = await call_next(request)
    return response


add_exception_handlers(app)
