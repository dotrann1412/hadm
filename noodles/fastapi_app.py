'''
dependencies to install for fastapi app:
---
git+https://github.com/dotrann1412/conveyor@4237d6ad9ddc52c6451dfdc0817ad828833bfbb5 \
fastapi[standard]==0.136.1 \
uvicorn==0.46.0
---
'''

from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvalReq(BaseModel):
    image_url: str = Field(..., description="The URL of the image to evaluate")

class EvalInterResponse(EvalReq):
    image: Image.Image | None = None
    image_tensor: torch.Tensor | None = None
    l_predictions: dict | None = Field(default=None, description="The l predictions from the model")
    g_predictions: dict | None = Field(default=None, description="The g predictions from the model")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class EvalFinalResponse(EvalReq):
    qualified: bool = Field(default=False, description="Whether the image is qualified or not")
    predictions: list[dict] = Field(default_factory=list, description="The predictions from the model")
    explanation: str = Field(default="", description="The explanation for the decision")
    model_config = ConfigDict(arbitrary_types_allowed=True)

import httpx
from io import BytesIO
import base64

from hadm import HADM, load_hadm_weights, get_class_names
from torchvision.transforms.functional import to_tensor

import re

async def pre_processing_stage(req: EvalReq) -> EvalInterResponse:
    b64_pat = re.compile(r"data:image/(png|jpeg|jpg);base64,(.*)", re.IGNORECASE)
    resp = EvalInterResponse(image_url=req.image_url)

    if req.image_url.startswith("http"):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(req.image_url)
            response.raise_for_status()
            resp.image = Image.open(BytesIO(response.content))
            resp.image_tensor = to_tensor(resp.image)

    elif match := b64_pat.match(req.image_url):
        resp.image = Image.open(BytesIO(base64.b64decode(match.group(2))))
        resp.image_tensor = to_tensor(resp.image)

    else:
        logger.warning(f"Invalid image URL: {req.image_url}")
        raise ValueError(f"Invalid image URL: {req.image_url}")

    return resp


def make_infer_stage(device: str):
    model_l = HADM(mode="local")
    load_hadm_weights(model_l, "HADM-L_0249999.pth", use_ema=True)
    model_l.to(device).eval()

    model_g = HADM(mode="global")
    load_hadm_weights(model_g, "HADM-G_0249999.pth", use_ema=True)
    model_g.to(device).eval()

    def batch_detect(reqs: list[EvalInterResponse]) -> list[EvalInterResponse]:
        masks = [req.image_tensor is not None for req in reqs]
        valid_inputs = [req.image_tensor for req in reqs if req.image_tensor is not None]

        if not valid_inputs:
            return reqs

        lresults = iter(model_l(valid_inputs, score_thresh=0.3))
        gresults = iter(model_g(valid_inputs, score_thresh=0.3))

        for i, (mask, _) in enumerate(zip(masks, reqs)):
            if mask:
                lr, gr = next(lresults), next(gresults)

                if lr['boxes'].numel() > 0:
                    reqs[i].l_predictions = lr

                if gr['boxes'].numel() > 0:
                    reqs[i].g_predictions = gr

        return reqs
    
    return batch_detect

def post_processing_stage(req: EvalInterResponse) -> EvalFinalResponse:

    predictions = []

    w, h = req.image.size # type: ignore

    l_names = get_class_names("local")
    g_names = get_class_names("global")

    pad = 10

    if req.l_predictions is not None:
        for i in range(req.l_predictions['boxes'].shape[0]):
            lbl = req.l_predictions['labels'][i].item()
            cls = l_names[lbl - 1] if 1 <= lbl <= len(l_names) else f"cls_{lbl}"
            score = round(req.l_predictions['scores'][i].item(), 2)

            x1, y1, x2, y2 = [round(v, 0) for v in req.l_predictions['boxes'][i].tolist()]
            _x1, _y1 = max(pad, x1), max(pad, y1)
            _x2, _y2 = min(w - pad, x2), min(h - pad, y2)

            if (_x2 - _x1) * (_y2 - _y1) < 0.05 * w * h:
                continue

            predictions.append({
                "xyxy": [_x1, _y1, _x2, _y2],
                "score": score,
                "class": cls
            })


    if req.g_predictions is not None:
        for i in range(req.g_predictions['boxes'].shape[0]):
            lbl = req.g_predictions['labels'][i].item()
            cls = g_names[lbl - 1] if 1 <= lbl <= len(g_names) else f"cls_{lbl}"
            score = round(req.g_predictions['scores'][i].item(), 2)

            x1, y1, x2, y2 = [round(v, 0) for v in req.g_predictions['boxes'][i].tolist()]
            _x1, _y1 = max(pad, x1), max(pad, y1)
            _x2, _y2 = min(w - pad, x2), min(h - pad, y2)

            if (_x2 - _x1) * (_y2 - _y1) < 0.05 * w * h:
                continue

            predictions.append({
                "xyxy": [_x1, _y1, _x2, _y2],
                "score": score,
                "class": cls
            })

    qualified = len(predictions) == 0
    explanation = "No anomalies detected" if qualified else "Anomalies detected"

    return EvalFinalResponse(
        qualified=qualified, 
        explanation=explanation, 
        predictions=predictions,
        image_url=req.image_url
    )

from conveyor import Pipeline, Stage, BatchStage
from fastapi import Request
from fastapi import Depends
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.eval_pipeline_1 = Pipeline(
        stages=[
            Stage(fns=[pre_processing_stage], queue_size_per_worker=10),
            BatchStage(
                fns=[make_infer_stage('cuda:0' if torch.cuda.is_available() else 'cpu')], 
                worker_queue_size=10, 
                max_batch_size=8, 
                timeout_s=1
            ),
            Stage(fns=[post_processing_stage], queue_size_per_worker=10),
        ]
    )

    async with app.state.eval_pipeline_1:
        yield

def depends_eval_pipeline_1(request: Request) -> Pipeline:
    return request.app.state.eval_pipeline_1

app = FastAPI(lifespan=lifespan)

@app.post("/evaluate")
async def evaluate(req: EvalReq, pipeline: Pipeline = Depends(depends_eval_pipeline_1)) -> EvalFinalResponse:
    return await pipeline.submit(req)

'''
example usage:
---
curl -X 'POST' \
  'http://localhost:9090/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_url": "https://example.com/image.jpg"
}'
---


response:
---
{
  "image_url": "https://example.com/image.jpg",
  "qualified": false,
  "predictions": [
    {
      "xyxy": [
        131,
        265,
        225,
        466
      ],
      "score": 0.44,
      "class": "hand"
    }
  ],
  "explanation": "Anomalies detected"
}
---
'''