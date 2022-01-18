from typing import Optional
from fastapi import FastAPI, Response, status, HTTPException, Request, File, UploadFile, Form
from fastapi.params import Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import psycopg2
from psycopg2.extras import RealDictCursor

from pydantic import BaseModel

from random import randrange

from PIL import Image
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class Coordinates(BaseModel):
    NW: float
    NE: float
    SE: float
    NE: float


def get_sat_img():
    pass


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/upload_drone_img")
async def handle_img(drone_img: UploadFile = File(...)):
    content_drone_img = await drone_img.read()
    print(type(content_drone_img))
    img = Image.open(io.BytesIO(content_drone_img))
    img.save('test.png')


@app.post("/upload_coordinates")
async def handle_coordinates(NW: float = Form(...), NE: float = Form(...), SE: float = Form(...), SW: float = Form(...)):
    coordinates = {}
    coordinates['NW'] = NW
    coordinates['NE'] = NE
    coordinates['SE'] = SE
    coordinates['SW'] = SW
    print(coordinates)
