from typing import Optional
from fastapi import FastAPI, Response, status, HTTPException, Request, File, UploadFile, Form
from fastapi.params import Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import time

from utils import get_sat_image


import psycopg2
from psycopg2.extras import RealDictCursor

from pydantic import BaseModel

from random import randrange

from PIL import Image
import io
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class Coordinates(BaseModel):
    bottom_left_long: float
    bottom_left_lat: float
    top_right_long: float
    top_right_lat: float


class Result(BaseModel):
    result: float


PATH = os.getcwd()+'/static/images/'


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/upload_drone_img")
async def handle_img(drone_img: UploadFile = File(...)):
    content_drone_img = await drone_img.read()
    img = Image.open(io.BytesIO(content_drone_img))
    img = img.resize((224, 224))
    img.save('./static/images/drone.png')


@app.post("/upload_coordinates")
async def handle_coordinates(bottom_left_long: float = Form(...), bottom_left_lat: float = Form(...), top_right_long: float = Form(...), top_right_lat: float = Form(...)):
    coordinates = {}
    coordinates['bottom_left_long'] = bottom_left_long
    coordinates['bottom_left_lat'] = bottom_left_lat
    coordinates['top_right_long'] = top_right_long
    coordinates['top_right_lat'] = top_right_lat
    get_sat_image(coordinates, PATH)


def val():
    time.sleep(3)
    res = random.randint(0, 1)
    result = {
        "result": res
    }
    return result


@app.get("/validate")
async def validate():
    result = val()
    return result
