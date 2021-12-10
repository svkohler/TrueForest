from typing import Optional
from fastapi import FastAPI, Response, status, HTTPException, Request
from fastapi.params import Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import psycopg2
from psycopg2.extras import RealDictCursor

from pydantic import BaseModel

from random import randrange

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})
