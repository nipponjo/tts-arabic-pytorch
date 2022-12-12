import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.app_utils import TTSManager

app = FastAPI()

tts_manager = TTSManager('app/static')

class TTSRequest(BaseModel):
    buckw: str
    rate: float
    denoise: float

app.mount('/static', StaticFiles(directory='./app/static'), 'static')


@app.get('/')
async def main():
    return FileResponse('./app/index.html')


@app.get('/{filename}')
async def get_file(filename: str):
    filepath = f'./app/{filename}'
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return Response(status_code=404)


@app.post('/api/tts')
async def tts(req: TTSRequest):
    print(req)
    response_data = tts_manager.tts(req.buckw, req.rate, 
                                    req.denoise)

    return response_data


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
