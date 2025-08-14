from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from text_routes import register_text_routes
from image_routes import register_image_routes
from audio_routes import register_audio_routes


app = FastAPI(title="PrivaSee API")
app.mount("/static", StaticFiles(directory="static"), name="static")


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _init_state():
    app.state.analyzer_cache = {}
    app.state.uncert_cache = {}


@app.get("/")
def read_root():
    return {"message": "PrivaSee API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")


# Register modular routes
register_text_routes(app)
register_image_routes(app)
register_audio_routes(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


