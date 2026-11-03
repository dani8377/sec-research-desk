from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.sec_client import SECClient, SECError
from app.sec_routes import router
from app.sec_service import SECService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = SECClient(settings)
        app.state.sec_service = SECService(client)
        try:
            yield
        finally:
            client.close()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
        # Full Company Facts responses can contain megabytes of JSON.
        swagger_ui_parameters={"syntaxHighlight": False},
    )
    app.include_router(router)

    @app.exception_handler(SECError)
    async def sec_error_handler(request: Request, exc: SECError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
