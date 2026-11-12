from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.rag_embeddings import LocalEmbeddings, RAGError
from app.rag_routes import router as rag_router
from app.rag_service import RAGService
from app.rag_store import VectorStore
from app.research_graph import GraphResearchService
from app.research_routes import router as research_router
from app.research_service import ResearchError, ResearchService
from app.sec_client import SECClient, SECError
from app.sec_routes import router
from app.sec_service import SECService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = SECClient(settings)
        app.state.sec_service = SECService(client)
        app.state.rag_service = RAGService(
            app.state.sec_service,
            LocalEmbeddings(settings),
            VectorStore(settings),
            settings,
        )
        app.state.research_service = ResearchService(
            app.state.sec_service, app.state.rag_service, settings
        )
        app.state.graph_research_service = GraphResearchService(
            app.state.sec_service, app.state.rag_service, settings
        )
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
    app.include_router(rag_router)
    app.include_router(research_router)

    @app.exception_handler(ResearchError)
    async def research_error_handler(
        request: Request, exc: ResearchError
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(SECError)
    async def sec_error_handler(request: Request, exc: SECError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
