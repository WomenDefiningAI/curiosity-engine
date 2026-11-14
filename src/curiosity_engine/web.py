from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import repository_root
from .db import connect
from .resources import discover_private_catalogs
from .service import CuriosityService


def create_app(db_path: str | Path | None = None, output_dir: str | Path | None = None) -> FastAPI:
    root = repository_root()
    database = Path(db_path or os.environ.get("CURIOSITY_DB") or root / "private" / "data" / "curiosity.db")
    output = Path(output_dir or os.environ.get("CURIOSITY_OUTPUT") or root / "private" / "output")
    service = CuriosityService(database, output)
    package = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=package / "templates")
    app = FastAPI(title="Curiosity Engine", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.service = service
    app.state.csrf = secrets.token_urlsafe(32)
    app.mount("/static", StaticFiles(directory=package / "static"), name="static")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        client = request.client.host if request.client else ""
        host = request.url.hostname or ""
        if client not in {"127.0.0.1", "::1", "localhost", "testclient"} or host not in {
            "127.0.0.1",
            "::1",
            "localhost",
            "testserver",
        }:
            return JSONResponse({"detail": "Curiosity Engine is local-only"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; img-src 'self' data:"
        return response

    def check_csrf(token: str) -> None:
        if not secrets.compare_digest(token, app.state.csrf):
            raise HTTPException(403, "invalid form token")

    def render_home(request: Request, *, child: str | None = None, notice: str | None = None, error: str | None = None):
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "csrf": app.state.csrf,
                "state": service.dashboard(child),
                "notice": notice,
                "error": error,
            },
        )

    def render_evals(request: Request, *, notice: str | None = None, error: str | None = None):
        return templates.TemplateResponse(
            request,
            "evals.html",
            {
                "csrf": app.state.csrf,
                "items": service.evaluation_queue(limit=10),
                "notice": notice,
                "error": error,
            },
        )

    def redirect_home(child_id: str | None = None, *, anchor: str | None = None) -> RedirectResponse:
        """Redirect only to a fixed local path with a database-owned child identifier."""

        known_child = next((child["id"] for child in service.children() if child["id"] == child_id), None)
        destination = "/"
        if known_child:
            destination += "?" + urlencode({"child": known_child})
        if anchor:
            destination += f"#{anchor}"
        return RedirectResponse(destination, status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, child: str | None = None):
        return render_home(request, child=child)

    @app.get("/evals", response_class=HTMLResponse)
    def evals(request: Request):
        return render_evals(request)

    @app.get("/evals/visual/{visual_asset_id}")
    def eval_visual(visual_asset_id: str):
        with connect(service.db_path) as conn:
            row = conn.execute(
                "SELECT path,mime_type,filename FROM visual_assets WHERE id=?",
                (visual_asset_id,),
            ).fetchone()
        if not row:
            raise HTTPException(404)
        path = Path(row["path"]).resolve()
        try:
            path.relative_to(service.output_dir)
        except ValueError as exc:
            raise HTTPException(403) from exc
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type=row["mime_type"], filename=row["filename"])

    @app.post("/evals/generate-activity-aids", response_class=HTMLResponse)
    def generate_eval_activity_aids(request: Request, csrf: Annotated[str, Form()] = ""):
        check_csrf(csrf)
        result = service.generate_evaluation_activity_aids(limit=10)
        notice = (
            f"Generated {result['generated']} activity aid(s); "
            f"{result['existing']} already existed; {result['failed']} failed review."
        )
        return render_evals(request, notice=notice)

    @app.post("/evals/{event_id}")
    def save_eval(
        request: Request,
        event_id: str,
        response_rating: Annotated[str, Form()],
        visual_rating: Annotated[str, Form()],
        preferred_response_shape: Annotated[str, Form()],
        preferred_visual_mix: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        note: Annotated[str | None, Form(max_length=2_000)] = None,
        visual_asset_id: Annotated[str | None, Form()] = None,
        artifact_id: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.record_output_evaluation(
                event_id=event_id,
                response_rating=response_rating,
                visual_rating=visual_rating,
                preferred_response_shape=preferred_response_shape,
                preferred_visual_mix=preferred_visual_mix,
                note=note,
                visual_asset_id=visual_asset_id or None,
                artifact_id=artifact_id or None,
            )
        except Exception as exc:
            return render_evals(request, error=str(exc))
        return RedirectResponse("/evals", status_code=303)

    @app.get("/api/health")
    def health():
        inventory = service.resource_inventory()
        return {
            "status": "ok",
            "database": "ready",
            "resource_units": len(inventory["units"]),
            "private_resource_excerpts_default": False,
        }

    @app.post("/children")
    def create_child(
        request: Request,
        child_id: Annotated[str, Form(min_length=1, max_length=120)],
        name: Annotated[str, Form(min_length=1, max_length=120)],
        csrf: Annotated[str, Form()] = "",
        birth_year: Annotated[int | None, Form()] = None,
        grade: Annotated[str | None, Form(max_length=40)] = None,
    ):
        check_csrf(csrf)
        try:
            service.add_child(child_id.strip(), name.strip(), birth_year, grade or None)
        except Exception as exc:
            return render_home(request, error=str(exc))
        return redirect_home(child_id)

    @app.post("/ask")
    def ask(
        request: Request,
        child_id: Annotated[str, Form()],
        question: Annotated[str, Form(min_length=1, max_length=20_000)],
        csrf: Annotated[str, Form()] = "",
        topics: Annotated[str | None, Form()] = None,
        private_excerpts: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.ask(
                child_id=child_id,
                text=question,
                topics=[item.strip() for item in (topics or "").split(",") if item.strip()],
                include_private_excerpts=private_excerpts == "yes",
            )
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="responses")

    @app.post("/inbox/{inbox_id}/assign")
    def assign_inbox(
        request: Request,
        inbox_id: str,
        child_id: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
    ):
        check_csrf(csrf)
        try:
            service.assign_inbox(inbox_id, child_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="responses")

    @app.post("/inbox/{inbox_id}/dismiss")
    def dismiss_inbox(
        request: Request,
        inbox_id: str,
        child_id: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
    ):
        check_csrf(csrf)
        try:
            service.dismiss_inbox(inbox_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="inbox")

    @app.post("/reflect")
    def reflect(request: Request, child_id: Annotated[str, Form()], csrf: Annotated[str, Form()] = ""):
        check_csrf(csrf)
        try:
            service.reflect(child_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="opportunities")

    @app.post("/opportunities/{opportunity_id}")
    def opportunity_decision(
        request: Request,
        opportunity_id: str,
        decision: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        child_id: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.respond_to_opportunity(opportunity_id, decision)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="opportunities")

    @app.post("/feedback")
    def feedback(
        request: Request,
        child_id: Annotated[str, Form()],
        outcome: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        artifact_id: Annotated[str | None, Form()] = None,
        experience_id: Annotated[str | None, Form()] = None,
        note: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.feedback(
                {
                    "child_id": child_id,
                    "outcome": outcome,
                    "artifact_id": artifact_id or None,
                    "experience_id": experience_id or None,
                    "note": note or None,
                }
            )
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return render_home(request, child=child_id, notice="Feedback saved as an observation.")

    @app.post("/resources/index")
    def index_resources(
        request: Request, csrf: Annotated[str, Form()] = "", child_id: Annotated[str | None, Form()] = None
    ):
        check_csrf(csrf)
        catalogs = discover_private_catalogs(root)
        try:
            if len(catalogs) != 1:
                raise ValueError(f"Expected one private resource catalog; found {len(catalogs)}.")
            report = service.index_resources(catalogs[0], root)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return render_home(
            request, child=child_id, notice=f"Indexed {report['units']} units and {report['documents']} documents."
        )

    @app.post("/actions/{action_id}/execute")
    def run_action(
        request: Request,
        action_id: str,
        csrf: Annotated[str, Form()] = "",
        child_id: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.execute_action(action_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="artifacts")

    @app.post("/responses/{event_id}/artifact")
    def response_artifact(
        request: Request,
        event_id: str,
        csrf: Annotated[str, Form()] = "",
        child_id: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.create_artifact_from_response(event_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="artifacts")

    @app.get("/artifacts/{artifact_id}/file")
    def artifact_file(artifact_id: str):
        with connect(service.db_path) as conn:
            row = conn.execute(
                "SELECT path,title FROM artifacts JOIN experiences ON experiences.id=artifacts.experience_id WHERE artifacts.id=?",
                (artifact_id,),
            ).fetchone()
        if not row:
            raise HTTPException(404)
        path = Path(row["path"]).resolve()
        try:
            path.relative_to(service.output_dir)
        except ValueError as exc:
            raise HTTPException(403) from exc
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    @app.get("/artifacts/{artifact_id}/preview")
    def artifact_preview(artifact_id: str):
        with connect(service.db_path) as conn:
            row = conn.execute("SELECT path FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        path = Path(row["path"]).with_suffix(".png").resolve()
        try:
            path.relative_to(service.output_dir)
        except ValueError as exc:
            raise HTTPException(403) from exc
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png")

    @app.post("/artifacts/{artifact_id}/approve")
    def approve(
        request: Request,
        artifact_id: str,
        csrf: Annotated[str, Form()] = "",
        child_id: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            service.approve_artifact(artifact_id)
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return redirect_home(child_id, anchor="artifacts")

    @app.post("/artifacts/{artifact_id}/print")
    def print_one(
        request: Request,
        artifact_id: str,
        approval_id: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        child_id: Annotated[str | None, Form()] = None,
        printer: Annotated[str | None, Form()] = None,
        send: Annotated[str | None, Form()] = None,
    ):
        check_csrf(csrf)
        try:
            result = service.print_artifact(
                artifact_id,
                approval_id,
                printer=printer or None,
                send=send == "yes",
            )
        except Exception as exc:
            return render_home(request, child=child_id, error=str(exc))
        return render_home(request, child=child_id, notice=f"Print workflow: {result['status']}.")

    return app
