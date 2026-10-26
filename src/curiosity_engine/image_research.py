from __future__ import annotations

import io
import json
import os
import re
import ssl
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from PIL import Image, UnidentifiedImageError

from .contracts import ImageResearchQuery, ImageRights, ImageSafetyReview, ResearchedImageAsset, utc_now
from .db import connect, init_db, jdump, utcnow

MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 32_000_000
MIN_DIMENSION = 300
MAX_DIMENSION = 8_192
RESEARCH_PIPELINE_VERSION = 3
RESEARCH_CACHE_TTL = timedelta(days=30)
FAILED_QUERY_COOLDOWN = timedelta(hours=1)
MAX_CANDIDATES_PER_REQUEST = 8
MAX_VISION_REVIEWS_PER_REQUEST = 4
SOURCE_PRIORITY = ("smithsonian", "nasa", "noaa", "usgs", "wikimedia")
API_HOSTS = {"api.si.edu", "images-api.nasa.gov", "commons.wikimedia.org"}
MEDIA_HOSTS = {
    "ids.si.edu",
    "images-assets.nasa.gov",
    "upload.wikimedia.org",
}
PUBLIC_QUERY_PROPER_NOUNS = {
    "earth",
    "mars",
    "moon",
    "sun",
    "nasa",
    "smithsonian",
    "noaa",
    "usgs",
    "hubble",
    "webb",
    "curiosity",
}


class ImageReviewUnavailable(RuntimeError):
    """The configured reviewer failed independently of a particular candidate."""


class ImageResearchUnavailable(RuntimeError):
    """Every viable collection or download failed transiently."""


class NoApprovedImageError(ValueError):
    """Candidates were reachable, but none passed permanent release gates."""


@dataclass(frozen=True)
class ImageCandidate:
    source_name: str
    source_item_id: str
    title: str
    description: str
    creator: str
    canonical_url: str
    media_url: str
    license_name: str
    license_url: str
    attribution: str
    rights_status: str


class CandidateSource(Protocol):
    name: str

    def search(self, query: ImageResearchQuery, *, limit: int) -> list[ImageCandidate]: ...


class _AllowlistedRedirects(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _require_https_host(newurl, API_HOSTS | MEDIA_HOSTS)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_https_host(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("image research attempted a non-allowlisted URL")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("image research URL contains disallowed authority fields")


def _get_bytes(url: str, *, allowed_hosts: set[str], maximum: int = MAX_DOWNLOAD_BYTES) -> tuple[bytes, str]:
    _require_https_host(url, allowed_hosts)
    opener = build_opener(HTTPSHandler(context=ssl.create_default_context()), _AllowlistedRedirects())
    request = Request(url, headers={"User-Agent": "CuriosityEngine/0.1 (+https://github.com/WomenDefiningAI/curiosity-engine)"})
    try:
        with opener.open(request, timeout=20) as response:
            final_url = str(response.geturl())
            _require_https_host(final_url, allowed_hosts)
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > maximum:
                raise ValueError("image response exceeded the download limit")
            payload = response.read(maximum + 1)
            if len(payload) > maximum:
                raise ValueError("image response exceeded the download limit")
            return payload, str(response.headers.get_content_type())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("public image collection request failed") from exc


def _get_json(url: str) -> dict[str, Any]:
    raw, mime = _get_bytes(url, allowed_hosts=API_HOSTS, maximum=4 * 1024 * 1024)
    if mime not in {"application/json", "text/json", "text/plain"}:
        raise ValueError("image collection returned a non-JSON response")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("image collection returned an invalid JSON document")
    return parsed


class NasaImageSource:
    name = "nasa"

    def search(self, query: ImageResearchQuery, *, limit: int) -> list[ImageCandidate]:
        url = "https://images-api.nasa.gov/search?" + urlencode(
            {"q": query.query, "media_type": "image", "page_size": min(limit, 20)}
        )
        payload = _get_json(url)
        candidates: list[ImageCandidate] = []
        for item in ((payload.get("collection") or {}).get("items") or []):
            data = (item.get("data") or [{}])[0]
            links = item.get("links") or []
            image_link = next((link for link in links if link.get("render") == "image" and link.get("href")), None)
            nasa_id = str(data.get("nasa_id") or "").strip()
            if not image_link or not nasa_id:
                continue
            copyright_holder = str(data.get("copyright") or "").strip()
            if copyright_holder:
                continue
            policy_text = " ".join(
                str(data.get(key) or "") for key in ("description", "photographer", "secondary_creator")
            ).casefold()
            if any(
                marker in policy_text
                for marker in (
                    "all rights reserved",
                    "permission",
                    "courtesy",
                    "non-commercial",
                    "noncommercial",
                    "not for commercial use",
                    "copyright",
                    "©",
                )
            ):
                continue
            candidates.append(
                ImageCandidate(
                    source_name="nasa",
                    source_item_id=nasa_id,
                    title=str(data.get("title") or nasa_id)[:300],
                    description=str(data.get("description") or "")[:2_000],
                    creator=str(data.get("photographer") or data.get("center") or "NASA")[:500],
                    canonical_url=f"https://images.nasa.gov/details/{quote_plus(nasa_id)}",
                    media_url=str(image_link["href"]),
                    license_name="NASA Media Usage Guidelines",
                    license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                    attribution="NASA" + (f" / {data.get('center')}" if data.get("center") else ""),
                    rights_status="policy_based",
                )
            )
        return candidates[:limit]


class WikimediaImageSource:
    name = "wikimedia"

    def search(self, query: ImageResearchQuery, *, limit: int) -> list[ImageCandidate]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query.query,
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 20),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": 1600,
            "origin": "*",
        }
        payload = _get_json("https://commons.wikimedia.org/w/api.php?" + urlencode(params))
        pages = (payload.get("query") or {}).get("pages") or {}
        candidates: list[ImageCandidate] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata") or {}
            media_url = str(info.get("thumburl") or info.get("url") or "")
            mime = str(info.get("mime") or "")
            if not media_url or mime not in {"image/jpeg", "image/png"}:
                continue
            license_name = str((metadata.get("LicenseShortName") or {}).get("value") or "")
            license_url = str((metadata.get("LicenseUrl") or {}).get("value") or "")
            if not license_url and "public domain" in license_name.casefold():
                license_url = "https://creativecommons.org/publicdomain/mark/1.0/"
            creator = _plain_text(str((metadata.get("Artist") or {}).get("value") or "Unknown creator"))
            attribution = _plain_text(
                str((metadata.get("Attribution") or {}).get("value") or creator or "Wikimedia Commons contributor")
            )
            description = _plain_text(str((metadata.get("ImageDescription") or {}).get("value") or ""))
            candidates.append(
                ImageCandidate(
                    source_name="wikimedia",
                    source_item_id=str(page.get("pageid") or page.get("title") or "")[:240],
                    title=str(page.get("title") or "Wikimedia Commons image").removeprefix("File:")[:300],
                    description=description[:2_000],
                    creator=creator[:500],
                    canonical_url=str(info.get("descriptionurl") or "https://commons.wikimedia.org/")[:2_000],
                    media_url=media_url,
                    license_name=license_name[:160],
                    license_url=license_url[:2_000],
                    attribution=attribution[:1_000],
                    rights_status="verified",
                )
            )
        return candidates[:limit]


class SmithsonianImageSource:
    name = "smithsonian"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SMITHSONIAN_OPEN_ACCESS_API_KEY", "")

    def search(self, query: ImageResearchQuery, *, limit: int) -> list[ImageCandidate]:
        if not self.api_key:
            return []
        url = "https://api.si.edu/openaccess/api/v1.0/search?" + urlencode(
            {"q": query.query, "rows": min(limit, 20), "api_key": self.api_key}
        )
        payload = _get_json(url)
        rows = ((payload.get("response") or {}).get("rows") or [])
        candidates: list[ImageCandidate] = []
        for row in rows:
            content = row.get("content") or {}
            descriptive = content.get("descriptiveNonRepeating") or {}
            online = descriptive.get("online_media") or {}
            media = online.get("media") or []
            image = next((entry for entry in media if str(entry.get("type") or "").casefold() == "images"), None)
            if not image or not image.get("content"):
                continue
            usage = _plain_text(str(descriptive.get("usage") or ""))
            if "cc0" not in usage.casefold():
                continue
            record_id = str(row.get("id") or "")
            candidates.append(
                ImageCandidate(
                    source_name="smithsonian",
                    source_item_id=record_id[:240],
                    title=str(row.get("title") or record_id)[:300],
                    description=_plain_text(str(descriptive.get("title") or ""))[:2_000],
                    creator="Smithsonian Institution",
                    canonical_url=str(row.get("url") or f"https://www.si.edu/object/{record_id}")[:2_000],
                    media_url=str(image["content"]),
                    license_name="CC0 1.0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    attribution="Smithsonian Institution",
                    rights_status="verified",
                )
            )
        return candidates[:limit]


def _plain_text(value: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", value).replace("&nbsp;", " ").split())


def _license_allowed(candidate: ImageCandidate) -> bool:
    normalized = " ".join(candidate.license_name.casefold().replace("-", " ").split())
    if candidate.source_name == "nasa":
        return candidate.rights_status == "policy_based"
    if "public domain" in normalized or normalized.startswith("cc0") or "public domain mark" in normalized:
        return True
    return bool(re.search(r"\bcc by(?: [234]\.0)?\b", normalized)) and all(
        marker not in normalized for marker in ("share alike", " sa", "noncommercial", " nc", "no derivatives", " nd")
    )


def _metadata_hash(candidate: ImageCandidate) -> str:
    return sha256(jdump(candidate.__dict__).encode()).hexdigest()


def _rights(candidate: ImageCandidate) -> ImageRights:
    if not _license_allowed(candidate):
        raise ValueError("candidate license is not in the conservative allowlist")
    return ImageRights(
        source_name=candidate.source_name,
        source_item_id=candidate.source_item_id,
        canonical_url=candidate.canonical_url,
        creator=candidate.creator,
        license_name=candidate.license_name,
        license_url=candidate.license_url,
        attribution=candidate.attribution,
        rights_status=candidate.rights_status,
        modifications_allowed=None if candidate.source_name == "nasa" else True,
        commercial_use_allowed=None if candidate.source_name == "nasa" else True,
        metadata_sha256=_metadata_hash(candidate),
        rights_notes=(
            [
                "NASA policy-based use; item metadata showed no explicit restriction, but modification and "
                "commercial permissions are not asserted. Check third-party credits, insignia, and endorsement."
            ]
            if candidate.source_name == "nasa"
            else []
        ),
    )


def _normalize_image(payload: bytes, declared_mime: str, cache_dir: Path) -> tuple[Path, str, int, int, int, str]:
    if declared_mime not in {"image/jpeg", "image/png", "application/octet-stream"}:
        raise ValueError("candidate did not declare a supported raster image type")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("animated or multi-frame images are not accepted")
            width, height = opened.size
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                raise ValueError("candidate is too small for a useful learning visual")
            if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
                raise ValueError("candidate pixel dimensions exceed the safe decode limit")
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            normalized = opened.convert("RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("candidate is not a safe decodable image") from exc
    clean = output.getvalue()
    if len(clean) > MAX_DOWNLOAD_BYTES:
        raise ValueError("normalized image exceeded the private cache size limit")
    digest = sha256(clean).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.chmod(0o700)
    path = cache_dir / f"{digest}.png"
    if not path.exists():
        path.write_bytes(clean)
        path.chmod(0o600)
    return path.resolve(), "image/png", width, height, len(clean), digest


def _review_image(
    candidate: ImageCandidate,
    query: ImageResearchQuery,
    png: bytes,
    backend: Any,
) -> ImageSafetyReview:
    if backend is None:
        raise RuntimeError("researched images require a configured vision safety reviewer")
    try:
        result = backend.complete(
            role="visual_qa",
            system=(
                "Release-gate one public photo for an elementary-age child. Fail on people, faces, graphic or gory "
                "content, frightening imagery, visible personal information, advertising, prominent watermarks, unsafe "
                "behavior, or a subject that does not clearly answer the minimized research query. In "
                "observable_description, describe only the useful visible details a child should notice, in one or two "
                "plain sentences suitable as image alt text. Classify representation_type. A direct camera image is a "
                "photograph; false-color, enhanced-color, composite, stitched, mapped, or heavily processed scientific "
                "imagery is processed_science_image; artwork is illustration. For processed science imagery, write a "
                "short context_note supported by the candidate metadata that tells a child what was processed. Use "
                "unknown when the representation cannot be verified. Do not infer facts beyond the visible image and "
                "provided source metadata. Return the exact ImageSafetyReview contract."
            ),
            payload={
                "query": query.model_dump(mode="json"),
                "candidate_metadata": {
                    "source": candidate.source_name,
                    "title": candidate.title,
                    "description": candidate.description[:800],
                },
                "image_data_urls": ["data:image/png;base64," + b64encode(png).decode("ascii")],
            },
            response_model=ImageSafetyReview,
        )
        review_payload = result.model_dump(mode="python") if hasattr(result, "model_dump") else dict(result)
        review_payload["reviewed_at"] = utc_now()
        review_payload["reviewer"] = backend.__class__.__name__[:160]
        review = ImageSafetyReview.model_validate(review_payload)
    except Exception as exc:
        raise ImageReviewUnavailable("image safety reviewer unavailable") from exc
    if review.representation_type not in {"photograph", "processed_science_image"}:
        raise ValueError("candidate was not verified as observational imagery")
    if review.representation_type == "processed_science_image" and not review.context_note.strip():
        raise ValueError("processed science imagery requires a child-readable context note")
    if review.verdict != "pass" or review.people_present or not review.subject_matches:
        raise ValueError("candidate did not pass child-safety and relevance review")
    return review


class ImageResearchService:
    """Find, quarantine, review, cache, and provenance one public learning photo."""

    def __init__(
        self,
        db_path: str | Path,
        output_dir: str | Path,
        *,
        review_backend: Any,
        sources: list[CandidateSource] | None = None,
        authorization_preflight: Callable[[], None] | None = None,
        authorization_transaction_assertion: Callable[[Any], None] | None = None,
    ):
        if (authorization_preflight is None) != (authorization_transaction_assertion is None):
            raise ValueError(
                "personalized image research requires both preflight and transaction authorization guards"
            )
        self.db_path = str(db_path)
        self.cache_dir = Path(output_dir).resolve() / "researched-images"
        self.review_backend = review_backend
        self.sources = sources or [SmithsonianImageSource(), NasaImageSource(), WikimediaImageSource()]
        self.authorization_preflight = authorization_preflight
        self.authorization_transaction_assertion = authorization_transaction_assertion
        init_db(self.db_path)

    def _preflight(self) -> None:
        if self.authorization_preflight is not None:
            self.authorization_preflight()

    def _assert_authorized_in_transaction(self, conn: Any) -> None:
        if self.authorization_transaction_assertion is not None:
            self.authorization_transaction_assertion(conn)

    def _discard_unreferenced_image(self, path: Path | None) -> None:
        if path is None or not path.exists() or path.is_symlink():
            return
        resolved = path.resolve()
        if not resolved.is_relative_to(self.cache_dir.resolve()) or not resolved.is_file():
            return
        digest = sha256(resolved.read_bytes()).hexdigest()
        with connect(self.db_path) as conn:
            referenced = conn.execute(
                "SELECT 1 FROM researched_image_cache WHERE asset_sha256=? AND status='approved'",
                (digest,),
            ).fetchone()
        if not referenced:
            resolved.unlink(missing_ok=True)

    def find(self, query: ImageResearchQuery, *, private_source_text: str | None = None) -> ResearchedImageAsset:
        self._preflight()
        self._reject_private_query(query, private_source_text=private_source_text)
        cache_key = sha256(
            jdump({"pipeline_version": RESEARCH_PIPELINE_VERSION, "query": query.model_dump(mode="json")}).encode()
        ).hexdigest()
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status,asset_json,updated_at FROM researched_image_cache WHERE query_sha256=?",
                (cache_key,),
            ).fetchone()
        if row:
            try:
                cached_at = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
                age = datetime.now(UTC) - cached_at
                if row["status"] == "failed" and age <= FAILED_QUERY_COOLDOWN:
                    raise NoApprovedImageError("no approved public image is available for this topic yet")
                if row["status"] == "approved":
                    cached = ResearchedImageAsset.model_validate_json(row["asset_json"])
                    path = Path(cached.local_path)
                    if age <= RESEARCH_CACHE_TTL and path.is_file() and sha256(path.read_bytes()).hexdigest() == cached.sha256:
                        self._preflight()
                        return cached
            except NoApprovedImageError:
                raise
            except (TypeError, ValueError):
                pass

        preferred = list(dict.fromkeys([*query.preferred_sources, *SOURCE_PRIORITY]))
        indexed = {source.name: source for source in self.sources}
        failures: list[str] = []
        candidates_seen = 0
        vision_reviews = 0
        permanent_rejections = 0
        transient_failures = 0
        for source_name in preferred:
            if candidates_seen >= MAX_CANDIDATES_PER_REQUEST or vision_reviews >= MAX_VISION_REVIEWS_PER_REQUEST:
                break
            source = indexed.get(source_name)
            if source is None:
                continue
            try:
                self._preflight()
                candidates = source.search(query, limit=MAX_CANDIDATES_PER_REQUEST - candidates_seen)
            except Exception as exc:
                self._preflight()
                failures.append(f"{source_name}: {exc.__class__.__name__}")
                transient_failures += 1
                continue
            for candidate in candidates:
                if candidates_seen >= MAX_CANDIDATES_PER_REQUEST or vision_reviews >= MAX_VISION_REVIEWS_PER_REQUEST:
                    break
                candidates_seen += 1
                path: Path | None = None
                try:
                    self._preflight()
                    rights = _rights(candidate)
                    payload, mime = _get_bytes(candidate.media_url, allowed_hosts=MEDIA_HOSTS)
                    self._preflight()
                    path, clean_mime, width, height, byte_count, digest = _normalize_image(
                        payload, mime, self.cache_dir
                    )
                    vision_reviews += 1
                    self._preflight()
                    safety = _review_image(candidate, query, path.read_bytes(), self.review_backend)
                    observable = safety.observable_description.strip() or (
                        f"Public learning photo titled {candidate.title}, selected to help a child "
                        f"{query.purpose} details related to {query.query}."
                    )
                    context_note = safety.context_note.strip()
                    alt_text = f"{observable} {context_note}".strip()
                    asset = ResearchedImageAsset(
                        title=candidate.title,
                        caption=f"{observable} {context_note}".strip()[:500],
                        alt_text=alt_text,
                        content_role=query.purpose,
                        local_path=str(path),
                        mime_type=clean_mime,
                        width=width,
                        height=height,
                        byte_count=byte_count,
                        sha256=digest,
                        rights=rights,
                        safety=safety,
                        transformations=["decoded", "converted to RGB", "re-encoded as PNG", "metadata stripped"],
                    )
                    self._preflight()
                    self._save(cache_key, query, asset)
                    return asset
                except ImageReviewUnavailable:
                    self._discard_unreferenced_image(path)
                    raise
                except Exception as exc:
                    # Authorization failures remain active. Remove any newly
                    # normalized, unreferenced bytes before asking the guard to
                    # re-raise the exact revocation error.
                    self._discard_unreferenced_image(path)
                    self._preflight()
                    failures.append(f"{source_name}/{candidate.source_item_id}: {exc.__class__.__name__}")
                    if isinstance(exc, RuntimeError):
                        transient_failures += 1
                    else:
                        permanent_rejections += 1
        if transient_failures and not permanent_rejections:
            raise ImageResearchUnavailable("public image collections or downloads were temporarily unavailable")
        self._preflight()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_authorized_in_transaction(conn)
            conn.execute(
                """INSERT INTO researched_image_cache(
                     query_sha256,query_json,status,failure_json,created_at,updated_at
                   ) VALUES(?,?,'failed',?,?,?)
                   ON CONFLICT(query_sha256) DO UPDATE SET status='failed',failure_json=excluded.failure_json,
                     updated_at=excluded.updated_at""",
                (cache_key, query.model_dump_json(), jdump(failures[-20:]), utcnow(), utcnow()),
            )
        raise NoApprovedImageError("no public image passed rights, safety, and relevance review")

    def _reject_private_query(self, query: ImageResearchQuery, *, private_source_text: str | None) -> None:
        lowered = query.query.casefold()
        with connect(self.db_path) as conn:
            names = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM children UNION ALL SELECT display_name FROM parent_principals"
                )
            ]
        for name in names:
            private_tokens = [token for token in re.findall(r"[a-z]+", name.casefold()) if len(token) >= 3]
            if any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in private_tokens):
                raise ValueError("image research query overlaps a private household identity")
        if private_source_text:
            proper_nouns = {
                token.casefold()
                for token in re.findall(r"\b[A-Z][a-z]{2,}\b", private_source_text)
                if token.casefold() not in PUBLIC_QUERY_PROPER_NOUNS
                and token.casefold()
                not in {"can", "could", "does", "draw", "how", "is", "please", "show", "what", "why"}
            }
            if any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in proper_nouns):
                raise ValueError("image research query overlaps a proper name from private source text")
            named_private_places = {
                token.casefold()
                for token in re.findall(
                    r"\b([A-Za-z][A-Za-z'-]{2,})\s+(?:elementary|school|academy|clinic|hospital|street|avenue|road)\b",
                    private_source_text,
                    flags=re.IGNORECASE,
                )
            }
            if any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in named_private_places):
                raise ValueError("image research query overlaps a named private place")

    def _save(self, cache_key: str, query: ImageResearchQuery, asset: ResearchedImageAsset) -> None:
        now = utcnow()
        with connect(self.db_path) as conn:
            # BEGIN IMMEDIATE serializes a concurrent revocation with this
            # exact-decision assertion and the approved cache write.
            conn.execute("BEGIN IMMEDIATE")
            self._assert_authorized_in_transaction(conn)
            conn.execute(
                """INSERT INTO researched_image_cache(
                     query_sha256,query_json,status,asset_sha256,asset_json,rights_json,review_json,
                     source_name,source_item_id,canonical_url,created_at,updated_at
                   ) VALUES(?,?,'approved',?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(query_sha256) DO UPDATE SET status='approved',asset_sha256=excluded.asset_sha256,
                     asset_json=excluded.asset_json,rights_json=excluded.rights_json,review_json=excluded.review_json,
                     source_name=excluded.source_name,source_item_id=excluded.source_item_id,
                     canonical_url=excluded.canonical_url,failure_json=NULL,updated_at=excluded.updated_at""",
                (
                    cache_key,
                    query.model_dump_json(),
                    asset.sha256,
                    asset.model_dump_json(),
                    asset.rights.model_dump_json(),
                    asset.safety.model_dump_json(),
                    asset.rights.source_name,
                    asset.rights.source_item_id,
                    asset.rights.canonical_url,
                    now,
                    now,
                ),
            )
