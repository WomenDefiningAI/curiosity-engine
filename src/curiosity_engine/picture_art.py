"""Generate related artwork together, then crop exact reusable picture assets."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from uuid import uuid4

from PIL import Image

from .contracts import GeneratedIllustrationAsset


def prepare_pictures(page, *, backend, cache_dir, validate, cache_lock, preflight):
    used = {item.symbol_id for cell in page.cells for item in cell.items}
    ready = {}
    missing = []
    for symbol in page.symbols:
        asset = symbol.illustration
        if symbol.id not in used:
            ready[symbol.id] = symbol
        elif asset and _valid_file(asset):
            ready[symbol.id] = symbol
        else:
            missing.append(symbol)

    # One sheet keeps recurring characters and art direction in the same model
    # context. Counts, page layout, labels and text remain renderer-owned.
    for start in range(0, len(missing), 4):
        group = missing[start:start + 4]
        prompt = (
            "Create polished useful illustrations for a young child's printable. Warm storybook watercolor "
            "with crisp contours on pure white. Clear recognizable shapes; light colors for home printing. "
            "No words, letters, digits, captions, arrows, labels, answer spaces, borders or watermark. "
            "Every described individual object occurs ONCE in its illustration; exact repeated counts are "
            "handled later. Scene states must match each description, including which objects are absent. "
            "Keep recurring characters identical across scenes: same face, body, clothes and colors. "
        )
        if len(group) == 1:
            prompt += "One square illustration, subject centered, occupying 85% of the square, nothing cropped. "
        else:
            prompt += (
                "A square CONTACT SHEET with exactly four equally sized square quadrants in a 2 by 2 grid. "
                "The quadrant boundaries are exactly the horizontal and vertical middle of the canvas. "
                "Each picture is separately centered within its own quadrant, occupying 85% of that quadrant. "
                "Keep a white margin around every picture. No lines between quadrants. Nothing crosses a "
                "quadrant boundary. Unused quadrants must be completely white. Each quadrant contains only "
                "its own described scene; never borrow an object from another scene. "
            )
        positions = ["TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT"]
        prompt += "\n" + "\n".join(
            f"{positions[i] if len(group) > 1 else 'SUBJECT'}: {symbol.description}"
            for i, symbol in enumerate(group)
        )
        digest = sha256(f"picture-sheet-v1\n{backend.name}\n{backend.model}\n{prompt}".encode()).hexdigest()
        with cache_lock(digest):
            if preflight:
                preflight()
            assets = []
            for i in range(len(group)):
                path = cache_dir / f"picture-sheet-{digest}-{i}.png"
                metadata = path.with_suffix(".json")
                try:
                    asset = GeneratedIllustrationAsset.model_validate_json(metadata.read_text())
                    asset = asset.model_copy(update={"local_path": str(path)})
                    if asset.prompt_hash != digest or asset.model != backend.model or not _valid_file(asset):
                        raise ValueError("invalid picture cache")
                    assets.append(asset)
                except (OSError, ValueError):
                    assets = []
                    break
            if not assets:
                generated = backend.generate(prompt)
                if preflight:
                    preflight()
                # Validate the provider bytes before decoding/cropping. Retain the
                # full source sheet and prompt for reproducible visual inspection.
                sheet_path = cache_dir / f"picture-sheet-{digest}.png"
                validate(generated.data, path=sheet_path, backend=backend, prompt_hash=digest,
                         alt_text="Source artwork sheet", request_id=generated.request_id)
                with Image.open(BytesIO(generated.data)) as sheet:
                    width, height = sheet.size
                    if len(group) > 1 and (width != height or width < 1024):
                        raise ValueError("A multi-picture sheet must be square and at least 1024 pixels")
                    for i, symbol in enumerate(group):
                        path = cache_dir / f"picture-sheet-{digest}-{i}.png"
                        half = width // 2
                        x, y = (i % 2) * half, (i // 2) * half
                        picture = sheet if len(group) == 1 else sheet.crop((x, y, x + half, y + half))
                        buffer = BytesIO()
                        picture.save(buffer, format="PNG")
                        data = buffer.getvalue()
                        asset = validate(data, path=path, backend=backend, prompt_hash=digest,
                                         alt_text=("Activity picture: " + symbol.description)[:500],
                                         request_id=generated.request_id).model_copy(
                                             update={"prompt_version": "picture-sheet-v1"})
                        _write(path, data)
                        _write(path.with_suffix(".json"), (asset.model_dump_json(indent=2) + "\n").encode())
                        assets.append(asset)
                _write(sheet_path, generated.data)
                _write(sheet_path.with_suffix(".txt"), prompt.encode())
            for symbol, asset in zip(group, assets, strict=True):
                ready[symbol.id] = symbol.model_copy(update={"illustration": asset})
    if preflight:
        preflight()
    return page.model_copy(update={"symbols": [ready[s.id] for s in page.symbols]})


def _valid_file(asset):
    from pathlib import Path
    path = Path(asset.local_path)
    return path.is_file() and not path.is_symlink() and sha256(path.read_bytes()).hexdigest() == asset.sha256


def _write(path, data):
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o600)
    temporary.replace(path)
