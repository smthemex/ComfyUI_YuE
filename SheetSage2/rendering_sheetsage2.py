"""Optional offline piano and staff rendering. No model or GPU is loaded."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import mimetypes
from pathlib import Path
import re
import wave


PARTS = ("mix", "melody", "vocal", "instrumental", "chords")
FORMATS = ("pdf", "svg", "png")


def _options(value, allowed, label):
    values = value.split(",") if isinstance(value, str) else list(value)
    values = tuple(dict.fromkeys(x.strip().lower() for x in values if x.strip()))
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(sorted(unknown))}. Choose from {', '.join(allowed)}.")
    return values


def validate_render_options(*, audio=False, score=(), parts=("mix",)):
    """Validate output choices without importing a model or browser runtime.

    Return normalized ``(score_formats, audio_parts)`` tuples. ``score=True``
    selects PDF; no requested output is valid for an inference-only call.
    """
    score = ("pdf",) if score is True else () if score is False or score is None else score
    formats = _options(score, FORMATS, "score format")
    requested = _options(parts, PARTS + ("all",), "audio part")
    if "all" in requested:
        requested = PARTS
    if audio and not requested:
        raise ValueError("Choose at least one audio part.")
    return formats, requested


def _tracks(midi):
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("Install rendering support first: python setup_render.py") from exc
    try:
        source = io.BytesIO(bytes(midi)) if isinstance(midi, (bytes, bytearray, memoryview)) else str(midi)
        parsed = pretty_midi.PrettyMIDI(source)
    except Exception as exc:
        name = midi.name if isinstance(midi, Path) else "in-memory MIDI"
        raise ValueError(f"Cannot read MIDI: {name}: {exc}") from exc
    result = []
    for instrument in parsed.instruments:
        if instrument.is_drum:
            raise ValueError("Piano rendering does not support drum tracks.")
        notes = []
        for note in instrument.notes:
            if not all(math.isfinite(x) for x in (note.start, note.end)) or note.start < 0 or note.end <= note.start:
                raise ValueError("MIDI note times must be finite, nonnegative, and increasing.")
            if not 21 <= note.pitch <= 109:
                raise ValueError(f"Piano samples cover MIDI pitches 21–109; received {note.pitch}.")
            notes.append(dict(pitch=int(note.pitch), start=float(note.start), end=float(note.end), velocity=int(note.velocity)))
        result.append(dict(name=instrument.name, notes=notes))
    return result, float(parsed.get_end_time())


def _select(tracks, part):
    def role(track):
        name = track["name"].strip().lower()
        if "chord" in name:
            return "chords"
        if "vocal" in name:
            return "vocal"
        if name in {"ins", "instrument", "instrumental"} or "instrumental" in name:
            return "instrumental"
        return "melody"
    if part == "mix":
        return tracks
    if part == "melody":
        return [track for track in tracks if role(track) != "chords"]
    return [track for track in tracks if role(track) == part]


def _verify_assets(directory, *, audio):
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Rendering assets are missing. Download render_assets/ from the model repository.")
    inventory = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    for relative, digest in inventory.items():
        if not audio and relative.startswith("soundfonts/"):
            continue
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Rendering asset is missing or corrupt: {relative}")
    return set(inventory)


def _metadata_duration(directory):
    for name, key in (("result.json", "duration_seconds"), ("playback.json", "duration")):
        path = directory / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
            if value is not None:
                return float(value)
    return None


def render_outputs(input_dir=None, *, midi=None, abc=None, output_dir=None,
                   audio=False, score=(), parts=("mix",), duration=None,
                   assets_dir=None):
    """Render existing outputs, preserving MIDI note times and canonical ABC.

    ``audio=True`` writes piano WAV; ``score`` selects pdf/svg/png. ``parts``
    selects audio stems; ``all`` requests every available stem. SVG and PNG
    contain one A4 page per file; PDF contains every page. No network request
    is permitted during rendering. Assets and the browser must be installed.
    """
    formats, requested = validate_render_options(audio=audio, score=score, parts=parts)
    if not audio and not formats:
        raise ValueError("Request audio or at least one score format.")
    directory = Path(input_dir).expanduser().resolve() if input_dir else None
    if directory is not None and not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    destination = Path(output_dir or directory or "rendered").expanduser().resolve()
    if midi is None and directory:
        midi = directory / ("transcription.mid" if (directory / "transcription.mid").exists() else "melody.mid")
    if abc is None and directory:
        abc = directory / "score.abc"
    if duration is None and directory:
        duration = _metadata_duration(directory)
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError("Duration must be finite and nonnegative.")
    tracks, seconds = [], 0.0
    if audio:
        if midi is None or not Path(midi).is_file():
            raise FileNotFoundError("Audio rendering needs an existing MIDI file (--midi or --input).")
        tracks, end = _tracks(Path(midi))
        seconds = max(end, duration or 0)
        if seconds <= 0:
            raise ValueError("An empty MIDI needs a positive --duration.")
    abc_text = None
    if formats:
        if abc is None or not Path(abc).is_file():
            raise FileNotFoundError("Score rendering needs an existing ABC file (--abc or --input).")
        abc_text = Path(abc).read_text(encoding="utf-8")
        if not abc_text.strip():
            raise ValueError("ABC input is empty.")
    return _render(tracks, seconds, abc_text, audio=audio, formats=formats,
                   requested=requested, destination=destination, assets_dir=assets_dir)


def render_memory(*, midi=None, abc=None, audio=False, score=(), parts=("mix",),
                  duration=None, assets_dir=None):
    """Render MIDI bytes and ABC text without writing input or result files.

    Return WAV bytes in ``audio[part]`` and page lists in ``score[format]``:
    PDF and PNG values are bytes, SVG values are strings. A PDF list contains
    one complete document. The optional browser runtime may use its own profile.
    """
    formats, requested = validate_render_options(audio=audio, score=score, parts=parts)
    if not audio and not formats:
        raise ValueError("Request audio or at least one score format.")
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError("Duration must be finite and nonnegative.")
    tracks, seconds = [], 0.0
    if audio:
        if not isinstance(midi, (bytes, bytearray, memoryview)):
            raise ValueError("Audio rendering needs MIDI bytes.")
        tracks, end = _tracks(midi)
        seconds = max(end, duration or 0)
        if seconds <= 0:
            raise ValueError("An empty MIDI needs a positive duration.")
    if formats and (not isinstance(abc, str) or not abc.strip()):
        raise ValueError("Score rendering needs nonempty ABC text.")
    return _render(tracks, seconds, abc, audio=audio, formats=formats,
                   requested=requested, assets_dir=assets_dir)


def _render(tracks, seconds, abc_text, *, audio, formats, requested,
            destination=None, assets_dir=None):
    assets = Path(assets_dir) if assets_dir else Path(__file__).with_name("render_assets")
    assets = assets.resolve()
    asset_files = _verify_assets(assets, audio=audio)
    try:
        from playwright.sync_api import sync_playwright, Error as BrowserError
    except ImportError as exc:
        raise RuntimeError("Install rendering support first: python setup_render.py") from exc

    output = {"audio": {}, "score": {}, "warnings": []}
    blocked = []
    previous_pages = []
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for path in destination.iterdir():
            match = re.fullmatch(r"score_(\d{3,})\.(svg|png)", path.name)
            if match and match[2] in formats and path.is_file() and not path.is_symlink():
                previous_pages.append((int(match[1]), path))
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required", "--disable-gpu"])
        except Exception as exc:
            raise RuntimeError("Could not start the renderer. Run python setup_render.py; on Linux with missing system libraries, use --with-deps.") from exc
        try:
            context = browser.new_context(viewport={"width": 794, "height": 1123}, device_scale_factor=2, service_workers="block", offline=True)
            def route_handler(route):
                from urllib.parse import unquote, urlsplit
                url = urlsplit(route.request.url)
                if url.scheme != "https" or url.netloc != "render.invalid":
                    blocked.append(route.request.url)
                    route.abort()
                    return
                if url.path == "/":
                    route.fulfill(status=200, content_type="text/html", body="<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;background:white}.score-page{display:block;break-after:page;width:210mm;height:297mm}.score-page:last-child{break-after:auto}@page{size:A4;margin:0}</style></head><body></body></html>")
                    return
                relative = unquote(url.path.lstrip("/"))
                file = assets / relative
                # HF snapshots link verified assets into the shared blob cache.
                # Whitelist logical names instead of rejecting those symlinks.
                if relative not in asset_files or not file.is_file():
                    blocked.append(url.path)
                    route.abort()
                    return
                route.fulfill(status=200, path=str(file), content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            context.route("**/*", route_handler)
            page = context.new_page()
            page.goto("https://render.invalid/")
            page.add_script_tag(url="https://render.invalid/abcjs-basic-min.js")
            page.add_script_tag(url="https://render.invalid/renderer.js")
            if audio:
                for part in requested:
                    selected = _select(tracks, part)
                    if not selected and part not in {"mix", "melody"}:
                        output["warnings"].append(f"No {part} track is available; skipped.")
                        continue
                    info = page.evaluate("renderAudio", {"tracks": selected, "duration": seconds})
                    target = destination / f"piano_{part}.wav" if destination is not None else None
                    temporary = target.with_suffix(".wav.partial") if target is not None else None
                    buffer = io.BytesIO() if target is None else None
                    try:
                        with wave.open(str(temporary) if temporary is not None else buffer, "wb") as stream:
                            stream.setnchannels(info["channels"])
                            stream.setsampwidth(2)
                            stream.setframerate(info["sample_rate"])
                            for start in range(0, info["frames"], 32768):
                                encoded = page.evaluate("audioBlock", {"start": start, "count": 32768})
                                stream.writeframesraw(base64.b64decode(encoded))
                        if temporary is not None:
                            temporary.replace(target)
                        output["audio"][part] = str(target) if target is not None else buffer.getvalue()
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                        page.evaluate("window.renderedAudio = null")
            if formats:
                page.evaluate("""async ({font, license}) => {
                    window.renderFontData = font;
                    window.renderFontLicense = license;
                    const face = new FontFace('SheetSageSans', 'url(data:font/ttf;base64,' + font + ')');
                    document.fonts.add(await face.load());
                }""", {"font": base64.b64encode((assets / "DejaVuSans.ttf").read_bytes()).decode("ascii"),
                        "license": (assets / "LICENSE.font").read_text(encoding="utf-8")})
                info = page.evaluate("renderScore", abc_text)
                for kind in formats:
                    output["score"][kind] = []
                if "pdf" in formats:
                    if destination is None:
                        output["score"]["pdf"] = [page.pdf(format="A4", print_background=True, prefer_css_page_size=True)]
                    else:
                        target = destination / "score.pdf"
                        temporary = target.with_suffix(".pdf.partial")
                        page.pdf(path=str(temporary), format="A4", print_background=True, prefer_css_page_size=True)
                        temporary.replace(target)
                        output["score"]["pdf"] = [str(target)]
                for index, svg in enumerate(info["pages"], 1):
                    if "svg" in formats:
                        if destination is None:
                            output["score"]["svg"].append(svg)
                        else:
                            target = destination / f"score_{index:03}.svg"
                            target.write_text(svg, encoding="utf-8")
                            output["score"]["svg"].append(str(target))
                    if "png" in formats:
                        element = page.locator(".score-page").nth(index-1)
                        if destination is None:
                            output["score"]["png"].append(element.screenshot(type="png"))
                        else:
                            target = destination / f"score_{index:03}.png"
                            element.screenshot(path=str(target))
                            output["score"]["png"].append(str(target))
                output["warnings"].extend(info["warnings"])
            if blocked:
                raise RuntimeError(f"Rendering attempted to load unavailable assets: {', '.join(blocked)}")
        except BrowserError as exc:
            raise RuntimeError(str(exc).split("Call log:", 1)[0].strip()) from exc
        finally:
            browser.close()
    # A failed render leaves existing pages intact. Only remove obsolete pages
    # in the requested formats, within this renderer's numbered-file namespace.
    if formats:
        for number, path in previous_pages:
            if number > len(info["pages"]):
                path.unlink(missing_ok=True)
    return output
