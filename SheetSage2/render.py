"""Render existing SheetSage2 MIDI and ABC outputs without loading a model."""
import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rendering_sheetsage2 import render_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Transcription output directory.")
    parser.add_argument("--midi", help="MIDI file with original note timing.")
    parser.add_argument("--abc", help="ABC score file.")
    parser.add_argument("--output", help="Output directory; defaults to --input or rendered/.")
    parser.add_argument("--audio", action="store_true", help="Render piano WAV.")
    parser.add_argument("--score", default="", help="Comma-separated pdf,svg,png.")
    parser.add_argument("--parts", default="mix", help="Comma-separated mix,melody,vocal,instrumental,chords, or all (audio only).")
    parser.add_argument("--duration", type=float, help="Minimum audio duration in seconds; preserves trailing silence.")
    args = parser.parse_args()
    audio = args.audio
    score = args.score
    if not audio and not score:
        if args.input:
            audio, score = True, "pdf"
        else:
            audio, score = bool(args.midi), "pdf" if args.abc else ""
    if not args.input and not args.midi and not args.abc:
        parser.error("Provide --input, --midi, or --abc.")
    try:
        result = render_outputs(args.input, midi=args.midi, abc=args.abc,
                                output_dir=args.output, audio=audio, score=score,
                                parts=args.parts, duration=args.duration)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Rendering failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
