"""Transcribe a music recording with SheetSage2."""
import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(description="SheetSage2: music audio to ABC, MIDI, annotations and features")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    local = SCRIPT_DIR
    parser.add_argument("--model", default=str(local) if (local / "config.json").exists() else "m-a-p/SheetSage2")
    parser.add_argument("--revision", help="Model and code commit on Hugging Face")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--lookahead", type=float)
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--melody-only", action="store_true",
                        help="Keep vocal and instrumental melodies; omit chords from ABC and playback")
    parser.add_argument("--export-logits", action="store_true")
    parser.add_argument("--export-scores", action="store_true")
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--all-layers", action="store_true", help="Also export all 24 MERT block features")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--render-score", nargs="?", const="pdf", default=False, help="pdf,svg,png (default: pdf)")
    parser.add_argument("--render-parts", default="mix", help="mix,melody,vocal,instrumental,chords,all")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.render_audio or args.render_score:
        from rendering_sheetsage2 import validate_render_options
        try:
            validate_render_options(audio=args.render_audio, score=args.render_score, parts=args.render_parts)
        except ValueError as exc:
            parser.error(str(exc))
    import torch
    from transformers import AutoModel
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if args.device != "auto":
        device = args.device
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, code_revision=args.revision,
        local_files_only=args.local_files_only, trust_remote_code=True,
    ).eval().to(device)
    def progress(value):
        if value["stage"] == "encoding":
            print(f"Window {value['window']}/{value['windows']}", flush=True)
    options = dict(dtype=args.dtype, preset=args.preset, max_seconds=args.max_seconds,
                   overlap_seconds=args.overlap, lookahead_seconds=args.lookahead,
                   export_logits=args.export_logits, export_scores=args.export_scores,
                   export_embeddings=args.export_embeddings, output_hidden_states=args.all_layers,
                   render_audio=args.render_audio, render_score=args.render_score,
                   render_parts=tuple(args.render_parts.split(",")), progress=progress)
    if args.prompts:
        options["prompts"] = args.prompts
    if args.melody_only:
        options["melody_only"] = True
    try:
        result = model.transcribe(args.audio, output_dir=args.output, **options)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"SheetSage2: {exc}\n")
    if args.melody_only and (result.get("abc_error") or not result.get("abc")):
        reason = result.get("abc_error") or "no ABC score was produced"
        parser.exit(1, f"SheetSage2: melody-only ABC unavailable: {reason}. "
                       f"Transcription outputs were saved to {args.output.resolve()}.\n")
    print(f"Saved transcription to {args.output.resolve()}")
    if result.get("abc_error"):
        print(f"ABC unavailable: {result['abc_error']}. MIDI and annotations were saved.")


if __name__ == "__main__":
    main()
