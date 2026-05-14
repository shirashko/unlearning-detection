"""Re-run only the Gemini judge step against an existing audit dir.

Reads ``<audit_dir>/judge_prompt.txt``, calls Gemini, and writes:
  - <audit_dir>/judge_response_raw.txt
  - <audit_dir>/judge_response.json   (only if the response parses as JSON)

It also patches ``<audit_dir>/audit_summary.json`` in place: clears the old
``judge_error`` and fills in ``judge_verdict`` so the summary is consistent
with the new response.

Usage:
  /home/morg/students/rashkovits/envs/snmf_env/bin/python \
      scripts/audit/rerun_judge.py \
      --audit-dir outputs/wmdp/audit_general/<run-folder> \
      [--model gemini-2.5-flash]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", required=True, type=Path,
                   help="Existing audit run dir containing judge_prompt.txt.")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Gemini model id. Default: gemini-2.5-flash "
                        "(use gemini-3-flash if your key has access).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=1500)
    p.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    p.add_argument(
        "--env-file", type=Path,
        default=Path("/home/morg/students/rashkovits/snmf/.env"),
        help="Path to the .env file holding GOOGLE_API_KEY.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    audit_dir: Path = args.audit_dir.resolve()
    prompt_path = audit_dir / "judge_prompt.txt"
    raw_path = audit_dir / "judge_response_raw.txt"
    parsed_path = audit_dir / "judge_response.json"
    summary_path = audit_dir / "audit_summary.json"

    if not prompt_path.is_file():
        print(f"ERROR: {prompt_path} not found.", file=sys.stderr)
        return 1

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set "
              f"(checked env + {args.env_file}).", file=sys.stderr)
        return 1

    prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Loaded prompt: {len(prompt)} chars from {prompt_path}")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai not installed. "
              "Install with: pip install google-genai", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    print(f"Calling {args.model} (temperature={args.temperature}, "
          f"max_output_tokens={args.max_output_tokens})...")
    resp = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    text = getattr(resp, "text", "") or ""
    raw_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} chars to {raw_path}")

    if not text:
        print("WARNING: empty response from Gemini.", file=sys.stderr)
        return 2

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"WARNING: response is not valid JSON: {e}")
        print(text[:500])
        return 2

    parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"Wrote parsed verdict to {parsed_path}")
    print(f"  unlearning_confidence    = {parsed.get('unlearning_confidence')}")
    print(f"  likely_unlearned_concept = {parsed.get('likely_unlearned_concept')!r}")

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["judge_verdict"] = parsed
        summary["judge_error"] = None
        meta = summary.setdefault("meta", {})
        meta["judge_model"] = args.model
        meta["judge_temperature"] = args.temperature
        meta["judge_max_output_tokens"] = args.max_output_tokens
        meta["judge_skipped"] = False
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Patched judge fields in {summary_path}")
    else:
        print(f"NOTE: {summary_path} not found; skipping summary patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
