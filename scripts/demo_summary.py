"""Use the retained direct-summary prompt and adapter on a fictional or local dialogue."""
import argparse
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from qa_summary_study_common import stage2_input
from validate_qa_aware_pipeline import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/synthetic_dialogue.json")
    parser.add_argument("--prompt-only", action="store_true", help="Print prompt without GPU or weights")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    prompt = stage2_input(source)
    if args.prompt_only:
        print(prompt)
        return
    os.chdir(ROOT)
    from csds_handoff128 import handoff_common as h
    from peft import PeftModel
    if not h.SUMMARY_ADAPTER.is_dir():
        parser.error(f"Restore the retained adapter at {h.SUMMARY_ADAPTER}")
    if not h.BASE_MODEL.is_dir():
        parser.error(f"Restore the exact base model at {h.BASE_MODEL}")
    tokenizer = h.get_tokenizer()
    model = PeftModel.from_pretrained(h.load_base(), str(h.SUMMARY_ADAPTER), is_trainable=False)
    model.eval()
    summary, tokens = generate(model, tokenizer, prompt, h.DECODE_MAX, "demo")
    print(json.dumps({"summary": summary, "generated_tokens": tokens}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
