"""Rebuild results/viz/bit2bit_viewer*.html from the saved frames.json (no retraining)."""
from common import *
import json
VIZ = os.path.join(ROOT, "results", "viz")
tpl = open(os.path.join(ROOT, "python", "tdnet", "viz_template.html")).read()
blob = open(os.path.join(VIZ, "frames.json")).read()
page = tpl.replace("/*DATA*/{}", blob)
open(os.path.join(VIZ, "bit2bit_viewer_body.html"), "w").write(page)
open(os.path.join(VIZ, "bit2bit_viewer.html"), "w").write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n' + page + "\n</body></html>")
print("rebuilt", os.path.getsize(os.path.join(VIZ, "bit2bit_viewer.html")) / 1e6, "MB")
