import argparse
import csv
import json
from pathlib import Path

def box_center(box):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2, (y0 + y1) / 2)

def point_in_box(pt, box):
    x, y = pt
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1

def box_intersects(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    total = skipped = analyzable = with_flip = 0
    hit1 = intersect1 = 0
    rows = []

    for line in open(args.input):
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1

        if r.get("skipped"):
            skipped += 1
            continue

        analyzable += 1
        bbox = r.get("bbox")
        patches = r.get("patches", [])
        flipped = [p for p in patches if p.get("flip") == 1]

        if not flipped:
            rows.append({
                "image_id": r.get("image_id"),
                "pathology": r.get("pathology"),
                "num_flips": 0,
                "hit1": 0,
                "intersect1": 0,
            })
            continue

        with_flip += 1

        # 目前 flip 是 binary，没有 score。Hit@1 取第一个 flipped patch。
        top = flipped[0]
        patch_box = top["box"]
        center = box_center(patch_box)

        h = int(point_in_box(center, bbox))
        inter = int(box_intersects(patch_box, bbox))

        hit1 += h
        intersect1 += inter

        rows.append({
            "image_id": r.get("image_id"),
            "pathology": r.get("pathology"),
            "num_flips": len(flipped),
            "top_patch_gx": top.get("gx"),
            "top_patch_gy": top.get("gy"),
            "top_patch_box": patch_box,
            "bbox": bbox,
            "hit1": h,
            "intersect1": inter,
        })

    hit1_rate = hit1 / with_flip if with_flip else 0
    intersect1_rate = intersect1 / with_flip if with_flip else 0
    flip_case_rate = with_flip / analyzable if analyzable else 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("CRS metrics")
    print("total:", total)
    print("skipped:", skipped)
    print("analyzable:", analyzable)
    print("with_flip:", with_flip)
    print(f"flip_case_rate: {flip_case_rate:.4f}")
    print(f"hit1_rate_center_inside_bbox: {hit1_rate:.4f}")
    print(f"intersect1_rate_patch_intersects_bbox: {intersect1_rate:.4f}")
    print("saved ->", args.output)

if __name__ == "__main__":
    main()
