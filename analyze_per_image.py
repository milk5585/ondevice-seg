"""per_image.csv → 사람 면적·사람 수별 person IoU / Boundary IoU (520 vs 384). 구간 안에서는 픽셀을 모두 더해 한 번 나눔."""
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent / "results" / "per_image.csv", encoding="utf-8")))


def area_bucket(r):
    f = float(r["person_frac"])
    return "작은 인물 (면적 5% 미만)" if f < 0.05 else ("중간 (5~20%)" if f < 0.20 else "큰 인물 (20% 이상)")


def count_bucket(r):
    n = int(r["n_persons"])
    return "1명" if n <= 1 else ("2~3명" if n <= 3 else "4명 이상")


def table(key_fn, order):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    imgs = defaultdict(set)
    for r in rows:
        k = key_fn(r)
        a = acc[k][r["size"]]
        a[0] += int(r["p_inter"]); a[1] += int(r["p_union"]); a[2] += int(r["b_inter"]); a[3] += int(r["b_union"])
        imgs[k].add(r["id"])
    out = []
    for k in order + ["전체"]:
        if k == "전체":
            tot = defaultdict(lambda: [0, 0, 0, 0])
            for kk in acc:
                for s, v in acc[kk].items():
                    for j in range(4):
                        tot[s][j] += v[j]
            d, n = tot, len({r["id"] for r in rows})
        else:
            d, n = acc[k], len(imgs[k])
        p520, p384 = d["520"][0] / d["520"][1] * 100, d["384"][0] / d["384"][1] * 100
        b520, b384 = d["520"][2] / d["520"][3] * 100, d["384"][2] / d["384"][3] * 100
        out.append((k, n, p520, p384, p384 - p520, b520, b384, b384 - b520))
    return out


if __name__ == "__main__":
    for title, fn, order in [("사람 면적", area_bucket, ["작은 인물 (면적 5% 미만)", "중간 (5~20%)", "큰 인물 (20% 이상)"]),
                             ("사람 수", count_bucket, ["1명", "2~3명", "4명 이상"])]:
        print(f"\n[{title}]  구간 | 장수 | person IoU 520 → 384 (차이) | Boundary IoU 520 → 384 (차이)")
        for k, n, p5, p3, dp, b5, b3, db in table(fn, order):
            print(f"  {k:22s} {n:4d}장 | {p5:6.2f} → {p3:6.2f} ({dp:+.2f}) | {b5:6.2f} → {b3:6.2f} ({db:+.2f})")
