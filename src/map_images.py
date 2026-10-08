# -*- coding: utf-8 -*-
"""为锚定样本解析 iNaturalist 图像路径。

背景（踩坑记录）
==================================================================
E-VQA 有三个和图像有关的字段，含义完全不同，很容易用错：

  dataset_category_id   6513 -> iNat **类别**编号。**必须零填充**成 5 位
                        （'6513' -> '06513'）才能对上 val.json 的 category id。
                        直接比较会得到 0% 交集，让人误判"映射方案不可行"。
  dataset_image_ids     ['1083578', ...] -> iNaturalist 上的**原始照片 ID**，
                        不在竞赛发布的 100000 张里，**无法用来定位文件**。
  dataset_name          'inaturalist'

正确的定位链条（三段都要，缺一不可）：

  wikipedia_title  --(学名一致, 实测 100%)-->  val.json.categories.name
      -> categories.image_dir_name
      -> images.file_name（100000 条 image_id -> 文件路径）

因为类别已精确对应，**同一目录下所有图都是同一物种**，取第一张即可
（实测 1983 个所需类别 100% 有图，共 19830 张，每类 10 张）。
不需要、也无法用 dataset_image_ids 精确到某一张。

用法：
  python map_images.py --anchors ../data/raw/evqa_anchors_raw.jsonl \
                       --out ../data/medusa_anchors.jsonl \
                       --inat ../data/inat_val
"""
import argparse, collections, io, json, os, random, sys


def build_index(inat_root):
    """val.json -> {category_id: dir_name} 与每类可用图像列表。"""
    vj = os.path.join(inat_root, "val.json")
    with io.open(vj, encoding="utf-8") as f:
        d = json.load(f)
    cat2dir = {str(c["id"]): c["image_dir_name"] for c in d["categories"]}
    cat2name = {str(c["id"]): c["name"] for c in d["categories"]}
    base = os.path.join(inat_root, "val")
    dir2imgs = {}
    for dn in os.listdir(base):
        p = os.path.join(base, dn)
        if not os.path.isdir(p):
            continue
        dir2imgs[dn] = sorted(
            f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return cat2dir, cat2name, dir2imgs, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_raw.jsonl")
    ap.add_argument("--out", default="../data/medusa_anchors.jsonl")
    ap.add_argument("--inat", default="../data/inat_val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="../results/image_mapping_report.json")
    args = ap.parse_args()

    cat2dir, cat2name, dir2imgs, base = build_index(args.inat)
    print("iNat: %d 类别, %d 目录有图" % (len(cat2dir), len(dir2imgs)))

    rows = [json.loads(l) for l in io.open(args.anchors, encoding="utf-8") if l.strip()]
    print("锚定样本 %d 条" % len(rows))

    rng = random.Random(args.seed)
    out, missing, name_mismatch = [], [], 0
    for r in rows:
        cid = str(r["dataset_category_id"])
        dn = cat2dir.get(cid)
        if not dn:
            missing.append((r["wikipedia_title"], cid, "category 未找到"))
            continue
        # 学名校验：E-VQA 的 wikipedia_title 应与 iNat 类别学名一致
        if cat2name.get(cid) != r["wikipedia_title"]:
            name_mismatch += 1
        imgs = dir2imgs.get(dn, [])
        if not imgs:
            missing.append((r["wikipedia_title"], cid, "目录无图"))
            continue
        # 固定种子选一张，可复现；同目录同物种，选哪张科学上无差别
        img = rng.choice(imgs)
        r = dict(r)
        r["image"] = os.path.join(base, dn, img)
        r["inat_dir"] = dn
        r["inat_image_file"] = img
        r["inat_n_images_in_category"] = len(imgs)
        out.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rep = {
        "n_anchors": len(rows),
        "n_with_image": len(out),
        "coverage": round(len(out) / float(len(rows) or 1), 4),
        "name_mismatch": name_mismatch,
        "missing": missing[:20],
        "n_missing": len(missing),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    json.dump(rep, io.open(args.report, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 56)
    print("  图像映射结果")
    print("=" * 56)
    print("  锚定样本        %d" % len(rows))
    print("  成功配图        %d  (%.1f%%)" % (len(out), 100 * rep["coverage"]))
    print("  学名不一致      %d  <= 应为 0，非 0 需人工查" % name_mismatch)
    print("  配图失败        %d" % len(missing))
    for m in missing[:5]:
        print("     ", m)
    print()
    print("  每类图像数 分布:",
          dict(collections.Counter(r["inat_n_images_in_category"] for r in out).most_common(5)))
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
