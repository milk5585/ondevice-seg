"""
calibration / 평가용 이미지 목록을 파일로 고정한다 (C가 한 번만 실행).
만들어진 lists/ 폴더를 Git에 올리면 셋이 같은 이미지를 쓰게 된다.

    python make_lists.py
"""

import random

from common.voc import LIST_DIR, PERSON_CLASS, load_label, read_ids

N_CALIB = 300
SEED = 0


def main() -> None:
    train, val = read_ids("train"), read_ids("val")
    overlap = set(train) & set(val)
    assert not overlap, f"train과 val이 겹칩니다: {len(overlap)}장"

    def with_person(ids):
        return [i for i in ids if (load_label(i) == PERSON_CLASS).any()]

    train_person = with_person(train)
    val_person = with_person(val)

    # 정렬 후 고정 시드로 뽑아서 누가 돌려도 같은 목록이 나오게 한다
    calib = sorted(random.Random(SEED).sample(sorted(train_person), N_CALIB))

    LIST_DIR.mkdir(exist_ok=True)
    (LIST_DIR / f"calib_train_person{N_CALIB}.txt").write_text("\n".join(calib) + "\n")
    (LIST_DIR / "val_person.txt").write_text("\n".join(val_person) + "\n")

    print(f"train {len(train)}장 (person 포함 {len(train_person)}장)")
    print(f"val   {len(val)}장 (person 포함 {len(val_person)}장)")
    print(f"calibration: train person 중 {N_CALIB}장 → lists/calib_train_person{N_CALIB}.txt")
    print("train/val 겹침 없음 확인")


if __name__ == "__main__":
    main()
