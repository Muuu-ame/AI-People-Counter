
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from collections import defaultdict, deque
from statistics import median
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor
from deepface import DeepFace


# ========================================
# OSNet：人物の特徴量を抽出
# ========================================
def extract_feature(extractor, frame, box):
    x1, y1, x2, y2 = map(int, box)

    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 20:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    tensor = extractor.preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = extractor.model(tensor.to(extractor.device))
        features = F.normalize(features, p=2, dim=1)

    return features[0].cpu().numpy()


# ========================================
# 特徴量の類似度を計算
# ========================================
def cosine_similarity(a, b):
    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


# ========================================
# 年齢層を分類
# ========================================
def get_age_group(age):
    if age < 20:
        return "Under 20"
    elif age < 40:
        return "20-39"
    elif age < 60:
        return "40-59"
    else:
        return "60+"


# ========================================
# 年齢推定
# ========================================
def estimate_age(frame, box):
    x1, y1, x2, y2 = map(int, box)

    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:
        return None

    try:
        results = DeepFace.analyze(
            img_path=person_crop,
            actions=["age"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True
        )

        if not results:
            return None

        # 顔が複数見つかった場合は、最大の顔を使う
        valid_faces = [
            result for result in results
            if result.get("region", {}).get("w", 0) >= 40
            and result.get("region", {}).get("h", 0) >= 40
        ]

        if not valid_faces:
            return None

        face = max(
            valid_faces,
            key=lambda result:
                result["region"]["w"] * result["region"]["h"]
        )

        age = face.get("age")

        if age is None:
            return None

        return float(age)

    except Exception:
        # 顔を検出できない場合などは年齢不明
        return None


# ========================================
# メイン処理
# ========================================
def main():
    print("YOLOを読み込み中...")
    detector = YOLO("yolo11n.pt")

    print("OSNetを読み込み中...")
    extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        device="cpu"
    )

    print("年齢推定モデルを読み込み中...")
    DeepFace.build_model(
        task="facial_attribute",
        model_name="Age"
    )
    print("モデルの読み込み完了")

    # カメラは main.py だけで開く
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    # ====================================
    # 人物管理
    # ====================================
    gallery = {}
    track_to_person = {}

    pending = defaultdict(lambda: deque(maxlen=5))

    next_person_id = 1
    frame_number = 0

    # 再識別設定
    MATCH_THRESHOLD = 0.72
    CONFIRM_FRAMES = 3
    LOST_FRAMES = 1800
    MAX_TEMPLATES = 5
    FEATURE_INTERVAL = 5

    # 年齢推定設定
    age_history = defaultdict(lambda: deque(maxlen=5))
    last_age_frame = {}
    AGE_INTERVAL = 45

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("映像を取得できませんでした")
                break

            frame_number += 1

            # --------------------------------
            # YOLOによる人物検出・追跡
            # --------------------------------
            result = detector.track(
                frame,
                persist=True,
                tracker="botsort_reid.yaml",
                classes=[0],
                conf=0.4,
                imgsz=512,
                verbose=False
            )[0]

            visible_people = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    person_id = track_to_person.get(track_id)

                    # --------------------------------
                    # 既知人物の特徴量を更新
                    # --------------------------------
                    if person_id is not None:
                        if frame_number % FEATURE_INTERVAL == 0:
                            feature = extract_feature(
                                extractor, frame, box
                            )

                            if feature is not None:
                                data = gallery[person_id]
                                data["last_seen"] = frame_number

                                best = max(
                                    cosine_similarity(feature, old)
                                    for old in data["templates"]
                                )

                                if best < 0.98:
                                    data["templates"].append(feature)

                                    if len(data["templates"]) > MAX_TEMPLATES:
                                        data["templates"].pop(0)

                    # --------------------------------
                    # 未登録トラックの再識別
                    # --------------------------------
                    else:
                        feature = None

                        if frame_number % FEATURE_INTERVAL == 0:
                            feature = extract_feature(
                                extractor, frame, box
                            )

                        if feature is None:
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2),
                                (0, 165, 255), 2
                            )
                            cv2.putText(
                                frame, "Checking...",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2
                            )
                            continue

                        best_id = None
                        best_score = -1.0

                        for pid, data in gallery.items():
                            if (
                                frame_number - data["last_seen"]
                                > LOST_FRAMES
                            ):
                                continue

                            score = max(
                                cosine_similarity(feature, old)
                                for old in data["templates"]
                            )

                            if score > best_score:
                                best_score = score
                                best_id = pid

                        candidate = (
                            best_id
                            if best_score >= MATCH_THRESHOLD
                            else None
                        )

                        pending[track_id].append(candidate)
                        recent = list(pending[track_id])

                        valid = [
                            pid for pid in recent
                            if pid is not None
                        ]

                        confirmed_id = None

                        if valid:
                            counts = {
                                pid: valid.count(pid)
                                for pid in set(valid)
                            }

                            top_id = max(counts, key=counts.get)

                            if counts[top_id] >= CONFIRM_FRAMES:
                                confirmed_id = top_id

                        if confirmed_id is not None:
                            person_id = confirmed_id
                            track_to_person[track_id] = person_id
                            pending.pop(track_id, None)

                            gallery[person_id]["last_seen"] = frame_number

                            print(
                                f"再識別: Person {person_id} "
                                f"(similarity={best_score:.3f})"
                            )

                        elif len(recent) >= 5:
                            person_id = next_person_id
                            next_person_id += 1

                            gallery[person_id] = {
                                "templates": [feature.copy()],
                                "last_seen": frame_number
                            }

                            track_to_person[track_id] = person_id
                            pending.pop(track_id, None)

                            print(f"新しい人物: Person {person_id}")

                        else:
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2),
                                (0, 165, 255), 2
                            )
                            cv2.putText(
                                frame, "Checking...",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2
                            )
                            continue

                    # --------------------------------
                    # 年齢推定
                    # 1人につき45フレームごと
                    # --------------------------------
                    if (
                        person_id not in last_age_frame
                        or frame_number - last_age_frame[person_id]
                        >= AGE_INTERVAL
                    ):
                        last_age_frame[person_id] = frame_number

                        age = estimate_age(frame, box)

                        if age is not None:
                            age_history[person_id].append(age)

                    # --------------------------------
                    # 表示
                    # --------------------------------
                    visible_people += 1

                    if age_history[person_id]:
                        estimated_age = median(
                            age_history[person_id]
                        )
                        age_group = get_age_group(estimated_age)

                        label = (
                            f"Person {person_id} | "
                            f"{age_group} (~{estimated_age:.0f})"
                        )
                    else:
                        label = (
                            f"Person {person_id} | Age: Unknown"
                        )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            # --------------------------------
            # 画面上部の情報
            # --------------------------------
            cv2.putText(
                frame,
                f"Registered people: {next_person_id - 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.putText(
                frame,
                f"Visible people: {visible_people}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.imshow("AI People Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()