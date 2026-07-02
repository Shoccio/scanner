import argparse
import os
import sys

import cv2
import numpy as np

# Ignore area thresholds for candidate detection.
# Tune these values to exclude regions outside the answer bubble columns.
TOP_IGNORE_Y = 800
BOTTOM_IGNORE_Y = 3000

# Four-column sheet layout tuning.
# Adjust these pixel coordinates to match the physical column layout on the paper.
COLUMN_SPLIT_X = (530, 1100, 1700)
# Adjust the ignore x-coordinate per column to skip left-side margins or noise.
COLUMN_IGNORE_X = (180, 680, 1360, 1850)
QUESTIONS_PER_COLUMN = 25

# Blank detection threshold based on mean intensity range across one question's bubbles.
# If the mean intensity difference is small, the question is likely unshaded.
BLANK_MEAN_INTENSITY_RANGE = 15.0

# Bubble candidate tuning for darker marks that spill slightly over the bubble outline.
# Slightly tighter preset while still allowing a little spill-over.
BUBBLE_MIN_AREA = 30
BUBBLE_MAX_AREA_RATIO = 0.15
BUBBLE_MIN_SIZE = 8
BUBBLE_MAX_SIZE_RATIO = 0.40
BUBBLE_RATIO_RANGE = (0.30, 3.2)
BUBBLE_CIRCULARITY_RANGE = (0.14, 1.6)
BUBBLE_ELLIPSE_FIT_TOLERANCE = 0.90
BUBBLE_AXIS_RATIO_LIMIT = 2.6


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    return gray, enhanced, cleaned


def detect_sheet_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < 0.08 * image_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            if len(approx) == 4:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, thickness=-1)
                return mask

        largest = contours[0]
        if cv2.contourArea(largest) > 0.08 * image_area:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
            return mask

    _, white_mask = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sheet_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(sheet_contour) > 0.06 * image_area:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [sheet_contour], -1, 255, thickness=-1)
            return mask

    return np.ones(gray.shape, dtype=np.uint8) * 255


def get_column_index(center_x: float) -> int:
    for column_index, threshold in enumerate(COLUMN_SPLIT_X):
        if center_x < threshold:
            return column_index
    return len(COLUMN_SPLIT_X)


def find_candidate_bubbles(binary: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    detection_binary = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detection_binary = cv2.dilate(detection_binary, kernel, iterations=1)

    contours, _ = cv2.findContours(detection_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    candidates = []
    image_width = w

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if y < TOP_IGNORE_Y or y + ch > BOTTOM_IGNORE_Y:
            continue
        center_x = x + cw / 2.0
        column_index = get_column_index(center_x)
        ignore_threshold = COLUMN_IGNORE_X[column_index]
        if x < ignore_threshold:
            continue

        area = cv2.contourArea(contour)
        if area < BUBBLE_MIN_AREA or area > h * w * BUBBLE_MAX_AREA_RATIO:
            continue

        if cw < BUBBLE_MIN_SIZE or ch < BUBBLE_MIN_SIZE or cw > w * BUBBLE_MAX_SIZE_RATIO or ch > h * BUBBLE_MAX_SIZE_RATIO:
            continue

        ratio = cw / float(ch)
        if ratio < BUBBLE_RATIO_RANGE[0] or ratio > BUBBLE_RATIO_RANGE[1]:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < BUBBLE_CIRCULARITY_RANGE[0] or circularity > BUBBLE_CIRCULARITY_RANGE[1]:
            continue

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                major, minor = ellipse[1]
                if minor <= 0 or major <= 0:
                    continue
                axis_ratio = max(major, minor) / min(major, minor)
                if axis_ratio > BUBBLE_AXIS_RATIO_LIMIT:
                    continue
                ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
                if abs(area - ellipse_area) / max(ellipse_area, 1.0) > BUBBLE_ELLIPSE_FIT_TOLERANCE:
                    continue
            except cv2.error:
                pass

        candidates.append((x, y, cw, ch, contour))

    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates


def group_contours_by_rows(candidates: list[tuple[int, int, int, int, np.ndarray]]) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    rows: list[list[tuple[int, int, int, int, np.ndarray]]] = []
    if not candidates:
        return rows

    candidates = sorted(candidates, key=lambda item: item[1])
    current_row = [candidates[0]]
    _, y0, _, h0, _ = candidates[0]
    row_height = h0
    row_center_y = y0 + h0 / 2.0

    for x, y, cw, ch, contour in candidates[1:]:
        center_y = y + ch / 2.0
        threshold = max(row_height * 0.70, 18)
        if abs(center_y - row_center_y) > threshold:
            rows.append(current_row)
            current_row = [(x, y, cw, ch, contour)]
            y0 = y
            row_height = ch
            row_center_y = center_y
        else:
            current_row.append((x, y, cw, ch, contour))
            row_height = max(row_height, ch)
            row_center_y = float(
                (row_center_y * (len(current_row) - 1) + center_y) / len(current_row)
            )

    rows.append(current_row)
    for row in rows:
        row.sort(key=lambda item: item[0])
    return rows


def split_row_into_groups(row: list[tuple[int, int, int, int, np.ndarray]]) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    row = sorted(row, key=lambda item: item[0])
    if len(row) < 4:
        return []

    widths = [cw for _, _, cw, _, _ in row]
    avg_width = float(np.median(widths))
    gap_threshold = max(avg_width * 0.7, 12)

    groups: list[list[tuple[int, int, int, int, np.ndarray]]] = []
    for start in range(len(row) - 3):
        group = row[start : start + 4]
        gaps = [group[i + 1][0] - (group[i][0] + group[i][2]) for i in range(3)]
        if any(gap > gap_threshold * 2.2 for gap in gaps):
            continue

        heights = [h for _, _, _, h, _ in group]
        if max(heights) - min(heights) > max(8, np.median(heights) * 0.45):
            continue

        total_width = (group[-1][0] + group[-1][2]) - group[0][0]
        if total_width > avg_width * 7.5:
            continue

        groups.append(group)

    unique_group_keys: set[tuple[tuple[int, int, int, int], ...]] = set()
    unique_groups: list[list[tuple[int, int, int, int, np.ndarray]]] = []
    for group in groups:
        group_key = tuple((int(x), int(y), int(cw), int(ch)) for x, y, cw, ch, _ in group)
        if group_key not in unique_group_keys:
            unique_group_keys.add(group_key)
            unique_groups.append(group)
    return unique_groups


def get_question_clusters(
    rows: list[list[tuple[int, int, int, int, np.ndarray]]],
    image_width: int,
) -> list[tuple[int, list[tuple[int, int, int, int, np.ndarray]], tuple[int, int, int, int]]]:
    column_groups: list[list[tuple[list[tuple[int, int, int, int, np.ndarray]], tuple[int, int, int, int]]]] = [
        [] for _ in range(len(COLUMN_SPLIT_X) + 1)
    ]

    for row in rows:
        groups = split_row_into_groups(row)
        for group in groups:
            if len(group) != 4:
                continue

            group = sorted(group, key=lambda item: item[0])
            x0 = min(x for x, _, _, _, _ in group)
            y0 = min(y for _, y, _, _, _ in group)
            x1 = max(x + cw for x, _, cw, _, _ in group)
            y1 = max(y + ch for _, y, _, ch, _ in group)
            center_x = (x0 + x1) / 2.0
            column_index = get_column_index(center_x)
            column_groups[column_index].append((group, (x0, y0, x1, y1)))

    clusters: list[tuple[int, list[tuple[int, int, int, int, np.ndarray]], tuple[int, int, int, int]]] = []
    for column_index, groups in enumerate(column_groups):
        groups.sort(key=lambda item: (item[1][1], item[1][0]))
        question_number = column_index * QUESTIONS_PER_COLUMN + 1
        for group, bbox in groups:
            clusters.append((question_number, group, bbox))
            question_number += 1

    return clusters


def build_answer_grid(
    rows: list[list[tuple[int, int, int, int, np.ndarray]]],
    gray: np.ndarray,
    binary: np.ndarray,
    image_width: int,
    color: np.ndarray,
    debug: bool = False,
) -> list[tuple[int, str, float]]:
    answers = []
    letter_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    question_number = 1

    clusters = get_question_clusters(rows, image_width)
    for question_number, group, _ in clusters:
        choice_scores = []
        debug_rows: list[str] = []

        for letter_index, (x, y, cw, ch, contour) in enumerate(group):
            mask = np.zeros((ch, cw), dtype=np.uint8)
            contour_shifted = contour - np.array([[x, y]])
            cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=-1)

            roi_gray = gray[y : y + ch, x : x + cw]
            roi_binary = binary[y : y + ch, x : x + cw]
            roi_color = color[y : y + ch, x : x + cw]

            masked_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=mask)
            masked_binary = cv2.bitwise_and(roi_binary, roi_binary, mask=mask)

            inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            if cv2.countNonZero(inner_mask) < 0.3 * cv2.countNonZero(mask):
                inner_mask = mask

            masked_inner_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=inner_mask)
            total_pixels = cv2.countNonZero(inner_mask)
            if total_pixels == 0:
                continue

            mean_intensity = float(np.sum(masked_inner_gray)) / total_pixels
            thresh = float(np.mean(masked_inner_gray[inner_mask == 255]))
            dark_pixels = int(np.count_nonzero(masked_inner_gray < thresh))
            dark_ratio = float(dark_pixels) / total_pixels
            fill_score = (255.0 - mean_intensity) / 255.0

            hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            masked_saturation = cv2.bitwise_and(hsv[:, :, 1], hsv[:, :, 1], mask=mask)
            mean_saturation = float(np.sum(masked_saturation)) / max(cv2.countNonZero(mask), 1)
            blue_score = mean_saturation / 255.0

            choice_scores.append((x, fill_score, mean_intensity, dark_ratio, blue_score))

            if debug:
                debug_rows.append(
                    f"  {letter_map[letter_index]}: mean_intensity={mean_intensity:.1f}, "
                    f"fill_score={fill_score:.3f}, dark_ratio={dark_ratio:.3f}, blue={blue_score:.3f}"
                )

        if not choice_scores:
            continue

        choice_scores.sort(key=lambda item: item[0])
        mean_intensities = [item[2] for item in choice_scores]
        fill_scores = [item[1] for item in choice_scores]
        intensity_range = float(np.max(mean_intensities) - np.min(mean_intensities))
        max_fill = float(np.max(fill_scores))
        second_fill = float(sorted(fill_scores, reverse=True)[1]) if len(fill_scores) > 1 else 0.0
        fill_diff = max_fill - second_fill

        is_blank = intensity_range < BLANK_MEAN_INTENSITY_RANGE

        if is_blank:
            answer_letter = "?"
            confidence = 0.0
        else:
            best_choice = min(choice_scores, key=lambda item: item[2])
            best_index = choice_scores.index(best_choice)
            answer_letter = letter_map[best_index] if best_index < len(letter_map) else "?"
            confidence = float(best_choice[1])

        if debug:
            print(
                f"Q{question_number:02d}: {answer_letter} debug values (range={intensity_range:.1f}, "
                f"max_fill={max_fill:.3f}, diff={fill_diff:.3f}):"
            )
            for line in debug_rows:
                print(line)

        answers.append((question_number, answer_letter, confidence))

    return answers


def draw_debug_overlay(image: np.ndarray, rows: list[list[tuple[int, int, int, int, np.ndarray]]]) -> np.ndarray:
    overlay = image.copy()
    for row in rows:
        for x, y, cw, ch, contour in row:
            cv2.rectangle(overlay, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read shaded multiple-choice bubbles from a phone photo.")
    parser.add_argument("image", help="Path to the test sheet image file.")
    parser.add_argument("--debug", action="store_true", help="Save a debug overlay image next to the input file.")
    parser.add_argument("--output", default=None, help="Save the extracted answers to a text file.")
    return parser.parse_args()


def extract_answers(image_path: str, debug: bool = False) -> list[tuple[int, str, float]]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot load image: {image_path}")

    sheet_mask = detect_sheet_mask(image)
    masked_image = cv2.bitwise_and(image, image, mask=sheet_mask)

    gray, enhanced, binary_image = preprocess_image(masked_image)
    candidates = find_candidate_bubbles(binary_image)
    rows = group_contours_by_rows(candidates)
    clusters = get_question_clusters(rows, image.shape[1])

    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_questions")
        os.makedirs(debug_dir, exist_ok=True)
        for question_number, _, (x0, y0, x1, y1) in clusters:
            x0_expanded = max(0, x0 - 300)
            crop = image[y0:y1, x0_expanded:x1]
            debug_path = os.path.join(debug_dir, f"question_{question_number:02d}.png")
            cv2.imwrite(debug_path, crop)
        debug_image = draw_debug_overlay(image, rows)
        debug_path = os.path.splitext(image_path)[0] + "_debug.png"
        cv2.imwrite(debug_path, debug_image)
        print(f"Saved question clusters to {debug_dir}")
        print(f"Saved debug overlay: {debug_path}")

    answers = build_answer_grid(rows, gray, binary_image, image.shape[1], masked_image, debug)
    return answers


def main() -> int:
    args = parse_args()
    try:
        answers = extract_answers(args.image, args.debug)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(answers)

    if not answers:
        print("No candidate bubbles were found. Try a clearer crop or increase contrast on the image.")
        return 1

    print("Detected shaded answers:")
    for question_number, answer_letter, confidence in answers:
        print(f"Q{question_number:02d}: {answer_letter} (fill score: {confidence:.3f})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            for question_number, answer_letter, confidence in answers:
                output_file.write(f"Q{question_number:02d}: {answer_letter} (score: {confidence:.3f})\n")
        print(f"Saved answers to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
