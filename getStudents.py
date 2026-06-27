import argparse
import json
import os
from typing import Any

from main import extract_answers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"}
ANSWER_KEY_FILE = "answers.txt"


def collect_student_images(student_folder: str) -> list[str]:
    images = []
    for root, _, files in os.walk(student_folder):
        for file_name in sorted(files):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, file_name))
    return images


def load_answer_key(answer_file: str = ANSWER_KEY_FILE) -> dict[int, str]:
    answers = {}
    if not os.path.isfile(answer_file):
        raise FileNotFoundError(f"Answer key file not found: {answer_file}")

    with open(answer_file, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            raw = raw.rstrip(",")
            raw = raw.strip()
            if not raw:
                continue
            answer_letter = raw.split(",")[0].strip().upper()
            if not answer_letter:
                continue
            answers[index] = answer_letter
    return answers


def compare_answers(student_answers: list[tuple[int, str, float]], key: dict[int, str]) -> tuple[int, int, list[tuple[int, str, bool]]]:
    correct = 0
    total = len(key)
    details: list[tuple[int, str, bool]] = []

    for question_number in sorted(key.keys()):
        correct_letter = key[question_number]
        student_letter = next(
            (answer for qnum, answer, _ in student_answers if qnum == question_number),
            "?",
        )
        is_correct = student_letter == correct_letter
        if is_correct:
            correct += 1
        details.append((question_number, student_letter, is_correct))

    return correct, total, details


def scan_students(students_root: str, debug: bool = False) -> list[dict[str, Any]]:
    answer_key = load_answer_key()
    entries: list[dict[str, Any]] = []
    for student_name in sorted(os.listdir(students_root)):
        student_folder = os.path.join(students_root, student_name)
        if not os.path.isdir(student_folder):
            continue

        image_paths = collect_student_images(student_folder)
        if not image_paths:
            print(f"Skipping {student_name}: no sheet images found")
            continue

        for image_path in image_paths:
            try:
                answers = extract_answers(image_path, debug=debug)
                correct, total, details = compare_answers(answers, answer_key)
            except Exception as exc:
                print(f"Error scanning {image_path}: {exc}")
                continue

            entries.append(
                {
                    "Student": student_name,
                    "Image": image_path,
                    "Score": f"{correct}/{total}",
                    "Answers": [(student_letter, is_correct) for _, student_letter, is_correct in details],
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan student folders for sheet answer images.")
    parser.add_argument("students_root", help="Path to the root students folder.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and save debug files.")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to write the JSON results file. Defaults to students_root/results.json.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.students_root):
        print(f"Error: root folder not found: {args.students_root}")
        return 1

    results = scan_students(args.students_root, debug=args.debug)
    if not results:
        print("No student sheet images were processed.")
        return 1

    output_path = args.json_output or os.path.join(args.students_root, "results.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print(f"Saved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
