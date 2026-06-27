import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Tuple

import getStudents

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_score(score: str) -> Tuple[int, int]:
    if "/" not in score:
        raise ValueError(f"Invalid score format: {score}")
    correct_str, total_str = score.split("/", 1)
    return int(correct_str), int(total_str)


def generate_statistics(
    students_root: str,
    debug: bool = False,
) -> Dict[str, Any]:
    student_results = getStudents.scan_students(students_root, debug=debug)
    if not student_results:
        return {
            "students": [],
            "mean_score": None,
            "median_score": None,
            "mode_score": [],
            "question_correct_counts": {},
        }

    scores = []
    question_correct_counts: Dict[int, int] = {}
    question_students: Dict[int, List[str]] = {}
    students: List[Dict[str, Any]] = []

    for result in student_results:
        score_value, score_total = parse_score(result["Score"])
        percentage = (score_value / score_total) * 100 if score_total else 0.0
        scores.append(percentage)

        students.append(
            {
                "Student": result["Student"],
                "Score": result["Score"],
                "ScorePercentage": round(percentage, 2),
                "Image": result.get("Image"),
                "Answers": result["Answers"],
            }
        )

        for question_index, answer_entry in enumerate(result["Answers"], start=1):
            if isinstance(answer_entry, (list, tuple)) and len(answer_entry) >= 3:
                question_number = answer_entry[0]
                is_correct = answer_entry[2]
            else:
                question_number = question_index
                is_correct = answer_entry[1] if isinstance(answer_entry, (list, tuple)) and len(answer_entry) >= 2 else False

            question_correct_counts.setdefault(question_number, 0)
            question_students.setdefault(question_number, [])
            if is_correct:
                question_correct_counts[question_number] += 1
                question_students[question_number].append(result["Student"])

    answer_key = getStudents.load_answer_key()
    total_questions = len(answer_key)

    question_stats: Dict[str, Dict[str, Any]] = {
        str(question_index): {
            "correct_cnt": question_correct_counts.get(question_index, 0),
            "students": question_students.get(question_index, []),
        }
        for question_index in range(1, total_questions + 1)
    }

    mean_score = round(statistics.mean(scores), 2)
    median_score = round(statistics.median(scores), 2)
    mode_scores = statistics.multimode(scores)
    mode_scores = [round(value, 2) for value in sorted(set(mode_scores))]

    max_score = max(scores)
    min_score = min(scores)
    highest_students = [
        {"Student": student["Student"], "Score": student["Score"]}
        for student in students
        if student["ScorePercentage"] == round(max_score, 2)
    ]
    lowest_students = [
        {"Student": student["Student"], "Score": student["Score"]}
        for student in students
        if student["ScorePercentage"] == round(min_score, 2)
    ]

    stats = {
        "students": students,
        "mean_score": mean_score,
        "median_score": median_score,
        "mode_score": mode_scores,
        "highest_students": highest_students,
        "lowest_students": lowest_students,
        "question_correct_counts": question_correct_counts,
        "question_stats": question_stats,
        "total_questions": total_questions,
    }

    return stats


def save_question_bar_chart(
    question_correct_counts: Dict[int, int],
    total_questions: int,
    output_path: str,
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping chart generation.")
        return

    question_numbers = list(range(1, total_questions + 1))
    correct_counts = [question_correct_counts.get(q, 0) for q in question_numbers]

    plt.figure(figsize=(12, 6))
    plt.bar(question_numbers, correct_counts, color="#4c72b0")
    plt.xlabel("Question Number")
    plt.ylabel("Correct Count")
    plt.title("Correct Answers per Question")
    plt.xticks(question_numbers)
    y_max = max(total_questions, max(correct_counts) if correct_counts else 0)
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute statistics for student answer sheets.")
    parser.add_argument("students_root", help="Path to the root students folder.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output while scanning sheets.")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to save statistics JSON. Defaults to students_root/statistics.json.",
    )
    parser.add_argument(
        "--chart-output",
        default=None,
        help="Path to save the question correctness bar chart image. Defaults to students_root/question_correct_counts.png.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.students_root):
        print(f"Error: root folder not found: {args.students_root}")
        return 1

    stats = generate_statistics(args.students_root, debug=args.debug)
    output_path = args.json_output or os.path.join(args.students_root, "statistics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    chart_output = args.chart_output or os.path.join(args.students_root, "question_correct_counts.png")
    save_question_bar_chart(stats["question_correct_counts"], stats["total_questions"], chart_output)

    print(f"Saved statistics to {output_path}")
    print(f"Saved chart to {chart_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
