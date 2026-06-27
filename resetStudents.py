import argparse
import os
from typing import Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"}


def collect_student_images(student_folder: str) -> List[str]:
    images = []
    for root, _, files in os.walk(student_folder):
        for file_name in sorted(files):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, file_name))
    return images


def delete_student_images(student_folder: str, dry_run: bool = False) -> int:
    image_paths = collect_student_images(student_folder)
    deleted_count = 0
    for image_path in image_paths:
        if dry_run:
            print(f"Would delete: {image_path}")
        else:
            try:
                os.remove(image_path)
                deleted_count += 1
            except OSError as exc:
                print(f"Failed to delete {image_path}: {exc}")
    return deleted_count


def reset_students(students_root: str, dry_run: bool = False) -> int:
    total_deleted = 0
    if not os.path.isdir(students_root):
        raise FileNotFoundError(f"Students root not found: {students_root}")

    for student_name in sorted(os.listdir(students_root)):
        student_folder = os.path.join(students_root, student_name)
        if not os.path.isdir(student_folder):
            continue

        deleted = delete_student_images(student_folder, dry_run=dry_run)
        if deleted or dry_run:
            print(f"{student_name}: {deleted} image file(s) {'found' if dry_run else 'deleted'}.")
        total_deleted += deleted

    return total_deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all student sheet images from student folders.")
    parser.add_argument("students_root", help="Path to the root students folder.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which image files would be deleted without removing them.",
    )
    args = parser.parse_args()

    try:
        deleted_count = reset_students(args.students_root, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.dry_run:
        print("Dry run complete. No files were deleted.")
    else:
        print(f"Deleted {deleted_count} image file(s) from student folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
