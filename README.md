# ScanLetters

A small Python project for scanning student answer sheets, extracting answers from images, scoring them against an answer key, generating statistics, and cleaning up student image folders.

## Project Structure

- `main.py` - OCR and answer extraction from a scanned sheet image.
- `getStudents.py` - Process student folders, compare scanned answers to `answers.txt`, and write results as JSON.
- `getStat.py` - Compute class statistics from student scan results and optionally save a bar chart.
- `resetStudents.py` - Delete all image files inside each student folder.
- `answers.txt` - Answer key file used by `getStudents.py` and `getStat.py`.
- `students/` - Root folder containing one subfolder per student.

## Requirements

- Python 3
- `matplotlib` (optional for chart generation)
- `opencv-python`, `numpy`, and other image dependencies if `main.py` uses OCR and OpenCV processing

Install packages if needed:

```bash
python -m pip install matplotlib opencv-python numpy
```

## Usage

### 1. Scan one image and export answers

```bash
python main.py <image-file> --debug --output answers.txt
```

- `--debug` saves extra debug output when scanning.
- `--output` writes the extracted answers to a file.

### 2. Scan student folders and write results

```bash
python getStudents.py students
```

- `students` should be a folder containing one folder per student.
- Each student folder may contain one or more answer-sheet images.
- The script writes `results.json` inside the students root by default.

### 3. Generate statistics and chart

```bash
python getStat.py students
```

- Reads the same `students` folder structure and answer key.
- Writes `statistics.json` inside the students root.
- Generates a chart file `question_correct_counts.png` by default.

Optional chart output path:

```bash
python getStat.py students --chart-output question_correct_counts.png
```

### 4. Remove all student image files

Dry run first:

```bash
python resetStudents.py students --dry-run
```

Delete images:

```bash
python resetStudents.py students
```

## Notes

- `answers.txt` must contain one answer per line.
- Supported image file extensions in the student folders: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.jfif`.
- `getStat.py` will scale the bar chart y-axis to the total number of questions.

## Example

1. Put student scan images into `students/<student_name>/`.
2. Ensure `answers.txt` contains the correct answers.
3. Run `python getStudents.py students`.
4. Run `python getStat.py students`.
5. If needed, clean up images with `python resetStudents.py students`.
