# Execution Instructions

## Prerequisites

- Docker installed, or Python 3.10+ and pip available locally.
- Your input PDFs should be placed in the `input` directory.
- The input JSON file (e.g., `input.json`) should also be in the `input` directory.

## Running with Docker

1. **Build the Docker image:**
   ```sh
   docker build -t pdf-section-extractor .
   ```

2. **Run the container:**
   ```sh
   docker run --rm -v "$PWD/input":/app/input -v "$PWD/output":/app/output pdf-section-extractor
   ```

## Running Locally (Without Docker)

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```sh
   python main.py
   ```

## Output

- The output JSON will be written to the `output` directory as `output.json`.

## Notes

- Ensure input PDF filenames and input JSON match the expected references.
- The script is generic and will select relevant sections based on the persona and job-to-be-done in the input JSON.