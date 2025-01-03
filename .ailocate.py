import os
import argparse
import sqlite3
import concurrent.futures
from pathlib import Path
from datetime import datetime
import torch
from yolov5 import YOLO

# Constants
DEFAULT_DB_PATH = os.path.expanduser('~/.ailocate_db')
DEFAULT_MODEL = "yolov5"
DEFAULT_THRESHOLD = 0.3
DEFAULT_DIR = "/"

# Database setup
def init_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        size INTEGER,
                        created_at TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY,
                        image_id INTEGER,
                        model TEXT,
                        label TEXT,
                        confidence REAL,
                        FOREIGN KEY(image_id) REFERENCES images(id)
                    )''')
    conn.commit()
    return conn

# Add image metadata to the database
def add_image_metadata(conn, file_path):
    cursor = conn.cursor()
    stats = os.stat(file_path)
    created_at = datetime.fromtimestamp(stats.st_ctime).isoformat()
    cursor.execute('''INSERT OR IGNORE INTO images (file_path, size, created_at)
                      VALUES (?, ?, ?)''', (file_path, stats.st_size, created_at))
    conn.commit()
    return cursor.lastrowid

# Add detections to the database
def add_detections(conn, image_id, model, detections):
    cursor = conn.cursor()
    for detection in detections:
        label, confidence = detection
        cursor.execute('''INSERT INTO detections (image_id, model, label, confidence)
                          VALUES (?, ?, ?, ?)''', (image_id, model, label, confidence))
    conn.commit()

# Search for images recursively
def find_images(directory):
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                yield os.path.join(root, file)

# Analyze an image using YOLO
def analyze_image(model, image_path, threshold):
    results = model(image_path)
    detections = [(pred['name'], pred['confidence']) for pred in results if pred['confidence'] >= threshold]
    return detections

# Process a single image
def process_image(image_path, model, threshold, conn):
    image_id = add_image_metadata(conn, image_path)
    detections = analyze_image(model, image_path, threshold)
    add_detections(conn, image_id, model.model_name, detections)

# Main function
def main():
    parser = argparse.ArgumentParser(description="YOLO File Indexer")
    parser.add_argument("search", nargs="?", help="Search term for indexed results", default=None)
    parser.add_argument("--index", action="store_true", help="Index images in the specified directory")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory to search or index")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for detection")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (0-1)")
    parser.add_argument("--dbfile", default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser.add_argument("--max_concurrency", type=int, default=1, help="Maximum concurrency for processing")
    args = parser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")

    conn = init_database(args.dbfile)

    if args.index:
        model = YOLO(args.model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = []
            for image_path in find_images(args.dir):
                futures.append(executor.submit(process_image, image_path, model, args.threshold, conn))
            concurrent.futures.wait(futures)
    elif args.search:
        cursor = conn.cursor()
        cursor.execute('''SELECT images.file_path, detections.label, detections.confidence
                          FROM images JOIN detections ON images.id = detections.image_id
                          WHERE detections.label LIKE ?''', (f"%{args.search}%",))
        for row in cursor.fetchall():
            print(row)

if __name__ == "__main__":
    main()
