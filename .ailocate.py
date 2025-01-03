import os
import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

# Constants
DEFAULT_DB_PATH = os.path.expanduser('~/.ailocate_db')
DEFAULT_MODEL = "yolov5s.pt"
DEFAULT_THRESHOLD = 0.3
DEFAULT_DIR = "/"

supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

parser = argparse.ArgumentParser(description="YOLO File Indexer")
parser.add_argument("search", nargs="?", help="Search term for indexed results", default=None)
parser.add_argument("--index", action="store_true", help="Index images in the specified directory")
parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory to search or index")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for detection")
parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (0-1)")
parser.add_argument("--dbfile", default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
args = parser.parse_args()

try:
    import torch
    import yolov5
    from pprint import pprint
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print(f"CTRL+c was pressed")
    sys.exit(0)

def dbg(msg):
    if args.debug:
        print(msg)

# Database setup
def init_database(db_path):
    dbg(f"init_database({db_path})")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        size INTEGER,
                        created_at TEXT,
                        last_modified_at TEXT
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
    dbg(f"add_image_metadata({conn}, {file_path})")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    created_at = datetime.fromtimestamp(stats.st_ctime).isoformat()
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
    cursor.execute('''INSERT OR IGNORE INTO images (file_path, size, created_at, last_modified_at)
                      VALUES (?, ?, ?, ?)''', (file_path, stats.st_size, created_at, last_modified_at))
    conn.commit()
    return cursor.lastrowid

# Check if an image is already indexed with the same model and last modified date
def is_image_indexed(conn, file_path, model):
    dbg(f"is_image_indexed({conn}, {file_path}, {model})")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
    cursor.execute('''SELECT COUNT(*) FROM images
                      JOIN detections ON images.id = detections.image_id
                      WHERE images.file_path = ?
                      AND detections.model = ?
                      AND images.last_modified_at = ?''',
                   (file_path, model, last_modified_at))
    return cursor.fetchone()[0] > 0

# Add detections to the database
def add_detections(conn, image_id, model, detections):
    dbg(f"add_detections({conn}, {image_id}, {detections})")
    cursor = conn.cursor()
    for label, confidence in detections:
        cursor.execute('''INSERT INTO detections (image_id, model, label, confidence)
                          VALUES (?, ?, ?, ?)''', (image_id, model, label, confidence))
    conn.commit()

# Search for images recursively
def find_images(directory):
    dbg(f"find_images({directory})")
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                yield os.path.join(root, file)

# Analyze an image using YOLO
def analyze_image(model, image_path, threshold):
    dbg(f"analyze_image(model, {image_path}, {threshold})")
    try:
        results = model(image_path)
        predictions = results.pred[0]  # Access predictions for the first image
        detections = [(model.names[int(pred[5])], float(pred[4])) for pred in predictions if float(pred[4]) >= threshold]
        return detections
    except OSError:
        return None
    except RuntimeError:
        return None

# Process a single image
def process_image(image_path, model, threshold, conn):
    dbg(f"process_image({image_path}, model, {threshold}, {conn})")
    if is_image_indexed(conn, image_path, args.model):
        dbg(f"Skipping already indexed image: {image_path}")
        return

    image_id = add_image_metadata(conn, image_path)
    detections = analyze_image(model, image_path, threshold)
    if detections:
        add_detections(conn, image_id, args.model, detections)

# Main function
def main():
    dbg(f"Arguments: {args}")

    conn = init_database(args.dbfile)

    if args.index:
        model = yolov5.load(args.model)
        model.conf = args.threshold  # Set confidence threshold
        for image_path in find_images(args.dir):
            process_image(image_path, model, args.threshold, conn)

    if args.search:
        cursor = conn.cursor()
        cursor.execute('''SELECT images.file_path, detections.label, detections.confidence
                          FROM images JOIN detections ON images.id = detections.image_id
                          WHERE detections.label LIKE ?''', (f"%{args.search}%",))
        for row in cursor.fetchall():
            print(row)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("You pressed CTRL+C")
        sys.exit(0)
