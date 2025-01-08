import sys

try:
    import numpy
    import subprocess
    import requests
    import tempfile
    import re
    import uuid
    import os
    import argparse
    import sqlite3
    import random
    from pprint import pprint
    import time
    from typing import Optional, Any, Generator

    from pathlib import Path
    from datetime import datetime
    import hashlib
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.console import Console
    import PIL
    from PIL import Image
    from sixel import converter
    import cv2

    from rich_argparse import RichHelpFormatter
except KeyboardInterrupt:
    print("You pressed CTRL+c")
    sys.exit(0)
except ModuleNotFoundError as e:
    print(f"The following module could not be found: {e}")
    sys.exit(1)

def dier(msg: Any) -> None:
    pprint(msg)
    sys.exit(10)

console: Console = Console(
    force_interactive=True,
    soft_wrap=True,
    color_system="256",
    force_terminal=True
)

DEFAULT_MIN_CONFIDENCE_FOR_SAVING: float = 0.1
DEFAULT_DB_PATH: str = os.path.expanduser('~/.smartlocate_db')
DEFAULT_ENCODINGS_FILE: str = os.path.expanduser("~/.smartlocate_face_encodings.pkl")
DEFAULT_MODEL: str = "yolov5s.pt"
DEFAULT_YOLO_THRESHOLD: float = 0.3
DEFAULT_SIXEL_WIDTH: int = 400
DEFAULT_MAX_SIZE: int = 5
DEFAULT_DIR: str = "/"
DEFAULT_BLIP_MODEL_NAME: str = "Salesforce/blip-image-captioning-large"
DEFAULT_TOLERANCE_FACE_DETECTION: float = 0.6

blip_processor: Any = None
blip_model: Any = None
reader: Any = None

supported_formats: set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
allowed_document_extensions: list = ['.doc', '.docx', '.pptx', '.ppt', '.odp', '.odt']

parser = argparse.ArgumentParser(description="Smart file indexer", formatter_class=RichHelpFormatter)
parser.add_argument("search", nargs="?", help="Search term for indexed results", default=None)
parser.add_argument("--index", action="store_true", help="Index images in the specified directory")
parser.add_argument("--size", type=int, default=DEFAULT_SIXEL_WIDTH, help=f"Size to which the image should be resized for displaying it with sixel (default: {DEFAULT_SIXEL_WIDTH}).")
parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory to search or index")
parser.add_argument("--blip_model_name", default=DEFAULT_BLIP_MODEL_NAME, help=f"Name of the blip model. Default: {DEFAULT_BLIP_MODEL_NAME}")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--no_sixel", action="store_true", help="Hide sixel graphics")
parser.add_argument("--yolo", action="store_true", help="Use yolo for indexing")
parser.add_argument("--delete_non_existing_files", action="store_true", help="Delete non-existing files")
parser.add_argument("--shuffle_index", action="store_true", help="Shuffle list of files before indexing")
parser.add_argument("--exact", action="store_true", help="Exact search")
parser.add_argument("--vacuum", action="store_true", help="Vacuums the sqlite3-file (reduces it's size without deleting data)")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for detection")
parser.add_argument("--describe", action="store_true", help="Enable image description")
parser.add_argument("--face_recognition", action="store_true", help="Enable face-recognition (needs user interaction)")
parser.add_argument("--ocr", action="store_true", help="Enable OCR")
parser.add_argument("--documents", action="store_true", help="Enable document indexing")
parser.add_argument("--lang_ocr", nargs='+', default=['de', 'en'], help="OCR languages, default: de, en. Accepts multiple languages.")
parser.add_argument("--yolo_threshold", type=float, default=DEFAULT_YOLO_THRESHOLD, help=f"Confidence YOLO threshold (0-1), default: {DEFAULT_YOLO_THRESHOLD}")
parser.add_argument("--tolerance_face_detection", type=float, default=DEFAULT_TOLERANCE_FACE_DETECTION, help=f"Tolerance for face detection (0-1), default: {DEFAULT_TOLERANCE_FACE_DETECTION}")
parser.add_argument("--yolo_min_confidence_for_saving", type=float, default=DEFAULT_MIN_CONFIDENCE_FOR_SAVING, help=f"Minimal Confidence YOLO required to save YOLO-detections in the database (0-1), default: {DEFAULT_MIN_CONFIDENCE_FOR_SAVING}. Different from --yolo_threshold in that this allows you to search for lower thresholds later on, since they are saved then.")
parser.add_argument("--max_size", type=int, default=DEFAULT_MAX_SIZE, help=f"Max-MB-Size in MB (default: {DEFAULT_MAX_SIZE})")
parser.add_argument("--encoding_face_recognition_file", default=DEFAULT_ENCODINGS_FILE, help=f"Default file for saving encodings (default: {DEFAULT_ENCODINGS_FILE})")
parser.add_argument("--dbfile", default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
parser.add_argument('--exclude', action='append', default=[], help="Folders or paths that should be ignored. Can be used multiple times.")
parser.add_argument("--dont_ask_new_faces", action="store_true", help="Don't ask for new faces (useful for automatically tagging all photos that can be tagged automatically)")
parser.add_argument("--dont_save_new_encoding", action="store_true", help="Don't save new encodings for faces automatically")
args = parser.parse_args()

if not 0 <= args.yolo_min_confidence_for_saving <= 1:
    console.print(f"[red]--yolo_min_confidence_for_saving must be between 0 and 1, is {args.yolo_min_confidence_for_saving}[/]")
    sys.exit(2)

if not 0 <= args.yolo_threshold <= 1:
    console.print(f"[red]--yolo_threshold must be between 0 and 1, is {args.yolo_threshold}[/]")
    sys.exit(2)

if not 0 < args.max_size:
    console.print(f"[red]--max_size must be greater than 0, is set to {args.max_size}[/]")
    sys.exit(2)

if not os.path.exists(args.dir):
    console.print(f"[red]--dir refers to a directory that doesn't exist: {args.dir}[/]")
    sys.exit(2)

yolo_error_already_shown: bool = False

def supports_sixel() -> bool:
    term = os.environ.get("TERM", "").lower()
    if "xterm" in term or "mlterm" in term:
        return True

    try:
        output = subprocess.run(["tput", "setab", "256"], capture_output=True, text=True, check=True)
        if output.returncode == 0 and "sixel" in output.stdout.lower():
            return True
    except FileNotFoundError:
        pass

    return False

console = Console()

if not supports_sixel() and not args.no_sixel:
    console.print("[red]Cannot use sixel. Will set --no_sixel to true.[/]")

    args.no_sixel = True

def dbg(msg: Any) -> None:
    if args.debug:
        console.log(f"[bold yellow]DEBUG:[/] {msg}")

try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    with console.status("[bold green]Loading pickle..."):
        import pickle

    if args.index:
        if args.yolo:
            with console.status("[bold green]Loading yolov5..."):
                import yolov5

        if args.ocr:
            with console.status("[bold green]Loading easyocr..."):
                import easyocr

            with console.status("[bold green]Loading reader..."):
                try:
                    reader = easyocr.Reader(args.lang_ocr)
                except ValueError as e:
                    console.print(f"[red]Loading OCR failed. This is probably an error with the --lang_ocr option. Error:[/] {e}")

        if args.ocr or args.face_recognition:
            with console.status("[bold green]Loading face_recognition..."):
                import face_recognition

        if args.describe or (not args.describe and not args.ocr and not args.yolo and not args.face_recognition and not args.documents):
            with console.status("[bold green]Loading transformers..."):
                import transformers

            with console.status("[bold green]Loading Blip-Transformers...") as load_status:
                from transformers import BlipProcessor, BlipForConditionalGeneration

            with console.status("[bold green]Loading Blip-Models..."):
                try:
                    blip_processor = BlipProcessor.from_pretrained(args.blip_model_name)
                except OSError as e:
                    console.print(f"[red]Loading BlipProcessor failed with this error:[/] {e}")

                try:
                    blip_model = BlipForConditionalGeneration.from_pretrained(args.blip_model_name)
                except OSError as e:
                    console.print(f"[red]Loading BlipModel failed with this error:[/] {e}")
except ModuleNotFoundError as e:
    console.print(f"[red]Module not found:[/] {e}")
    sys.exit(1)
except KeyboardInterrupt:
    console.print("\n[red]You pressed CTRL+C[/]")
    sys.exit(0)

def extract_face_encodings(image_path: str) -> tuple[list, list]:
    import face_recognition

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_encodings, face_locations

def compare_faces(known_encodings: list, unknown_encoding: numpy.ndarray, tolerance: float = args.tolerance_face_detection) -> list:
    import face_recognition

    results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance)

    return results

def save_encodings(encodings: dict, file_name: str) -> None:
    if not args.dont_save_new_encoding:
        with open(file_name, "wb") as file:
            pickle.dump(encodings, file)

def load_encodings(file_name: str) -> dict:
    if os.path.exists(file_name):
        with open(file_name, "rb") as file:
            return pickle.load(file)
    return {}

def detect_faces_and_name_them_when_needed(image_path: str, known_encodings: dict, tolerance: float = args.tolerance_face_detection) -> tuple[list[str], dict, bool]:
    face_encodings, face_locations = extract_face_encodings(image_path)

    manually_entered_name = False

    new_ids = []

    c = 0

    nr_new_faces = 0

    for face_encoding in face_encodings:
        matches = compare_faces(list(known_encodings.values()), face_encoding, tolerance)

        this_face_location = face_locations[c]

        if True in matches:
            matched_id = list(known_encodings.keys())[matches.index(True)]
            new_ids.append(matched_id)
        else:
            if c == 0:
                console.print(f"[yellow]{image_path}:[/]")
                display_sixel(image_path)

            if args.dont_ask_new_faces:
                if nr_new_faces == 0:
                    console.print(f"[yellow]Ignoring face(s) detected {image_path}, since --dont_ask_new_faces was set and new faces were detected[/]")
            else:
                display_sixel_part(image_path, this_face_location)
                try:
                    new_id = input("What is this person's name? [Just press enter if no person is visible or you don't want the person to be saved] ")
                    if any(char.strip() for char in new_id):
                        known_encodings[new_id] = face_encoding
                        new_ids.append(new_id)

                        manually_entered_name = True
                    else:
                        console.print(f"[yellow]Ignoring wrongly detected face in {image_path}[/]")
                except EOFError:
                    console.print("[red]You pressed CTRL+d[/]")
                    sys.exit(0)
            nr_new_faces = nr_new_faces + 1
        c = c + 1

    return new_ids, known_encodings, manually_entered_name

def recognize_persons_in_image(conn: sqlite3.Connection, image_path: str) -> tuple[list[str], bool]:
    known_encodings = load_encodings(args.encoding_face_recognition_file)

    new_ids, known_encodings, manually_entered_name = detect_faces_and_name_them_when_needed(image_path, known_encodings)
    console.print(f"[green]{image_path}: {new_ids}[/]")

    if len(new_ids):
        add_image_persons_mapping(conn, image_path, new_ids)
    else:
        insert_into_no_faces(conn, image_path)

    save_encodings(known_encodings, args.encoding_face_recognition_file)

    return new_ids, manually_entered_name

def to_absolute_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    return os.path.abspath(path)

def ocr_img(img: str) -> Optional[str]:
    global reader

    try:
        if reader is None:
            with console.status("[bold green]Loading easyocr...") as load_status:
                if "easyocr" not in sys.modules:
                    import easyocr

            with console.status("[bold green]Loading reader...") as load_status:
                reader = easyocr.Reader(args.lang_ocr)

        if reader is None:
            console.print("[red]reader was not defined. Will not OCR.[/]")
            return None

        if os.path.exists(img):
            result = reader.readtext(img)

            return result

        console.print(f"[red]ocr_img: file {img} not found[/]")
        return None
    except (cv2.error, ValueError) as e:
        console.print(f"[red]ocr_img: file {img} caused an error: {e}[/]")
        return None

def resize_image(input_path: str, output_path: str, max_size: int) -> bool:
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        img.save(output_path)

        return True

    return False

def display_sixel_part(image_path: str, location: list) -> None:
    top, right, bottom, left = location

    with tempfile.NamedTemporaryFile(mode="wb") as jpg:
        import face_recognition

        image = face_recognition.load_image_file(image_path)
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        pil_image.save(jpg.name, format="JPEG")

        display_sixel(jpg.name)

def display_sixel(image_path: str) -> None:
    if not supports_sixel():
        console.print(f"[red]Error: This terminal does not support sixel. Cannot display {image_path}[/]")
        return

    unique_filename = f"/tmp/{uuid.uuid4().hex}_resized_image.png"

    try:
        resize_image(image_path, unique_filename, args.size)

        c = converter.SixelConverter(unique_filename)
        c.write(sys.stdout)
    except FileNotFoundError:
        console.print(f"[red]Could not find {image_path}[/]")
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)

def load_existing_images(conn: sqlite3.Connection) -> dict[Any, Any]:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT file_path, md5 FROM images
        UNION ALL
        SELECT file_path, md5 FROM ocr_results;
    ''')
    rows = cursor.fetchall()
    cursor.close()
    return {row[0]: row[1] for row in rows}

def is_file_in_img_desc_db(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM image_description WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_file_in_ocr_db(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM ocr_results WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_file_in_yolo_db(conn: sqlite3.Connection, file_path: str, existing_files: Optional[dict]) -> bool:
    if existing_files and file_path in existing_files:
        return True

    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM images WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_existing_detections_label(conn: sqlite3.Connection, label: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT label FROM detections WHERE label = ? LIMIT 1''', (label,))
    res = cursor.fetchone()  # Gibt entweder eine Zeile oder None zurück
    cursor.close()

    return res is not None  # Wenn eine Zeile zurückgegeben wurde, existiert das Label

def get_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_empty_image(conn: sqlite3.Connection, file_path: str) -> None:
    dbg(f"add_empty_image(conn, {file_path})")
    md5_hash = get_md5(file_path)

    cursor = conn.cursor()

    while True:
        try:
            cursor.execute('''SELECT md5 FROM empty_images WHERE file_path = ?''', (file_path,))
            existing_hash = cursor.fetchone()

            if existing_hash:
                if existing_hash[0] != md5_hash:
                    cursor.execute('''UPDATE empty_images SET md5 = ? WHERE file_path = ?''', (md5_hash, file_path))
                    conn.commit()
                    dbg(f"Updated MD5 hash for {file_path}")
            else:
                cursor.execute('''INSERT INTO empty_images (file_path, md5) VALUES (?, ?)''', (file_path, md5_hash))
                conn.commit()
                dbg(f"Added empty image: {file_path}")
            cursor.close()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)

def add_image_persons_mapping(conn: sqlite3.Connection, file_path: str, person_names: list[str]) -> None:
    for elem in person_names:
        add_image_and_person_mapping(conn, file_path, elem)

def add_image_and_person_mapping(conn: sqlite3.Connection, file_path: str, person_name: str) -> None:
    cursor = conn.cursor()

    while True:
        try:
            # 1. Image ID aus der images-Tabelle holen oder einfügen
            cursor.execute('''SELECT id FROM images WHERE file_path = ?''', (file_path,))
            image_id = cursor.fetchone()

            if not image_id:
                cursor.execute('''INSERT INTO images (file_path) VALUES (?)''', (file_path,))
                conn.commit()
                image_id = cursor.lastrowid
            else:
                image_id = image_id[0]

            cursor.execute('''SELECT id FROM person WHERE name = ?''', (person_name,))
            person_id = cursor.fetchone()

            if not person_id:
                cursor.execute('''INSERT INTO person (name) VALUES (?)''', (person_name,))
                conn.commit()
                person_id = cursor.lastrowid
            else:
                person_id = person_id[0]

            # 3. Zuordnung in die image_person_mapping-Tabelle einfügen
            cursor.execute('''
                INSERT OR IGNORE INTO image_person_mapping (image_id, person_id)
                VALUES (?, ?)
            ''', (image_id, person_id))
            conn.commit()

            dbg(f"Mapped image '{file_path}' (ID: {image_id}) to person '{person_name}' (ID: {person_id})")
            cursor.close()
            return  # Erfolgreiche Zuordnung, Schleife beenden

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):  # Wenn die Datenbank gesperrt ist, erneut versuchen
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:  # Andere Fehler, die das Hinzufügen der Zuordnung verhindern
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(13)

def insert_into_no_faces(conn: sqlite3.Connection, file_path: str) -> None:
    execute_with_retry(conn, 'INSERT OR IGNORE INTO no_faces (file_path) VALUES (?)', (file_path, ))

def faces_already_recognized(conn: sqlite3.Connection, image_path: str) -> bool:
    cursor = conn.cursor()

    # Überprüfen, ob das Bild in der no_faces-Tabelle existiert
    cursor.execute('''SELECT 1 FROM no_faces WHERE file_path = ?''', (image_path,))
    if cursor.fetchone():
        cursor.close()
        return True  # Bild befindet sich in der no_faces-Tabelle

    # Überprüfen, ob das Bild in der image_person_mapping-Tabelle existiert
    cursor.execute('''SELECT 1 FROM image_person_mapping
                      JOIN images ON images.id = image_person_mapping.image_id
                      WHERE images.file_path = ?''', (image_path,))
    if cursor.fetchone():
        cursor.close()
        return True  # Bild befindet sich in der image_person_mapping-Tabelle

    cursor.close()
    return False  # Bild wurde noch nicht durchsucht

def get_image_id_by_file_path(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    try:
        # SQL query to retrieve the image ID
        query = '''SELECT id FROM images WHERE file_path = ?'''

        # Execute the query
        cursor = conn.cursor()
        cursor.execute(query, (file_path,))
        result = cursor.fetchone()

        # Check if a result was found
        if result:
            return int(result[0])
        return None
    except Exception as e:
        print(f"Error while fetching image ID for file_path '{file_path}': {e}")
        return None

def init_database(db_path: str) -> sqlite3.Connection:
    with console.status("[bold green]Initializing database...") as status:
        dbg(f"init_database({db_path})")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        status.update("[bold green]Creating table images...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            size INTEGER,
                            created_at TEXT,
                            last_modified_at TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table images.")

        status.update("[bold green]Creating index images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)')
        status.update("[bold green]Created index images(file_path).")

        status.update("[bold green]Creating index images(md5)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5)')
        status.update("[bold green]Created index images(md5).")

        status.update("[bold green]Creating table detections...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                            id INTEGER PRIMARY KEY,
                            image_id INTEGER,
                            model TEXT,
                            label TEXT,
                            confidence REAL,
                            FOREIGN KEY(image_id) REFERENCES images(id)
                        )''')
        status.update("[bold green]Created table detections.")

        status.update("[bold green]Creating index detections(image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id)')
        status.update("[bold green]Created index detections(image_id).")

        status.update("[bold green]Creating index detections(confidence)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(confidence)')
        status.update("[bold green]Created index detections(confidence).")

        status.update("[bold green]Creating index detections(image_id, model)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_model ON detections(image_id, model)')
        status.update("[bold green]Created index detections(image_id, model).")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);')
        status.update("[bold green]Created index detections(label).")

        status.update("[bold green]Creating table empty_images...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS empty_images (
                            file_path TEXT UNIQUE,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table empty_images.")

        status.update("[bold green]Creating index empty_images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_empty_images_file_path ON empty_images(file_path)')
        status.update("[bold green]Created index empty_images(file_path).")

        status.update("[bold green]Creating table ocr_results...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS ocr_results (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            extracted_text TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table ocr_results.")

        status.update("[bold green]Creating index ocr_results(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_results_file_path ON ocr_results(file_path)')
        status.update("[bold green]Created index ocr_results(file_path).")

        status.update("[bold green]Creating index ocr_results(md5)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_results_md5 ON ocr_results(md5)')
        status.update("[bold green]Created index ocr_results(md5).")

        status.update("[bold green]Creating table image_description...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS image_description (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            image_description TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table image_description.")

        status.update("[bold green]Creating index image_description(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_description_file_path ON image_description(file_path)')
        status.update("[bold green]Created index image_description(file_path).")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)')
        status.update("[bold green]Created index detections(label).")

        status.update("[bold green]Creating index detections(image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id)')
        status.update("[bold green]Created index detections(image_id).")

        status.update("[bold green]Creating index images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)')
        status.update("[bold green]Created index images(file_path).")

        status.update("[bold green]Creating index detections(label, image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label_image_id ON detections(label, image_id)')
        status.update("[bold green]Created index detections(label, image_id).")

        status.update("[bold green]Creating index image_description_no_case...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_description_no_case ON image_description(image_description COLLATE NOCASE)')
        status.update("[bold green]Created index image_description_no_case.")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)')
        status.update("[bold green]Created index detections(label).")

        status.update("[bold green]Creating tables for person mapping...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_person_mapping (
                image_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                PRIMARY KEY (image_id, person_id),
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS no_faces (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL
            );
        ''')

        status.update("[bold green]Created tables for person mapping.")

        status.update("[bold green]Creating tables for documents...")
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
                file_path,
                content,
                tokenize = 'porter'
            );
        ''')
        status.update("[bold green]Created tables for documents.")

        cursor.close()
        conn.commit()
        return conn

def document_already_exists(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.cursor()

    # Überprüfen, ob das Bild in der no_faces-Tabelle existiert
    cursor.execute('''SELECT 1 FROM documents WHERE file_path = ?''', (file_path,))
    if cursor.fetchone():
        cursor.close()
        return True

    cursor.close()
    return False

def insert_document_if_not_exists(conn: sqlite3.Connection, file_path: str, _pandoc: bool = True) -> bool:
    if document_already_exists(conn, file_path):
        return False

    text = ""

    if _pandoc:
        text = convert_file_to_text(file_path)
    else:
        text = open(file_path, "r").read()

    if text:
        insert_document(conn, file_path, text)

    return True

def insert_document(conn: sqlite3.Connection, file_path: str, document: str):
    execute_with_retry(conn, '''INSERT INTO documents (file_path, content) VALUES (?, ?);''', (file_path, document, ))

def convert_file_to_text(file_path: str, _format: str = "plain") -> Optional[str]:
    try:
        import pypandoc

        pypandoc.download_pandoc

        try:
            output = pypandoc.convert_file(file_path, _format)
            return output
        except Exception as e:
            return f"Error: {e}"
    except ModuleNotFoundError:
        console.print(f"[red]Module not found: pandoc[/]")

    return None

def traverse_document_files(conn: sqlite3.Connection, directory_path: str) -> bool:
    if not os.path.isdir(directory_path):
        console.print(f"[red]The provided path '{directory_path}' is not a valid directory.[/]")
        return False

    found_and_converted_some = False

    with console.status(f"[bold green]Finding documents in {args.dir}...") as status:
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                # Check if file has an allowed extension
                if any(file_name.lower().endswith(ext) for ext in allowed_document_extensions):
                    try:
                        status.update(f"[bold green]Found document {file_path}[/]")
                        found_something = insert_document_if_not_exists(conn, file_path)

                        if found_something:
                            console.print(f"[bold green]Indexed {file_path}[/]")
                            found_and_converted_some = True
                        else:
                            console.print(f"[bold green]Skipping {file_path} because nothing was found in it, it was not a valid file or it was already indexed.[/]")
                    except Exception as e:
                        console.print(f"Error processing file '{file_path}': {e}")
                elif file_name.lower().endswith(".md") or file_name.lower().endswith(".txt"):
                    try:
                        status.update(f"[bold green]Found document {file_path}[/]")
                        found_something = insert_document_if_not_exists(conn, file_path, False)

                        if found_something:
                            console.print(f"[bold green]Indexed {file_path}[/]")
                            found_and_converted_some = True
                        else:
                            console.print(f"[bold green]Skipping {file_path} because nothing was found in it, it was not a valid file or it was already indexed.[/]")
                    except Exception as e:
                        console.print(f"Error processing file '{file_path}': {e}")

    return found_and_converted_some

def execute_with_retry(conn: sqlite3.Connection, query: str, params: tuple) -> None:
    cursor = conn.cursor()

    while True:
        try:
            cursor.execute(query, params)
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                raise e

    while True:
        try:
            cursor.close()
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                raise e

def add_image_metadata(conn: sqlite3.Connection, file_path: str) -> tuple[int, str]:
    dbg(f"add_image_metadata(conn, {file_path})")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    md5_hash = get_md5(file_path)
    created_at = datetime.fromtimestamp(stats.st_ctime).isoformat()
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()

    execute_with_retry(conn, '''INSERT OR IGNORE INTO images (file_path, size, created_at, last_modified_at, md5) VALUES (?, ?, ?, ?, ?)''', (file_path, stats.st_size, created_at, last_modified_at, md5_hash))

    cursor.execute('SELECT id FROM images WHERE file_path = ?', (file_path,))
    image_id = cursor.fetchone()[0]

    return image_id, md5_hash

def is_image_indexed(conn: sqlite3.Connection, file_path: str) -> bool:
    dbg(f"is_image_indexed(conn, {file_path})")

    while True:
        try:
            cursor = conn.cursor()
            stats = os.stat(file_path)
            last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()

            cursor.execute('''SELECT COUNT(*) FROM images
                               JOIN detections ON images.id = detections.image_id
                               WHERE images.file_path = ?
                               AND detections.model = ?
                               AND images.last_modified_at = ?''',
                           (file_path, args.model, last_modified_at))

            res = cursor.fetchone()[0]
            cursor.close()

            return res > 0
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)
        except FileNotFoundError:
            return True

    return False

def add_detections(conn: sqlite3.Connection, image_id: int, model_name: str, detections: list) -> None:
    dbg(f"add_detections(conn, {image_id}, detections)")
    for label, confidence in detections:
        execute_with_retry(conn, '''INSERT INTO detections (image_id, model, label, confidence) VALUES (?, ?, ?, ?)''', (image_id, model_name, label, confidence))

def is_ignored_path(path: str) -> bool:
    if args.exclude:
        for excl in args.exclude:
            if path.startswith(to_absolute_path(excl)):
                return True

    return False

def find_images(existing_files: dict) -> Generator:
    for root, _, files in os.walk(args.dir):
        for file in files:
            if Path(file).suffix.lower() in supported_formats and file not in existing_files:
                _path = os.path.join(root, file)
                if not is_ignored_path(_path):
                    yield _path

def analyze_image(model: Any, image_path: str) -> Optional[list]:
    dbg(f"analyze_image(model, {image_path})")
    try:
        console.print(f"[bright_yellow]Predicting {image_path} with YOLO[/]")

        results = model(image_path)
        predictions = results.pred[0]
        detections = [(model.names[int(pred[5])], float(pred[4])) for pred in predictions if float(pred[4]) >= args.yolo_min_confidence_for_saving]
        return detections
    except (OSError, RuntimeError):
        return None
    except ValueError as e:
        console.print(f"[red]Value-Error: {e}[/]")
        return None
    except PIL.Image.DecompressionBombError as e:
        console.print(f"[red]Error: {e}, probably the image is too large[/]")
        return None
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return None

def process_image(image_path: str, model: Any, conn: sqlite3.Connection) -> None:
    dbg(f"process_image({image_path}, model, conn)")

    image_id, md5_hash = add_image_metadata(conn, image_path)

    detections = analyze_image(model, image_path)
    if detections:
        add_detections(conn, image_id, args.model, detections)
    else:
        add_empty_image(conn, image_path)

def show_general_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM documents')
        documents = cursor.fetchone()[0]

        console.print("[bold underline cyan]General Statistics[/bold underline cyan]\n")

        table = Table(title="General Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Images", str(total_images))
        table.add_row("Total Documents", str(documents))
        table.add_row("Total Detections", str(total_detections))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_yolo_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('''SELECT detections.label, COUNT(*)
                          FROM detections
                          JOIN images ON images.id = detections.image_id
                          GROUP BY detections.label''')
        yolo_stats = cursor.fetchall()

        cursor.close()

        console.print("[bold underline cyan]YOLO Detection Statistics[/bold underline cyan]\n")

        if yolo_stats:
            yolo_table = Table(title="YOLO Category Statistics")
            yolo_table.add_column("Label", justify="left", style="cyan")
            yolo_table.add_column("Count", justify="right", style="green")

            for row in yolo_stats:
                yolo_table.add_row(row[0], str(row[1]))

            console.print(yolo_table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_empty_images_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM empty_images')
        total_empty_images = cursor.fetchone()[0]

        console.print("[bold underline cyan]Empty Images Statistics[/bold underline cyan]\n")

        table = Table(title="Empty Images Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Empty Images", str(total_empty_images))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_documents_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_ocr_results = cursor.fetchone()[0]

        console.print("[bold underline cyan]Documents Statistics[/bold underline cyan]\n")

        table = Table(title="Documents Results Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Documents Results", str(total_ocr_results))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


def show_ocr_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ocr_results')
        total_ocr_results = cursor.fetchone()[0]

        console.print("[bold underline cyan]OCR Results Statistics[/bold underline cyan]\n")

        table = Table(title="OCR Results Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total OCR Results", str(total_ocr_results))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_image_description_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM image_description')
        total_image_descriptions = cursor.fetchone()[0]

        console.print("[bold underline cyan]Image Description Statistics[/bold underline cyan]\n")

        table = Table(title="Image Description Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Image Descriptions", str(total_image_descriptions))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_person_mapping_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM person')
        total_persons = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM image_person_mapping')
        total_mappings = cursor.fetchone()[0]

        console.print("[bold underline cyan]Person Mapping Statistics[/bold underline cyan]\n")

        table = Table(title="Person Mapping Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Persons", str(total_persons))
        table.add_row("Total Image-Person Mappings", str(total_mappings))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_face_recognition_stats(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT person.name, COUNT(image_person_mapping.image_id) AS recognition_count
            FROM person
            JOIN image_person_mapping ON person.id = image_person_mapping.person_id
            GROUP BY person.name
            ORDER BY recognition_count DESC
        ''')
        face_recognition_stats = cursor.fetchall()

        console.print("[bold underline cyan]Face Recognition Statistics[/bold underline cyan]\n")

        if face_recognition_stats:
            table = Table(title="Face Recognition Statistics (Sorted by Count)")
            table.add_column("Name", style="cyan")
            table.add_column("Recognition Count", justify="right", style="green")

            for name, count in face_recognition_stats:
                table.add_row(name, str(count))

            console.print(table)
        else:
            console.print("[yellow]No face recognition data found.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_statistics(conn: sqlite3.Connection) -> None:
    show_all = not args.describe and not args.ocr and not args.yolo and not args.face_recognition and not args.documents

    if show_all:
        show_general_stats(conn)
        show_empty_images_stats(conn)

    if args.yolo or show_all:
        show_yolo_stats(conn)

    if args.describe or show_all:
        show_image_description_stats(conn)

    if args.ocr or show_all:
        show_ocr_stats(conn)

    if args.documents or show_all:
        show_documents_stats(conn)

    if args.face_recognition or show_all:
        show_person_mapping_stats(conn)

        show_face_recognition_stats(conn)

def delete_yolo_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting detections for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM detections WHERE image_id IN (SELECT id FROM images WHERE file_path = ?)''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from detections for {file_path}.")

def delete_empty_images_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from empty_images for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM empty_images WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from empty_images for {file_path}.")

def delete_image_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from images for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM images WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from images for {file_path}.")

def delete_ocr_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from ocr_results for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM ocr_results WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from ocr_results for {file_path}.")

def delete_faces_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    image_id = get_image_id_by_file_path(conn, file_path)

    if image_id is None:
        return

    if delete_status:
        delete_status.update(f"[bold green]Deleting from image_person_mapping for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM image_person_mapping WHERE image_id = ?''', (image_id,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from image_person_mapping for {file_path}.")

def delete_no_faces_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from no_faces for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM no_faces WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from no_faces for {file_path}.")

def delete_image_description_from_image_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from image_description for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM image_description WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from image_description for {file_path}.")

def delete_document_from_document_path(conn: sqlite3.Connection, delete_status: Any, file_path: str) -> None:
    if delete_status:
        delete_status.update(f"[bold green]Deleting from document for {file_path}...")
    execute_with_retry(conn, '''DELETE FROM documents WHERE file_path = ?''', (file_path,))
    if delete_status:
        delete_status.update(f"[bold green]Deleted from document for {file_path}.")

def delete_entries_by_filename(conn: sqlite3.Connection, file_path: str) -> None:
    dbg(f"delete_entries_by_filename(conn, {file_path})")

    while True:
        try:
            cursor = conn.cursor()

            with console.status("[bold green]Deleting files from DB that do not exist...") as delete_status:
                delete_yolo_from_image_path(conn, delete_status, file_path)

                delete_image_from_image_path(conn, delete_status, file_path)

                delete_empty_images_from_image_path(conn, delete_status, file_path)

                delete_ocr_from_image_path(conn, delete_status, file_path)

                delete_no_faces_from_image_path(conn, delete_status, file_path)

                delete_image_description_from_image_path(conn, delete_status, file_path)

                delete_document_from_document_path(conn, delete_status, file_path)

                cursor.close()
                conn.commit()

                console.print(f"[red]Deleted all entries for {file_path}[/]")
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                cursor.close()
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)

def check_entries_in_table(conn: sqlite3.Connection, table_name: str, file_path: str | int, where_name: str = "file_path") -> int:
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_name} = ?"

    try:
        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        cursor = conn.cursor()
        cursor.execute(query, (file_path,))
        count = cursor.fetchone()[0]

        return count
    except Exception as e:
        print(f"Error while checking entries in table '{table_name}': {e}. Full query:\n{query}")
        return 0

def get_existing_documents(conn: sqlite3.Connection) -> dict[Any, Any]:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT file_path FROM documents
    ''')
    rows = cursor.fetchall()
    cursor.close()
    return {row[0] for row in rows}

def delete_non_existing_documents(conn: sqlite3.Connection) -> None:
    for file in get_existing_documents(conn):
        if not os.path.exists(file):
            delete_entries_by_filename(conn, file)

def delete_non_existing_image_files(conn: sqlite3.Connection, existing_files: Optional[dict]) -> Optional[dict]:
    if existing_files:
        for file in existing_files:
            if not os.path.exists(file):
                delete_entries_by_filename(conn, file)
        existing_files = load_existing_images(conn)

    return existing_files

def add_description(conn: sqlite3.Connection, file_path: str, desc: str) -> None:
    dbg(f"add_description(conn, {file_path}, <desc>)")
    md5_hash = get_md5(file_path)
    execute_with_retry(conn, '''INSERT INTO image_description (file_path, image_description, md5) VALUES (?, ?, ?)''', (file_path, desc, md5_hash))

def add_ocr_result(conn: sqlite3.Connection, file_path: str, extracted_text: str) -> None:
    dbg(f"add_ocr_result(conn, {file_path}, <extracted_text>)")
    md5_hash = get_md5(file_path)
    execute_with_retry(conn, '''INSERT INTO ocr_results (file_path, extracted_text, md5) VALUES (?, ?, ?)''', (file_path, extracted_text, md5_hash))

def search_yolo(conn: sqlite3.Connection) -> int:
    yolo_results = None

    if not is_existing_detections_label(conn, args.search):
        return 0

    with console.status("[bold green]Searching through YOLO-results...") as status:
        cursor = conn.cursor()
        cursor.execute('''SELECT images.file_path, detections.label, detections.confidence
                          FROM images JOIN detections ON images.id = detections.image_id
                          WHERE detections.label LIKE ? GROUP BY images.file_path''', (f"%{args.search}%",))
        yolo_results = cursor.fetchall()
        cursor.close()

    nr_yolo = 0

    if not args.no_sixel:
        for row in yolo_results:
            conf = row[2]
            if conf >= args.yolo_threshold:
                if not is_ignored_path(row[0]):
                    print(f"{row[0]} (certainty: {conf:.2f})")
                    display_sixel(row[0])
                    print("\n")

                    nr_yolo = nr_yolo + 1
    else:
        table = Table(title="Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Label", justify="center", style="magenta")
        table.add_column("Confidence", justify="right", style="green")
        for row in yolo_results:
            conf = row[2]
            if conf >= args.yolo_threshold:
                if not is_ignored_path(row[0]):
                    table.add_row(*map(str, row))

                    nr_yolo = nr_yolo + 1

        if len(yolo_results):
            console.print(table)

    return nr_yolo

def build_sql_query_description(words: list[str]) -> tuple[str, tuple[str, ...]]:
    conditions = ["image_description LIKE ? COLLATE NOCASE" for _ in words]
    sql_query = f"SELECT file_path, image_description FROM image_description WHERE {' AND '.join(conditions)}"
    values = tuple(f"%{word}%" for word in words)
    return sql_query, values

def clean_search_query(query: str) -> list[str]:
    if args.exact:
        return [query]

    cleaned_query = re.sub(r"[^a-zA-Z\s]", "", query)
    sp = cleaned_query.split()
    return sp

def search_description(conn: sqlite3.Connection) -> int:
    ocr_results = None

    nr_desc = 0

    with console.status("[bold green]Searching through descriptions...") as status:
        cursor = conn.cursor()
        words = clean_search_query(args.search)  # Clean and split the search string
        sql_query, values = build_sql_query_description(words)  # Build the SQL query dynamically
        cursor.execute(sql_query, values)
        ocr_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in ocr_results:
            if not is_ignored_path(row[0]):
                console.print(f"[italic]File: {row[0]}[/]\n")
                print("Description:\n{row[1]}\n")
                display_sixel(row[0])
                print("\n")

            nr_desc = nr_desc + 1
    else:
        table = Table(title="OCR Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Extracted Text", justify="center", style="magenta")
        for row in ocr_results:
            file_path, extracted_text = row
            if not is_ignored_path(file_path):
                table.add_row(file_path, extracted_text)

                nr_desc = nr_desc + 1
        if len(ocr_results):
            console.print(table)

    return nr_desc

def build_sql_query_documents(words: list[str]) -> tuple[str, tuple[str, ...]]:
    conditions = ["content LIKE ? COLLATE NOCASE" for _ in words]
    sql_query = f"SELECT file_path, content FROM documents WHERE {' AND '.join(conditions)}"
    values = tuple(f"%{word}%" for word in words)
    return sql_query, values

def build_sql_query_ocr(words: list[str]) -> tuple[str, tuple[str, ...]]:
    conditions = ["extracted_text LIKE ? COLLATE NOCASE" for _ in words]
    sql_query = f"SELECT file_path, extracted_text FROM ocr_results WHERE {' AND '.join(conditions)}"
    values = tuple(f"%{word}%" for word in words)
    return sql_query, values

def search_documents(conn: sqlite3.Connection) -> int:
    ocr_results = None
    nr_documents = 0

    with console.status("[bold green]Searching through documents...") as status:
        cursor = conn.cursor()

        # Clean and split the search string
        words = clean_search_query(args.search)

        # Build the SQL query dynamically
        sql_query, values = build_sql_query_documents(words)
        cursor.execute(sql_query, values)
        ocr_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in ocr_results:
            if not is_ignored_path(row[0]):
                console.print(f"[italic]File: {row[0]}[/]\n")
                console.print(format_text_with_keywords(f"Text:\n{row[1]}\n", words))
                print("\n")
                nr_documents += 1
    else:
        table = Table(title="Documents Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Text", justify="center", style="magenta")
        for row in ocr_results:
            file_path, extracted_text = row
            if not is_ignored_path(file_path):
                table.add_row(file_path, extracted_text)
                nr_documents += 1

        if len(ocr_results):
            console.print(table)

    return nr_documents


def search_ocr(conn: sqlite3.Connection) -> int:
    ocr_results = None
    nr_ocr = 0

    with console.status("[bold green]Searching through OCR results...") as status:
        cursor = conn.cursor()

        # Clean and split the search string
        words = clean_search_query(args.search)

        # Build the SQL query dynamically
        sql_query, values = build_sql_query_ocr(words)
        cursor.execute(sql_query, values)
        ocr_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in ocr_results:
            if not is_ignored_path(row[0]):
                console.print(f"[italic]File: {row[0]}[/]\n")
                console.print(format_text_with_keywords(f"Extracted Text:\n{row[1]}\n", words))
                display_sixel(row[0])
                print("\n")
                nr_ocr += 1
    else:
        table = Table(title="OCR Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Extracted Text", justify="center", style="magenta")
        for row in ocr_results:
            file_path, extracted_text = row
            if not is_ignored_path(file_path):
                table.add_row(file_path, extracted_text)
                nr_ocr += 1

        if len(ocr_results):
            console.print(table)

    return nr_ocr

def search_faces(conn: sqlite3.Connection) -> int:
    person_results = None

    # Überprüfen, ob der angegebene Name in der Person-Tabelle existiert
    cursor = conn.cursor()
    cursor.execute('''SELECT id FROM person WHERE name LIKE ?''', (f"%{args.search}%",))
    person_results = cursor.fetchall()
    cursor.close()

    if not person_results:
        return 0  # Keine Person gefunden

    # Suchen nach Bildern, die mit der gefundenen Person verknüpft sind
    with console.status("[bold green]Searching for images of the person...") as status:
        cursor = conn.cursor()
        person_ids = [str(row[0]) for row in person_results]
        placeholders = ",".join("?" * len(person_ids))  # Platzhalter für die IDs der Personen
        query = f'''
            SELECT images.file_path
            FROM images
            JOIN image_person_mapping ON images.id = image_person_mapping.image_id
            WHERE image_person_mapping.person_id IN ({placeholders})
        '''
        cursor.execute(query, person_ids)
        person_images = cursor.fetchall()
        cursor.close()

    nr_images = 0

    if not args.no_sixel:
        for row in person_images:
            print(row[0])
            display_sixel(row[0])  # Falls Sixel angezeigt werden soll
            print("\n")
            nr_images += 1
    else:
        table = Table(title="Person Image Results")
        table.add_column("File Path", justify="left", style="cyan")
        for row in person_images:
            table.add_row(row[0])

        if len(person_images):
            console.print(table)

        nr_images = len(person_images)

    return nr_images

def search(conn: sqlite3.Connection) -> None:
    try:
        table = Table(title="Search overview")

        yolo, ocr, desc, faces, documents = False, False, False, False, False

        if not args.yolo and not args.ocr and not args.describe and not args.face_recognition and not args.documents:
            yolo, ocr, desc, faces, documents = True, True, True, True, True
        else:
            if args.yolo:
                yolo = True

            if args.ocr:
                ocr = True

            if args.describe:
                desc = True

            if args.face_recognition:
                faces = True

            if args.documents:
                documents = True

        row = []

        if yolo:
            table.add_column("Nr. Yolo Results", justify="left", style="cyan")
            nr_yolo = search_yolo(conn)
            row.append(str(nr_yolo))

        if ocr:
            table.add_column("Nr. OCR Results", justify="left", style="cyan")
            nr_ocr = search_ocr(conn)
            row.append(str(nr_ocr))

        if desc:
            table.add_column("Nr. Description Results", justify="left", style="cyan")
            nr_desc = search_description(conn)
            row.append(str(nr_desc))

        if faces:
            table.add_column("Nr. Recognized faces", justify="left", style="cyan")
            nr_faces = search_faces(conn)
            row.append(str(nr_faces))

        if documents:
            table.add_column("Nr. Documents", justify="left", style="cyan")
            nr_documents = search_documents(conn)
            row.append(str(nr_documents))

        table.add_row(*row)

        console.print(table)
    except sqlite3.OperationalError as e:
        console.print(f"[red]Error while running sqlite-query: {e}[/]")

def yolo_file(conn: sqlite3.Connection, image_path: str, existing_files: Optional[dict], model: Any) -> None:
    if model is None:
        return

    if is_file_in_yolo_db(conn, image_path, existing_files):
        console.print(f"[green]Image {image_path} already in yolo-database. Skipping it.[/]")
    else:
        if is_image_indexed(conn, image_path):
            console.print(f"[green]Image {image_path} already indexed. Skipping it.[/]")
        else:
            process_image(image_path, model, conn)
            if existing_files is not None:
                existing_files[image_path] = get_md5(image_path)

def get_image_description(image_path: str) -> str:
    global blip_model, blip_processor

    try:
        image = Image.open(image_path).convert("RGB")
        if blip_processor is None:
            with console.status("[bold green]Loading transformers..."):
                import transformers

            with console.status("[bold green]Loading Blip-Transformers..."):
                from transformers import BlipProcessor, BlipForConditionalGeneration

            with console.status("[bold green]Loading Blip-Models..."):
                blip_processor = BlipProcessor.from_pretrained(args.blip_model_name)
                blip_model = BlipForConditionalGeneration.from_pretrained(args.blip_model_name)

        if blip_processor is None:
            console.print("blip_processor was none. Cannot describe image.")
            return ""

        if blip_model is None:
            console.print("blip_model was none. Cannot describe image.")
            return ""

        inputs = blip_processor(images=image, return_tensors="pt")

        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

        return caption
    except PIL.Image.DecompressionBombError as e:
        console.print(f"File {image_path} failed with error {e}")
        return ""

def describe_img(conn: sqlite3.Connection, image_path: str) -> None:
    if is_file_in_img_desc_db(conn, image_path):
        console.print(f"[green]Image {image_path} already in image-description-database. Skipping it.[/]")
    else:
        try:
            image_description = get_image_description(image_path)
            if image_description:
                console.print(f"[green]Saved description '{image_description}' for {image_path}[/]")
                add_description(conn, image_path, image_description)
            else:
                console.print(f"[yellow]Image {image_path} could not be described. Saving it as empty.[/]")
                add_description(conn, image_path, "")

        except FileNotFoundError:
            console.print(f"[red]File {image_path} not found[/]")

def ocr_file(conn: sqlite3.Connection, image_path: str) -> None:
    if is_file_in_ocr_db(conn, image_path):
        console.print(f"[green]Image {image_path} already in ocr-database. Skipping it.[/]")
    else:
        try:
            file_size = os.path.getsize(image_path)

            if file_size < args.max_size * 1024 * 1024:
                extracted_text = ocr_img(image_path)
                if extracted_text:
                    texts = [item[1] for item in extracted_text]
                    text = " ".join(texts)
                    if text:
                        add_ocr_result(conn, image_path, text)
                        console.print(f"[green]Saved OCR for {image_path}[/]")
                    else:
                        console.print(f"[yellow]Image {image_path} contains no text. Saving it as empty.[/]")
                        add_ocr_result(conn, image_path, "")
                else:
                    console.print(f"[yellow]Image {image_path} contains no text. Saving it as empty.[/]")
                    add_ocr_result(conn, image_path, "")

            else:
                console.print(f"[red]Image {image_path} is too large. Will skip OCR. Max-Size: {args.max_size}MB, is {file_size / 1024 / 1024}MB[/]")
        except FileNotFoundError:
            console.print(f"[red]File {image_path} not found[/]")

def is_valid_file_path(path: str) -> bool:
    try:
        normalized_path = os.path.abspath(path)
        return os.path.isfile(normalized_path)
    except Exception as e:
        print(f"Error checking the path {path}: {e}")

    return False

def is_valid_image_file(path: str) -> bool:
    try:
        if not os.path.isfile(path):
            return False

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        return False

def display_menu(options: list, prompt: str = "Choose an option (enter the number): ") -> Optional[str]:
    for idx, option in enumerate(options, start=1):
        prompt_color = ""
        if "Run" in option:
            prompt_color = "green"
        elif "Delete all" in option:
            prompt_color = "red"
        elif "Delete" in option:
            prompt_color = "yellow"
        elif "Show" in option:
            prompt_color = "cyan"
        elif "quit" in option:
            prompt_color = "magenta"

        if prompt_color:
            console.print(f"  [{prompt_color}]{idx}. {option}[/{prompt_color}]")
        else:
            print(f"  {idx}. {option}")

    while True:
        try:
            choice = input(f"{prompt}")
            if choice.isdigit():
                choice_int: int = int(choice)
                if 1 <= choice_int <= len(options):
                    return options[choice_int - 1]

                console.print("[red]Invalid option.[/]")
            else:
                if choice.strip() == "quit" or choice.strip() == "q":
                    sys.exit(0)
                else:
                    console.print("[red]Invalid option.[/]")
        except ValueError:
            console.print("[red]Invalid option. Please enter number.[/]")
        except EOFError:
            sys.exit(0)

def ask_confirmation() -> bool:
    try:
        response = input("Are you sure? This cannot be undone! (y/n): ").strip()
        return response in {'y', 'Y', 'j', 'J'}
    except Exception as e:
        print(f"An error occurred: {e}")

    return False

def get_value_by_condition(conn: sqlite3.Connection, table: str, field: str, search_by: str, where_column: str) -> Optional[str]:
    query = ""
    try:
        # Construct the SQL query with placeholders
        query = f"SELECT {field} FROM {table} WHERE {where_column} = ?"

        # Execute the query
        cursor = conn.cursor()
        cursor.execute(query, (search_by,))
        result = cursor.fetchone()

        # Return the value if found, otherwise None
        if result:
            return result[0]
        return None
    except Exception as e:
        if query:
            print(f"Error while fetching value from table '{table}': {e}. Query:\n{query}\n")
        else:
            print(f"Error while fetching value from table '{table}': {e}")
        return None

def list_document(conn: sqlite3.Connection, file_path: str) -> None:
    print("==================")
    print(get_value_by_condition(conn, "documents", "content", file_path, "file_path"))
    print("==================")

def list_desc(conn: sqlite3.Connection, file_path: str) -> None:
    print("==================")
    print(get_value_by_condition(conn, "image_description", "image_description", file_path, "file_path"))
    print("==================")

def list_ocr(conn: sqlite3.Connection, file_path: str) -> None:
    print("==================")
    print(get_value_by_condition(conn, "ocr_results", "extracted_text", file_path, "file_path"))
    print("==================")

def show_options_for_file(conn: sqlite3.Connection, file_path: str) -> None:
    strs = {}

    strs["show_image_again"] = "Show image again"

    strs["mark_image_as_no_face"] = "Mark image as 'contains no face'"

    strs["delete_all"] = "Delete all entries for this file"

    strs["delete_entry_no_faces"] = "Delete entries from no_faces table"
    strs["delete_ocr"] = "Delete OCR for this file"
    strs["delete_yolo"] = "Delete YOLO-Detections for this file"
    strs["delete_desc"] = "Delete descriptions for this file"
    strs["delete_face_recognition"] = "Delete face-recognition entries for this file"
    strs["delete_document"] = "Delete document entries for this file"

    strs["run_ocr"] = "Run OCR for this file"
    strs["run_yolo"] = "Run YOLO for this file"
    strs["run_face_recognition"] = "Run face recognition for this file"
    strs["run_desc"] = "Run description generation for this file"
    strs["run_document"] = "Run document generation for this file"

    strs["list_desc"] = "Show description for this file"
    strs["list_ocr"] = "Show OCR for this file"
    strs["list_document"] = "Show document content for this file"

    if is_valid_image_file(file_path):
        print(f"Options for file {file_path}:")

        display_sixel(file_path)

        while True:
            options: list[str] = []

            image_id = get_image_id_by_file_path(conn, file_path)

            if image_id is not None and check_entries_in_table(conn, "detections", image_id, "image_id"):
                options.insert(0, strs["delete_yolo"])

            if image_id is not None and check_entries_in_table(conn, "image_person_mapping", image_id, "image_id"):
                options.insert(0, strs["delete_face_recognition"])

            if check_entries_in_table(conn, "no_faces", file_path):
                options.insert(0, strs["delete_entry_no_faces"])
            else:
                options.insert(0, strs["mark_image_as_no_face"])

            if check_entries_in_table(conn, "image_description", file_path):
                options.insert(0, strs["delete_desc"])
                options.append(strs["list_desc"])

            if check_entries_in_table(conn, "ocr_results", file_path):
                options.insert(0, strs["delete_ocr"])
                options.append(strs["list_ocr"])

            options.insert(0, strs["run_desc"])
            options.insert(0, strs["run_ocr"])
            options.insert(0, strs["run_yolo"])
            options.insert(0, strs["run_face_recognition"])
            options.insert(0, strs["show_image_again"])

            options.append(strs["delete_all"])
            options.append("quit")

            option = display_menu(options)

            if option == "quit":
                sys.exit(0)
            elif option == strs["show_image_again"]:
                display_sixel(file_path)
            elif option == strs["delete_all"]:
                if ask_confirmation():
                    delete_entries_by_filename(conn, file_path)
            elif option == strs["delete_entry_no_faces"]:
                if ask_confirmation():
                    delete_no_faces_from_image_path(conn, None, file_path)
            elif option == strs["delete_desc"]:
                if ask_confirmation():
                    delete_image_description_from_image_path(conn, None, file_path)
            elif option == strs["delete_yolo"]:
                if ask_confirmation():
                    delete_yolo_from_image_path(conn, None, file_path)
            elif option == strs["delete_face_recognition"]:
                if ask_confirmation():
                    delete_faces_from_image_path(conn, None, file_path)
            elif option == strs["delete_ocr"]:
                if ask_confirmation():
                    delete_ocr_from_image_path(conn, None, file_path)

            elif option == strs["mark_image_as_no_face"]:
                if ask_confirmation():
                    delete_faces_from_image_path(conn, None, file_path)

                    insert_into_no_faces(conn, file_path)

            elif option == strs["run_desc"]:
                delete_image_description_from_image_path(conn, None, file_path)

                describe_img(conn, file_path)
            elif option == strs["run_yolo"]:
                try:
                    with console.status("[bold green]Loading yolov5..."):
                        if "yolov5" not in sys.modules:
                            import yolov5

                    model = yolov5.load(args.model)
                    model.conf = 0

                    delete_yolo_from_image_path(conn, None, file_path)

                    yolo_file(conn, file_path, None, model)
                except (FileNotFoundError, requests.exceptions.ConnectionError) as e:
                    console.print(f"[red]!!! Error while loading yolov5 model[/red]: {e}")
            elif option == strs["run_ocr"]:
                delete_ocr_from_image_path(conn, None, file_path)

                ocr_file(conn, file_path)
            elif option == strs["run_face_recognition"]:
                delete_no_faces_from_image_path(conn, None, file_path)
                delete_faces_from_image_path(conn, None, file_path)

                new_ids, manually_entered_name = recognize_persons_in_image(conn, file_path)

                if len(new_ids) and not manually_entered_name:
                    console.print(f"[green]In the following image, those persons were detected: {', '.join(new_ids)}")
                    display_sixel(file_path)
            elif option == strs["list_desc"]:
                list_desc(conn, file_path)
            elif option == strs["list_ocr"]:
                list_ocr(conn, file_path)
            else:
                console.print(f"[red]Unhandled option {option}[/]")
    elif document_already_exists(conn, file_path) or any(file_path.endswith(ext) for ext in allowed_document_extensions):
        while True:
            options: list[str] = []


            if check_entries_in_table(conn, "documents", file_path):
                options.insert(0, strs["delete_document"])
                options.append(strs["list_document"])

            options.append(strs["run_document"])

            options.append("quit")

            option = display_menu(options)

            if option == "quit":
                sys.exit(0)

            elif option == strs["list_document"]:
                list_document(conn, file_path)

            elif option == strs["delete_document"]:
                if ask_confirmation():
                    delete_document_from_document_path(conn, None, file_path)

            elif option == strs["run_document"]:
                delete_document_from_document_path(conn, None, file_path)

                insert_document_if_not_exists(conn, file_path)
            else:
                console.print(f"[red]Unhandled option {option}[/]")
    else:
        console.print(f"[red]The file {file_path} is not a valid image file. Currently, Only image files are supported.[/]")

def vacuum(conn):
    with console.status("[yellow]Vacuuming {args.dbfile}..."):
        conn.execute("VACUUM")
        console.print(f"[yellow]Vacuuming {args.dbfile} done!")

def format_text_with_keywords(text, keywords):
    def replace_placeholder(match):
        return f'__::__::__PLACEHOLDER__::__::__{match.group(0)}__::__::__PLACEHOLDER__::__::__'

    text = re.sub(r'\[.*?\](.*?)\[/.*?\]', replace_placeholder, text)

    for keyword in keywords:
        text = re.sub(rf'({re.escape(keyword)})', r'[bold reverse underline2 italic green]\1[/]', text)

    text = re.sub(r'__::__::__PLACEHOLDER__::__::__(.*?)__::__::__PLACEHOLDER__::__::__', r'\1', text)

    return text

def main() -> None:
    dbg(f"Arguments: {args}")

    shown_something = False

    conn = init_database(args.dbfile)

    existing_files = None

    if args.vacuum:
        vacuum(conn)
        shown_something = True

    if args.index or args.delete_non_existing_files:
        existing_files = load_existing_images(conn)

    if args.delete_non_existing_files:
        existing_files = delete_non_existing_image_files(conn, existing_files)

        delete_non_existing_documents(conn)

    if args.index:
        shown_something = True

        model = None

        do_all = (not args.describe and not args.ocr and not args.yolo and not args.face_recognition and not args.documents)

        if args.documents:
            traverse_document_files(conn, args.dir)

        if args.yolo:
            try:
                model = yolov5.load(args.model)
                model.conf = 0
            except (FileNotFoundError, requests.exceptions.ConnectionError) as e:
                console.print(f"[red]!!! Error while loading yolov5 model[/red]: {e}")

        image_paths = []

        with console.status(f"[bold green]Finding images in {args.dir}...") as status:
            if existing_files is not None:
                image_paths = list(find_images(existing_files))
        total_images: int = len(image_paths)

        if args.shuffle_index:
            random.shuffle(image_paths)

        if args.face_recognition or do_all:
            if supports_sixel():
                face_recognition_images: list = []

                for image_path in image_paths:
                    if not faces_already_recognized(conn, image_path):
                        face_recognition_images.append(image_path)

                c: int = 1
                for image_path in face_recognition_images:
                    console.print(f"Face recognition: {c}/{len(face_recognition_images)}")
                    if not faces_already_recognized(conn, image_path):
                        file_size = os.path.getsize(image_path)

                        if file_size < args.max_size * 1024 * 1024:
                            new_ids, manually_entered_name = recognize_persons_in_image(conn, image_path)

                            if len(new_ids) and not manually_entered_name:
                                console.print(f"[green]In the following image, those persons were detected: {', '.join(new_ids)}")
                                display_sixel(image_path)
                        else:
                            console.print(f"[yellow]The image {image_path} is too large for face recognition (), --max_size: {args.max_size}MB, file-size: ~{int(file_size / 1024 / 1024)}MB. Try increasing --max_size")
                    else:
                        console.print(f"[green]The image {image_path} was already in the index")
                    c = c + 1
            else:
                console.print("[red]Cannot use --face_recognition without a terminal that supports sixel. You could not label images without it.")

        if args.describe or args.yolo or args.ocr or do_all:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "[bold green]{task.completed}/{task.total} images",
                TimeElapsedColumn(),
                "[bold]Remaining[/]",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Indexing images...", total=total_images)

                for image_path in image_paths:
                    if os.path.exists(image_path):
                        if args.describe or do_all:
                            describe_img(conn, image_path)
                        if args.yolo:
                            if model is not None:
                                yolo_file(conn, image_path, existing_files, model)
                            else:
                                global yolo_error_already_shown

                                if not yolo_error_already_shown:
                                    console.print("[red]--yolo was set, but model could not be loaded[/]")

                                    yolo_error_already_shown = True
                        if args.ocr:
                            ocr_file(conn, image_path)
                    else:
                        console.print(f"[red]Could not find {image_path}[/]")

                    progress.update(task, advance=1)

    if args.search:
        shown_something = True

        if is_valid_file_path(args.search):
            show_options_for_file(conn, args.search)
        else:
            search(conn)

    if not shown_something:
        show_statistics(conn)

    conn.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]You pressed CTRL+C[/]")
        sys.exit(0)
