# ailocate - YOLO File Indexer

ailocate is a tool for Linux that uses YOLO (You Only Look Once) to detect objects in images and creates a database of detected objects. This database is stored locally and allows you to search for specific objects in images. ailocate uses an SQLite database to efficiently store and search image data.

<p align="center">
<img src="https://raw.githubusercontent.com/NormanTUD/ailocate/refs/heads/main/images/index.gif" alt="Indexing" width="600"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/NormanTUD/ailocate/refs/heads/main/images/search.gif" alt="Indexing" width="600"/>
</p>

## Features

- Object detection in images using YOLO.
- Stores detected objects in a local SQLite database (`~/.ailocate_db`).
- Fast searching for specific objects in images.
- Supports Sixel graphics for visualizing results.
- Easy to install and use.

## Installation

1. Clone the repository:

```bash
   git clone --depth 1 https://github.com/NormanTUD/ailocate.git
```

2. Navigate to the directory and run the following command to install the tool:

```bash
cd ailocate
./ailocate --index --dir ~/Pictures
```

ailocate will automatically install all necessary dependencies, and YOLO is already included.

# Usage

## Indexing Images

To index images in a specific directory, run the following command:

```bash
ailocate --dir /path/to/images --index
```

YOLO will be used to detect objects, and the results will be stored in the database. You need to re-run the index every time new images are added or changed.

## Searching for Objects

To search for a specific object (e.g., "cat"), run the following command:

```bash
ailocate --sixel cat
```

The tool will search the indexed images for the object and display the results.

## Additional Options

- `--index`: Indexes images in the specified directory.
- `--size SIZE`: Specifies the size to which images should be resized when indexing. Default is 400.
- `--dir DIR`: Specifies the directory to search or index.
- `--debug`: Enables debug mode to output detailed logs.
- `--sixel`: Displays results as Sixel graphics.
- `--delete_non_existing_files`: Deletes non-existing files from the database.
- `--shuffle_index`: Shuffles the list of files before indexing.
- `--model MODEL`: Specifies the YOLO model for object detection.
- `--threshold THRESHOLD`: Sets the confidence threshold for object detection (0-1).
- `--dbfile DBFILE`: Specifies the path to the SQLite database file.
- `--stat [STAT]`: Displays statistics for images or a specific file.

# Example Commands

## Indexing images in a directory:

```bash
ailocate --dir /home/user/images --index
```

## Search for images containing the object "cat":

```bash
ailocate --sixel cat
```

## Indexing and debugging:

```bash
ailocate --dir /home/user/images --index --debug
```

## Display statistics for a specific image:

```bash
ailocate --stat /home/user/images/cat_picture.jpg
```

# Database

The results of image indexing are stored in the SQLite database `~/.ailocate_db`. This database contains information about detected
objects in the images. The index must be re-run whenever new images are added or changes are made.

# Requirements

- Python 3.x
- All dependencies will be automatically installed when the tool is first run.

# License

Licensed under GPL2.
