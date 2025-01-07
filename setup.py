from pprint import pprint
import glob
import os
import site
import sys
import sysconfig

from setuptools import setup

def is_executable_in_path(executable_name):
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

def dier (msg):
    pprint(msg)
    sys.exit(1)

# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

def do_nothing(x):
    return x

do_nothing(site.ENABLE_USER_SITE) # to trick linters...

required_programs = [
    "uuidgen",
    "python3",
    "gcc",
]

_error = 0
for rp in required_programs:
    if not is_executable_in_path(rp):
        if _error == 0:
            print("=========================== ERROR WHILE TRYING TO INSTALL SMARTLOCATE ===========================")

        print(f"Missing required program {rp}")
        _error = _error + 1

if _error >= 1:
    sys.exit(_error % 255)

# Warn if we are installing over top of an existing installation. This can
# cause issues where files that were deleted from a more recent OmniOpt2 are
# still present in site-packages. See #18115.
overlay_warning = False
if "install" in sys.argv:
    lib_paths = [sysconfig.get_path('purelib')]
    if lib_paths[0].startswith("/usr/lib/"):
        print("You need to be in a virtual environment or something similar to install this package")
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "omniopt2"))
        if os.path.exists(existing_path):
            # We note the need for the warning here, but present it after the
            # command is run, so it's more likely to be seen.
            overlay_warning = True
            break

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path, mode="r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()

def is_python_script(file_path):
    if file_path.endswith(".py"):
        return True
    return False

def is_bash_script(file_path):
    if file_path == ".env":
        return False
    try:
        with open(file_path, mode='r', encoding="utf-8") as file:
            first_line = file.readline()
            return first_line.startswith("#!") and "bash" in first_line
    except (IOError, UnicodeDecodeError):
        return False

# Alle Dateien im Home-Verzeichnis durchsuchen
all_files = glob.glob("*")
all_files.extend(glob.glob(".*"))
bash_files = [f for f in all_files if is_bash_script(f)]
python_files = [f for f in all_files if is_python_script(f)]

all_needed_files = bash_files
all_needed_files.extend(python_files)

all_needed_files.append("LICENSE")
all_needed_files.append("requirements.txt")

print("Copying the following files:")
print("=====================")
print("\n".join(all_needed_files))
print("=====================")

setup(
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    name='omniopt2',
    version='0.1',
    description='Like locate, but looks at the file contents',
    author='Norman Koch',
    author_email='norman.koch@tu-dresden.de',
    url='https://github.com/NormanTUD/smartlocate',
    install_requires=install_requires,
    packages=['.',],
    data_files=[('bin', all_needed_files)],
    include_package_data=True,
    platforms=["Linux"]
)

if overlay_warning:
    sys.stderr.write(
        """

========
WARNING!
========

You have just installed OmniOpt2 over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
OmniOpt2. This is known to cause a variety of problems. You
should manually remove the

%(existing_path)s

directory and re-install OmniOpt2.

"""
        % {"existing_path": existing_path}
    )
