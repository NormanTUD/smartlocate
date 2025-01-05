#!/bin/bash

{
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	cd "$SCRIPT_DIR"

	install_those=()

	if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
		echo "The file $SCRIPT_DIR/requirements.txt doesn't exist."
		exit 1
	fi

	while IFS= read -r line
	do
		install_those+=("$line")
	done < "$SCRIPT_DIR/requirements.txt"

	FROZEN=""

	function displaytime {
		local T=$1
		local D=$((T/60/60/24))
		local H=$((T/60/60%24))
		local M=$((T/60%60))
		local S=$((T%60))
		(( $D > 0 )) && printf '%d days ' $D
		(( $H > 0 )) && printf '%d hours ' $H
		(( $M > 0 )) && printf '%d minutes ' $M
		(( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
		printf '%d seconds\n' $S
	}

	function error_message {
		if command -v resize 2>/dev/null >/dev/null; then
			eval "$(resize)"
		fi
		MSG=$1
		echo_red "$MSG"

		if command -v whiptail 2>/dev/null >/dev/null; then
			export NEWT_COLORS='
			window=,red
			border=white,red
			textbox=white,red
			button=black,white
			'
			whiptail --title "Error Message" --scrolltext --msgbox "$MSG" $LINES $COLUMNS $(( $LINES - 8 ))
			export NEWT_COLORS=""
		else
			echo_red "Whiptail not found. Try installing it, for example, with apt-get install whiptail"
		fi
	}

	export RUN_VIA_RUNSH=1

	export PYTHONDONTWRITEBYTECODE=1

	IFS=$'\n'

	Green='\033[0;32m'
	Color_Off='\033[0m'
	Red='\033[0;31m'

	function red_text {
		echo -ne "${Red}$1${Color_Off}"
	}

	function green {
		echo -ne "${Green}$1${Color_Off}"
	}

	function _tput {
		set +e
		CHAR=$1

		if ! command -v tput 2>/dev/null >/dev/null; then
			red_text "tput not installed" >&2
			set +e
			return 0
		fi

		if [[ -z $CHAR ]]; then
			red_text "No character given" >&2
			set +e
			return 0
		fi

		if ! tty 2>/dev/null >/dev/null; then
			echo ""
			set +e
			return 0
		fi

		tput "$CHAR"
		set +e
	}

	function green_reset_line {
		_tput cr
		_tput el
		green "$1"
	}

	function red_reset_line {
		_tput cr
		_tput el
		red_text "$1"
	}

	set -e

	LMOD_DIR=/software/foundation/$(uname -m)/lmod/lmod/libexec

	myml () {
		if [[ -e $LMOD_DIR/ml_cmd ]]; then
			eval "$($LMOD_DIR/ml_cmd "$@")" 2>/dev/null >/dev/null
		fi
	}

	if [ -z "$LOAD_MODULES" ] || [ "$LOAD_MODULES" -eq 1 ]; then
		myml release/23.04 GCCcore/12.2.0 Python/3.10.8 GCCcore/11.3.0 Tkinter/3.10.4 PostgreSQL/14.4

		if [[ $(uname -m) == "ppc64le" ]]; then
			myml GCC/12.2.0 OpenBLAS/0.3.21
		fi
	fi

	_cluster=""

	if env | grep CLUSTERHOST | sed -e 's#.*\.##' 2>/dev/null >/dev/null; then
		MYCLUSTER_HOST=$(env | grep CLUSTERHOST | sed -e 's#.*\.##')
		if [[ -n $MYCLUSTER_HOST ]]; then
			_cluster="_$MYCLUSTER_HOST"
		fi
	fi

	VENV_DIR_NAME=".smartlocate_$(uname -m)_$(python3 --version | sed -e 's# #_#g')$_cluster"

	ROOT_VENV_DIR=$HOME

	if [[ -n $root_venv_dir ]] && [[ -d $root_venv_dir ]]; then
		ROOT_VENV_DIR=$root_venv_dir
	fi

	VENV_DIR=$ROOT_VENV_DIR/$VENV_DIR_NAME

	export VENV_DIR

	NUMBER_OF_INSTALLED_MODULES=0
	PROGRESSBAR=""

	generate_progress_bar_setup() {
		local total_nr_modules=$1

		NUMBER_OF_INSTALLED_MODULES=$(get_nr_of_already_installed_modules)

		if ! [[ "$NUMBER_OF_INSTALLED_MODULES" =~ ^[0-9]+$ ]]; then
			echo "Error: NUMBER_OF_INSTALLED_MODULES must be a positive integer, but is $NUMBER_OF_INSTALLED_MODULES." >&2
			return 1
		fi

		if ! [[ "$total_nr_modules" =~ ^[0-9]+$ ]]; then
			echo "Error: total_nr_modules must be a positive integer, but is $total_nr_modules." >&2
			return 1
		fi

		if [ "$NUMBER_OF_INSTALLED_MODULES" -gt "$total_nr_modules" ]; then
			echo "Error: the current progress cannot exceed the total progress ($NUMBER_OF_INSTALLED_MODULES/$total_nr_modules)." >&2
			return 1
		fi

		# Call the generate_progress_bar function to print the progress bar
		generate_progress_bar "$NUMBER_OF_INSTALLED_MODULES" "$total_nr_modules"
	}

	generate_progress_bar() {
		local current="$1"
		local max="$2"

		if ! [[ "$current" =~ ^[0-9]+$ ]]; then
			echo "Error: current must be positive integer, but is $current." >&2
			return 1
		fi

		if ! [[ "$max" =~ ^[0-9]+$ ]]; then
			echo "Error: max must be positive integer, but is $max." >&2
			return 1
		fi

		if [ "$current" -gt "$max" ]; then
			echo "Error: the current progress cannot exceed the total progress ($current/$max)." >&2
			return 1
		fi

		local bar_length=30
		local filled_length=$((bar_length * current / max))
		local empty_length=$((bar_length - filled_length))
		local percentage=$((current * 100 / max))

		local bar=""
		for ((i = 0; i < filled_length; i++)); do
			bar="${bar}━"
		done
		for ((i = 0; i < empty_length; i++)); do
			bar="${bar} "
		done

		printf "[%s] %d%%\n" "$bar" "$percentage"
	}

	function ppip {
		MODULE=$1
		AS_REQUIREMENT_OF=$2
		NUMBER_OF_MAIN_MODULES=$3

		set +e

		PROGRESSBAR=$(generate_progress_bar_setup "$NUMBER_OF_MAIN_MODULES")

		MODULES_WITHOUT_VERSIONS=$(echo "$MODULE" | sed -e 's#[=<>]=.*##' -e 's#~.*##')

		echo "$FROZEN" | grep -i "$MODULES_WITHOUT_VERSIONS" 2>/dev/null >/dev/null
		_exit_code=$?

		if [[ "$_exit_code" != "0" ]]; then
			if [[ "$MODULE" != "$AS_REQUIREMENT_OF" ]] && [[ "$AS_REQUIREMENT_OF" != "-" ]]; then
				k=0

				for i in $(pip3 install --disable-pip-version-check --dry-run "$MODULE" | grep -v "already satisfied" | grep "Collecting" | sed -e 's#Collecting ##' | grep -v "^$MODULE$"); do
					if [[ "$i" != "$MODULE" ]]; then
						if [[ $k -eq 0 ]]; then
							green_reset_line "${PROGRESSBAR}➤Installing requirements for $MODULE$(bg_jobs_str)"
						fi
						ppip "$i" "$MODULE" "$NUMBER_OF_MAIN_MODULES" || {
							red_reset_line "❌Failed to install $i."

							exit 3
						}

						k=$((k+1))
					fi
				done

				if [[ $k -gt 0 ]]; then
					green_reset_line "${PROGRESSBAR}➤Installed all requirements for $MODULE, now installing the package itself$(bg_jobs_str)..."
				fi
			fi

			green_reset_line "${PROGRESSBAR}➤Installing $MODULE$(bg_jobs_str)..."
			mkdir -p logs
			export PIP_DISABLE_PIP_VERSION_CHECK=1
			INSTALL_ERRORS_FILE="logs/install_errors"

			if [[ -n $RUN_UUID ]]; then
				INSTALL_ERRORS_FILE="logs/${RUN_UUID}_install_errors"
			fi

			pip3 --disable-pip-version-check install -q $MODULE >&2 2>> $INSTALL_ERRORS_FILE || {
				if [[ "$MODULE" == *">="* || "$MODULE" == *"<="* || "$MODULE" == *"<"* || "$MODULE" == *"=="* ]]; then
					ppip "$MODULES_WITHOUT_VERSIONS" "$AS_REQUIREMENT_OF" "$NUMBER_OF_MAIN_MODULES"
				else
					red_reset_line "❌Failed to install $MODULE. Check $INSTALL_ERRORS_FILE"
					if [[ -n $CI ]]; then
						cat "$INSTALL_ERRORS_FILE" 
					fi
					exit 3
				fi
			}

			if [ -d logs ]; then
				if [ -f "$INSTALL_ERRORS_FILE" ] && [ ! -s "$INSTALL_ERRORS_FILE" ]; then
					# Prüfen, ob das Verzeichnis leer ist (inkl. versteckter Dateien)
					if [ "$(find "logs" -mindepth 1 -type f | wc -l)" -eq 1 ]; then
						# Lösche die leere Datei und das Verzeichnis
						rm "$INSTALL_ERRORS_FILE"
						rmdir "logs"
					fi
				fi
			fi

			FROZEN=$(pip --disable-pip-version-check list --format=freeze)

			PROGRESSBAR=$(generate_progress_bar_setup "$NUMBER_OF_MAIN_MODULES")

			if [[ -z $CI ]]; then
				green_reset_line "${PROGRESSBAR}✅$MODULE installed successfully$(bg_jobs_str)"
			fi
		fi
		set -e
	}

	get_nr_of_already_installed_modules () {
		nr=0
		for key in "${install_those[@]}"; do
			noversion=$(echo "$key" | sed -e 's#[=<>]=.*##' -e 's#~.*##')

			if [[ -z $FROZEN ]]; then
				FROZEN=$(pip --disable-pip-version-check list --format=freeze)
			fi

			if [[ $noversion -eq "rich_argparse" ]]; then
				noversion="rich-argparse"
			fi

			if echo "$FROZEN" | grep -i "$noversion" 2>/dev/null >/dev/null; then
				nr=$(($nr+1))
			fi
		done

		echo "$nr"
	}

	function get_nr_bg_jobs {
		jobs -r | wc -l | tr -d " "
	}

	function bg_jobs_str {
		bg_jobs=$(get_nr_bg_jobs)

		if [[ $bg_jobs -gt 0 ]]; then
			if [[ $bg_jobs -eq 1 ]]; then
				echo " (Currently $bg_jobs background job)"
			else
				echo " (Currently $bg_jobs background jobs)"
			fi
		fi
	}

	function install_required_modules {
		green_reset_line "➤Checking environment $VENV_DIR..."
		MAX_NR="${#install_those[@]}"
		NUMBER_OF_INSTALLED_MODULES=$(get_nr_of_already_installed_modules)

		PROGRESSBAR=$(generate_progress_bar_setup "$MAX_NR")

		if [[ "$MAX_NR" -ne "$NUMBER_OF_INSTALLED_MODULES" ]]; then
			for key in "${!install_those[@]}"; do
				install_this=${install_those[$key]}
				PROGRESSBAR=$(generate_progress_bar_setup "$MAX_NR")
				if [[ -z $CI ]]; then
					green_reset_line "${PROGRESSBAR}➤Checking if $install_this is installed$(bg_jobs_str)..."
				fi

				ppip "$install_this" "-" "$MAX_NR"
			done

			_tput cr
			_tput el
		fi

		wait

		green_reset_line "✅Environment checking done!"
		_tput cr
		_tput el
	}

	required_programs=("stdbuf:coreutils" "findmnt:util-linux" "base64:base64" "curl:curl" "wget:wget" "uuidgen:uuid-runtime" "git:git" "python3:python3" "gcc:gcc" "resize:xterm" "cat:coreutils" "ls:coreutils" "wget:wget" "whiptail:whiptail" "grep:grep" "tput:ncurses-bin" "sed:sed")
	not_found_programs=0

	for cmd_pkg in "${required_programs[@]}"; do
		cmd="${cmd_pkg%%:*}"
		pkg="${cmd_pkg##*:}"

		if ! command -v "$cmd" >/dev/null 2>&1; then
			red_text "❌$cmd not found. Try installing it with 'sudo apt-get install $pkg' (depending on your distro)\n"
			not_found_programs=$(($not_found_programs+1))
		fi
	done
	
	if [[ $not_found_programs -ne 0 ]]; then
		exit 11
	fi

	if [[ "$SCRIPT_DIR" != *"$VENV_DIR"* ]]; then
		if [[ ! -d "$VENV_DIR" ]]; then
			if ! python3 -c 'from distutils.sysconfig import get_makefile_filename as m; from os.path import isfile; import sys ; sys.exit(not isfile(m()))' >/dev/null 2>/dev/null; then
				red_text "❌python3 header files not found. Try installing them, for example, with 'sudo apt-get install python3-dev' (depending on your distro)\n"
				if [[ "$OSTYPE" == "darwin"* ]]; then
					red_text "Not exiting because I am not sure if you need it on Macs"
				else
					exit 5
				fi
			fi

			green_reset_line "${PROGRESSBAR}➤Environment $VENV_DIR was not found. Creating it$(bg_jobs_str)..."
			python3 -mvenv "$VENV_DIR/" || {
				red_text "❌Failed to create Virtual Environment in $VENV_DIR"
				exit 1
			}

			green_reset_line "✅Virtual Environment $VENV_DIR created. Activating it..."

			if [[ -e "$VENV_DIR/bin/activate" ]]; then
				source "$VENV_DIR/bin/activate" || {
					red_text "❌Failed to activate $VENV_DIR"
					exit 2
				}
			else
				red_text "❌Failed to activate $VENV_DIR"
				exit 2
			fi

			green_reset_line "✅Virtual Environment activated. Now installing software. This may take some time."

		fi
	fi

	if [[ -e "$VENV_DIR/bin/activate" ]]; then
		source "$VENV_DIR/bin/activate" || {
			red_reset_line "❌Failed to activate $VENV_DIR. Deleting venv and creating it again..."
			rm -rf "$VENV_DIR"

			python3 -mvenv "$VENV_DIR/" || {
				red_text "❌Failed to create Virtual Environment in $VENV_DIR"
				rm -rf "$VENV_DIR"
				exit 1
			}

			source "$VENV_DIR/bin/activate" || {
				red_reset_line "❌Failed to activate recreated $VENV_DIR. Deleting venv and NOT trying again..."
				exit 1
			}

			install_required_modules
		}
	else
		red_reset_line "❌Failed to activate $VENV_DIR. Deleting venv and creating it again..."
		rm -rf "$VENV_DIR"

		python3 -mvenv "$VENV_DIR/" || {
			red_text "❌Failed to create Virtual Environment in $VENV_DIR"
			exit 1
		}

		if [[ -e "$VENV_DIR/bin/activate" ]]; then
			source "$VENV_DIR/bin/activate" || {
				red_reset_line "❌Failed to activate recreated $VENV_DIR. Deleting venv and NOT trying again..."
				rm -rf "$VENV_DIR"
				exit 1
			}

			downgrade_output=$(pip3 --disable-pip-version-check install -q pip==24.0) || {
				red_text "Failed to downgrade pip. Output:"
				red_text "$downgrade_output"
			}
		else
			red_reset_line "❌Failed to activate recreated $VENV_DIR. Deleting venv and NOT trying again..."
			rm -rf "$VENV_DIR"
			exit 1
		fi

		install_required_modules

	fi

	if [[ -z $DONT_INSTALL_MODULES ]]; then
		if [[ -z $SLURM_JOB_ID ]]; then
			set +e
			FROZEN=$(pip --disable-pip-version-check list --format=freeze)
			exit_code_pip=$?
			set -e

			if [[ "$exit_code_pip" -ne "0" ]]; then
				printf "pip list --format=freeze exited with exit code %s\n" $exit_code_pip
				exit 12
			fi

			install_required_modules
		fi
	else
		if [[ -z $DONT_SHOW_DONT_INSTALL_MESSAGE ]]; then
			red_text "\$DONT_INSTALL_MODULES is set. Don't install modules.\n"
		fi
	fi

	export PYTHONPATH=$VENV_DIR:$PYTHONPATH
}
