#!/usr/bin/env bash

set -u

if (( $# < 1 )); then
    echo "Usage: $0 \"RUN_COMMAND\"" >&2
    echo
    echo "Example:" >&2
    echo "  $0 \"./swift --hydro --threads=16 example.yml\"" >&2
    exit 1
fi

RUN_COMMAND="$1"

read -r -p "Enter the YAML configuration file: " CONFIG_FILE

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: YAML file '$CONFIG_FILE' does not exist." >&2
    exit 1
fi

BACKUP_FILE="${CONFIG_FILE}.gpu_test_backup"

if [[ -e "$BACKUP_FILE" ]]; then
    echo "Error: backup file '$BACKUP_FILE' already exists." >&2
    echo "Remove or rename it before running this script." >&2
    exit 1
fi

cp -- "$CONFIG_FILE" "$BACKUP_FILE"

restore_config() {
    if [[ -f "$BACKUP_FILE" ]]; then
        echo
        echo "Restoring original YAML file."
        cp -- "$BACKUP_FILE" "$CONFIG_FILE"
        rm -f -- "$BACKUP_FILE"
    fi
}

trap restore_config EXIT INT TERM

for gpu_tester_param in 1 2 4 8 16 32; do

    output_file="CELL_COUNT_RATIO_${gpu_tester_param}.txt"

    echo
    echo "Setting Scheduler:gpu_tester_param to ${gpu_tester_param}"
    echo "Output will be written to ${output_file}"

    # Change gpu_tester_param only inside the Scheduler section.
    # Preserve the indentation and spacing before the existing value.
    sed -Ei \
        "/^[[:space:]]*Scheduler:[[:space:]]*$/,/^[^[:space:]#][^:]*:[[:space:]]*$/ {
            s/^([[:space:]]*gpu_tester_param[[:space:]]*:[[:space:]]*)[^#[:space:]]+/\1${gpu_tester_param}/
        }" \
        "$CONFIG_FILE"

    echo "Value immediately before launching SWIFT:"

    sed -n \
       "/^[[:space:]]*Scheduler:[[:space:]]*$/,/^[^[:space:]#][^:]*:[[:space:]]*$/p" \
       "$CONFIG_FILE" |
       grep "gpu_tester_param"
    
    # Verify that the requested value appears inside the Scheduler section.
    if ! sed -n \
        "/^[[:space:]]*Scheduler:[[:space:]]*$/,/^[^[:space:]#][^:]*:[[:space:]]*$/p" \
        "$CONFIG_FILE" |
        grep -Eq \
            "^[[:space:]]*gpu_tester_param[[:space:]]*:[[:space:]]*${gpu_tester_param}([[:space:]]*(#.*)?)?$"; then

        echo \
            "Error: could not set Scheduler:gpu_tester_param to ${gpu_tester_param} in '$CONFIG_FILE'." \
            >&2

        exit 1
    fi

    {
        echo "Configuration file: ${CONFIG_FILE}"
        echo "Scheduler:gpu_tester_param: ${gpu_tester_param}"
        echo "Command: ${RUN_COMMAND}"
        echo "Started: $(date)"
        echo "============================================================"
    } | tee "$output_file"

    # Show the output on screen and save it to the results file.
    bash -lc "$RUN_COMMAND" 2>&1 | tee -a "$output_file"
    status=${PIPESTATUS[0]}

    {
        echo
        echo "============================================================"
        echo "Finished: $(date)"
        echo "Exit status: ${status}"
    } | tee -a "$output_file"

    if (( status != 0 )); then
        echo \
            "Warning: command failed for gpu_tester_param=${gpu_tester_param} with exit status ${status}." \
            >&2
    else
        echo "Completed gpu_tester_param=${gpu_tester_param}"
    fi

done

echo
echo "All gpu_tester_param tests completed."
