#!/bin/bash

# Define base path
base_path="/nas/stid/projects/michael.wessler/nbm-powt"

# Define start and end dates
start_date="2024-10-01"
end_date="2025-04-30"

# Define forecast hours
forecast_hours=(36 48 72 120 168)

# Iterate through forecast hours first
for hour in "${forecast_hours[@]}"; do
    # Set interval based on the value of hour
    if [[ "$hour" -le 36 ]]; then
        interval=1
    else
        interval=3
    fi
        
    # Iterate through dates in one-month chunks
    current_date="$start_date"
    while [[ "$current_date" < "$end_date" ]]; do
        year=$(date -d "$current_date" +"%Y")
        month=$(date -d "$current_date" +"%m")
        formatted_date=$(date -d "$current_date" +"%Y-%m-%d")
        
        # Calculate the end-of-month date for the current month
        end_of_month=$(date -d "$year-$month-01 +1 month -1 day" +"%Y-%m-%d")
        
        # Log the Python script call for progress tracking
        echo "python $base_path/powt_archive_generator.py \
        "$formatted_date" "$end_of_month" "$hour" $interval" > $base_path/logs/progress.log
        
        # Execute the Python script and log its output
        python $base_path/powt_archive_generator.py \
        "$formatted_date" "$end_of_month" "$hour" $interval > \
        $base_path/logs/log_${formatted_date}_${end_of_month}_${hour}_${interval}.log

        # Move to the next month
        current_date=$(date -d "$year-$month-01 +1 month" +"%Y-%m-%d")
    done
done