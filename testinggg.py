import time

# Capture the start time
start_time = time.time()

# Count from 0 to 1,000,000
for i in range(1000000001):
    pass  # The pass statement is just a placeholder to indicate an empty loop

# Capture the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Print the duration
print(f"Time taken to count from 0 to 1 million: {duration} seconds.")
