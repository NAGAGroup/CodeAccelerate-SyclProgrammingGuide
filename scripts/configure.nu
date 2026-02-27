#!/usr/bin/env nu

# Configure script for the AdaptiveCpp tutorial project
# This script runs CMake configuration with the appropriate settings

# Get the project root directory (parent of scripts directory)
let project_root = ($env.CURRENT_FILE | path dirname | path dirname)

# Change to the project root directory
cd $project_root

print "Configuring AdaptiveCpp tutorial project..."

# Run CMake configuration
try {
    ^cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACPP_TARGETS=generic -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    print "Configuration completed successfully!"
} catch {
    print $"Error: Configuration failed with error: ($env.LAST_EXIT_CODE)"
    exit $env.LAST_EXIT_CODE
}