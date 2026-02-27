#!/usr/bin/env nu

# Build script for the AdaptiveCpp tutorial project
# This script runs the CMake build process

# Get the project root directory (parent of scripts directory)
let project_root = ($env.CURRENT_FILE | path dirname | path dirname)

# Change to the project root directory
cd $project_root

print "Building AdaptiveCpp tutorial project..."

# Run CMake build
try {
    ^cmake --build build --parallel
    print "Build completed successfully!"
} catch {
    print $"Error: Build failed with error: ($env.LAST_EXIT_CODE)"
    exit $env.LAST_EXIT_CODE
}