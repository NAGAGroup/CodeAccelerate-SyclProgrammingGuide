#!/usr/bin/env nu

# Test script for the AdaptiveCpp tutorial project
# This script runs ctest to execute the tests

# Get the project root directory (parent of scripts directory)
let project_root = ($env.CURRENT_FILE | path dirname | path dirname)

# Change to the build directory
cd $"($project_root)/build"

print "Running tests for AdaptiveCpp tutorial project..."

# Run ctest
try {
    ^ctest --output-on-failure
    print "Tests completed successfully!"
} catch {
    print $"Error: Tests failed with error: ($env.LAST_EXIT_CODE)"
    exit $env.LAST_EXIT_CODE
}