#!/bin/bash

echo "Building Computer Vision Assignment..."

# Check if OpenCV is available
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV4 not found. Please install OpenCV."
    echo "Try: sudo apt-get install libopencv-dev"
    exit 1
fi

echo "OpenCV found: $(pkg-config --modversion opencv4)"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Executables created:"
    echo "  ./build/custom_cv    - Custom implementation version"
    echo "  ./build/original_cv  - Original OpenCV version"
    echo ""
    echo "Usage:"
    echo "  cd build"
    echo "  ./custom_cv"
else
    echo "Build failed!"
    exit 1
fi