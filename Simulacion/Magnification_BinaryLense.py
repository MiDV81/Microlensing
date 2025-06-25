import sys
import os

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, lens_equation_binary_lense, magnification_binary_lense
import numpy as np

# Create a simple test case
print("Testing Binary Lens System...")

# Define binary lens parameters
m_t = 1.0      # Total mass
m_d = 0.5      # Mass difference  
z1 = 0.5       # Position of first lens

# Create array of source positions (complex numbers)
num_points = 100
x_range = np.linspace(-2, 2, num_points)
y_range = np.linspace(-2, 2, num_points)

# Create a line of source positions for testing
zeta_array = np.array([complex(x, 0.1) for x in x_range])

# Create the binary lens data object
binary_system = BinaryLens_Data(
    m_t=m_t,
    m_d=m_d,
    z1=z1,
    zeta=zeta_array
)

print(f"Created binary system:")
print(f"  Total mass (m_t): {binary_system.m_t}")
print(f"  Mass difference (m_d): {binary_system.m_d}")
print(f"  First lens position (z1): {binary_system.z1}")
print(f"  Second lens position (z2): {binary_system.z2}")
print(f"  Individual masses: m1={binary_system.m1:.3f}, m2={binary_system.m2:.3f}")
print(f"  Number of source positions: {len(binary_system.zeta)}")

# Solve the lens equation
print("\nSolving lens equation...")
binary_system = lens_equation_binary_lense(binary_system)

print(f"Found image positions for {len(binary_system.image_positions)} source positions")

# Calculate magnifications
print("Calculating magnifications...")
binary_system = magnification_binary_lense(binary_system)

print(f"Calculated magnifications: {len(binary_system.magnification)} total magnifications")

# Display some results
print(f"\nSample results:")
for i in range(min(5, len(binary_system.zeta))):
    source_pos = binary_system.zeta[i]
    images = binary_system.image_positions[i]
    total_mag = binary_system.magnification[i]
    
    print(f"  Source {i}: {source_pos:.3f}")
    print(f"    Found {len(images)} images")
    print(f"    Total magnification: {total_mag:.3f}")
    print()

# Test with a single source position
print("Testing with single source position...")
single_system = BinaryLens_Data(
    m_t=1.0,
    m_d=0.3,
    z1=0.8,
    zeta=np.array([0.1 + 0.2j])  # Single complex source position
)

single_system = lens_equation_binary_lense(single_system)
single_system = magnification_binary_lense(single_system)

print(f"Single source at {single_system.zeta[0]:.3f}")
print(f"Found {len(single_system.image_positions[0])} images:")
for j, img_pos in enumerate(single_system.image_positions[0]):
    print(f"  Image {j+1}: {img_pos:.3f}")
print(f"Total magnification: {single_system.magnification[0]:.3f}")