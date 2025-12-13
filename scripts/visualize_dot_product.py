#!/usr/bin/env python3
"""
Visualize Dot Product - Geometric Intuition

Run with: pixi run python scripts/visualize_dot_product.py
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_dot_product_2d(a, b, title="Dot Product Visualization"):
    """Visualize two 2D vectors and their dot product."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Vectors and angle
    ax1.set_xlim(-1, max(a[0], b[0]) + 1)
    ax1.set_ylim(-1, max(a[1], b[1]) + 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Draw vectors
    ax1.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label=f'a = [{a[0]}, {a[1]}]')
    ax1.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label=f'b = [{b[0]}, {b[1]}]')

    # Calculate dot product and angle
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)

    if mag_a > 0 and mag_b > 0:
        cos_theta = dot / (mag_a * mag_b)
        cos_theta = np.clip(cos_theta, -1, 1)  # Handle numerical errors
        theta = np.arccos(cos_theta)
        theta_deg = np.degrees(theta)
    else:
        theta_deg = 0

    # Draw angle arc
    if theta_deg > 0 and theta_deg < 180:
        angle_arc = np.linspace(0, np.radians(theta_deg), 50)
        arc_radius = 0.5
        ax1.plot(arc_radius * np.cos(angle_arc),
                arc_radius * np.sin(angle_arc),
                'g--', linewidth=2, label=f'θ = {theta_deg:.1f}°')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title}\nDot Product = {dot:.2f}')
    ax1.legend()

    # Plot 2: Projection visualization
    ax2.set_xlim(-1, max(a[0], b[0]) + 1)
    ax2.set_ylim(-1, max(a[1], b[1]) + 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)

    # Draw vectors
    ax2.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='a (to project)')
    ax2.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='b (onto)')

    # Draw projection
    if mag_b > 0:
        b_unit = b / mag_b
        projection_length = dot / mag_b
        projection = projection_length * b_unit

        ax2.quiver(0, 0, projection[0], projection[1],
                  angles='xy', scale_units='xy', scale=1,
                  color='green', width=0.015,
                  label=f'projection (length={projection_length:.2f})')

        # Draw perpendicular line from a to projection
        ax2.plot([a[0], projection[0]], [a[1], projection[1]],
                'g--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Projection of a onto b')
    ax2.legend()

    plt.tight_layout()
    return fig


def main():
    """Create multiple dot product visualizations."""

    # Example 1: Parallel vectors (maximum)
    print("\n1. Parallel Vectors (Maximum Dot Product)")
    a1 = np.array([2.0, 3.0])
    b1 = np.array([4.0, 6.0])
    fig1 = visualize_dot_product_2d(a1, b1, "Parallel Vectors")
    plt.savefig('results/dot_product_parallel.png', dpi=150, bbox_inches='tight')
    print(f"   a · b = {np.dot(a1, b1):.2f}")
    print(f"   Saved: results/dot_product_parallel.png")

    # Example 2: Perpendicular vectors (zero)
    print("\n2. Perpendicular Vectors (Zero Dot Product)")
    a2 = np.array([3.0, 0.0])
    b2 = np.array([0.0, 4.0])
    fig2 = visualize_dot_product_2d(a2, b2, "Perpendicular Vectors")
    plt.savefig('results/dot_product_perpendicular.png', dpi=150, bbox_inches='tight')
    print(f"   a · b = {np.dot(a2, b2):.2f}")
    print(f"   Saved: results/dot_product_perpendicular.png")

    # Example 3: 45 degree angle
    print("\n3. 45° Angle")
    a3 = np.array([3.0, 0.0])
    b3 = np.array([2.0, 2.0])
    fig3 = visualize_dot_product_2d(a3, b3, "45° Angle")
    plt.savefig('results/dot_product_45deg.png', dpi=150, bbox_inches='tight')
    print(f"   a · b = {np.dot(a3, b3):.2f}")
    print(f"   Saved: results/dot_product_45deg.png")

    # Example 4: Opposite vectors (negative)
    print("\n4. Opposite Direction (Negative Dot Product)")
    a4 = np.array([3.0, 2.0])
    b4 = np.array([-3.0, -2.0])
    fig4 = visualize_dot_product_2d(a4, b4, "Opposite Direction")
    plt.savefig('results/dot_product_opposite.png', dpi=150, bbox_inches='tight')
    print(f"   a · b = {np.dot(a4, b4):.2f}")
    print(f"   Saved: results/dot_product_opposite.png")

    # Example 5: Custom - You can modify these!
    print("\n5. Custom Example")
    a5 = np.array([1.0, 2.0])
    b5 = np.array([3.0, 1.0])
    fig5 = visualize_dot_product_2d(a5, b5, "Custom Vectors")
    plt.savefig('results/dot_product_custom.png', dpi=150, bbox_inches='tight')
    print(f"   a · b = {np.dot(a5, b5):.2f}")
    print(f"   Saved: results/dot_product_custom.png")

    print("\n✅ All visualizations saved to results/")
    print("\n💡 Try modifying a5 and b5 to experiment!")

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
