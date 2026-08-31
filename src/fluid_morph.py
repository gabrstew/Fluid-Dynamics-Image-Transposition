import taichi as ti
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import jax
import jax.numpy as jnp
from PIL import Image
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# Initialize Taichi with CPU support to avoid CUDA issues
ti.init(arch=ti.cpu, debug=False)

@ti.data_oriented
class FluidSimulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Simulation parameters
        self.dt = 8.0  # Ultra-extreme speed boost
        self.dx = 1.0
        self.dy = 1.0
        self.viscosity = 0.0  # No viscosity for faster movement
        
        # Fields for simulation
        self.density = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.density_temp = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.velocity_temp = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.transport_field = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))
        self.pressure = ti.field(dtype=ti.f32, shape=(width, height))
        self.divergence = ti.field(dtype=ti.f32, shape=(width, height))
        
        # Initialize fields
        self.density.fill(0)
        self.density_temp.fill(0)
        self.velocity.fill(0)
        self.velocity_temp.fill(0)
        self.transport_field.fill(0)
        self.pressure.fill(0)
        self.divergence.fill(0)
        
    @ti.kernel
    def advect_velocity(self):
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Compute back-traced position with RK2 (midpoint method)
                vel = self.velocity[i, j] + self.transport_field[i, j]
                pos_mid = ti.Vector([float(i), float(j)]) - 0.5 * self.dt * vel
                pos_mid = ti.math.clamp(pos_mid, 1.0, ti.Vector([float(self.width - 2), float(self.height - 2)]))
                
                # Sample velocity at midpoint
                i_mid = ti.cast(pos_mid, ti.i32)
                frac_mid = pos_mid - i_mid.cast(float)
                vel_mid = (
                    self.velocity[i_mid[0], i_mid[1]] * (1 - frac_mid[0]) * (1 - frac_mid[1]) +
                    self.velocity[i_mid[0] + 1, i_mid[1]] * frac_mid[0] * (1 - frac_mid[1]) +
                    self.velocity[i_mid[0], i_mid[1] + 1] * (1 - frac_mid[0]) * frac_mid[1] +
                    self.velocity[i_mid[0] + 1, i_mid[1] + 1] * frac_mid[0] * frac_mid[1]
                )
                
                # Final back-traced position
                pos = ti.Vector([float(i), float(j)]) - self.dt * (vel_mid + self.transport_field[i_mid[0], i_mid[1]])
                pos = ti.math.clamp(pos, 1.0, ti.Vector([float(self.width - 2), float(self.height - 2)]))
                
                # Bilinear sampling at final position
                i_pos = ti.cast(pos, ti.i32)
                frac = pos - i_pos.cast(float)
                self.velocity_temp[i, j] = (
                    self.velocity[i_pos[0], i_pos[1]] * (1 - frac[0]) * (1 - frac[1]) +
                    self.velocity[i_pos[0] + 1, i_pos[1]] * frac[0] * (1 - frac[1]) +
                    self.velocity[i_pos[0], i_pos[1] + 1] * (1 - frac[0]) * frac[1] +
                    self.velocity[i_pos[0] + 1, i_pos[1] + 1] * frac[0] * frac[1]
                )
            else:
                # No-slip boundary condition
                self.velocity_temp[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def advect_density(self):
        for i, j in self.density:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Compute back-traced position with RK2 (midpoint method)
                vel = self.velocity[i, j] + self.transport_field[i, j]
                pos_mid = ti.Vector([float(i), float(j)]) - 0.5 * self.dt * vel
                pos_mid = ti.math.clamp(pos_mid, 1.0, ti.Vector([float(self.width - 2), float(self.height - 2)]))
                
                # Sample velocity at midpoint
                i_mid = ti.cast(pos_mid, ti.i32)
                frac_mid = pos_mid - i_mid.cast(float)
                vel_mid = (
                    self.velocity[i_mid[0], i_mid[1]] * (1 - frac_mid[0]) * (1 - frac_mid[1]) +
                    self.velocity[i_mid[0] + 1, i_mid[1]] * frac_mid[0] * (1 - frac_mid[1]) +
                    self.velocity[i_mid[0], i_mid[1] + 1] * (1 - frac_mid[0]) * frac_mid[1] +
                    self.velocity[i_mid[0] + 1, i_mid[1] + 1] * frac_mid[0] * frac_mid[1]
                )
                
                # Final back-traced position
                pos = ti.Vector([float(i), float(j)]) - self.dt * (vel_mid + self.transport_field[i_mid[0], i_mid[1]])
                pos = ti.math.clamp(pos, 1.0, ti.Vector([float(self.width - 2), float(self.height - 2)]))
                
                # Cubic interpolation weights
                i_pos = ti.cast(pos, ti.i32)
                frac = pos - i_pos.cast(float)
                
                # Bilinear interpolation for better quality
                self.density_temp[i, j] = (
                    self.density[i_pos[0], i_pos[1]] * (1 - frac[0]) * (1 - frac[1]) +
                    self.density[i_pos[0] + 1, i_pos[1]] * frac[0] * (1 - frac[1]) +
                    self.density[i_pos[0], i_pos[1] + 1] * (1 - frac[0]) * frac[1] +
                    self.density[i_pos[0] + 1, i_pos[1] + 1] * frac[0] * frac[1]
                )
                
                # Apply slight diffusion to reduce numerical artifacts
                alpha = 0.001
                if i > 1 and i < self.width - 2 and j > 1 and j < self.height - 2:
                    self.density_temp[i, j] = (1 - alpha) * self.density_temp[i, j] + alpha * (
                        self.density[i+1, j] + self.density[i-1, j] +
                        self.density[i, j+1] + self.density[i, j-1]
                    ) / 4.0
            else:
                # Neumann boundary conditions for density
                if i == 0: self.density_temp[i, j] = self.density[i+1, j]
                elif i == self.width-1: self.density_temp[i, j] = self.density[i-1, j]
                elif j == 0: self.density_temp[i, j] = self.density[i, j+1]
                elif j == self.height-1: self.density_temp[i, j] = self.density[i, j-1]

    @ti.kernel
    def diffuse(self):
        alpha = self.dt * self.viscosity / (self.dx * self.dy)
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                self.velocity_temp[i, j] = (
                    self.velocity[i, j] + alpha * (
                        self.velocity[i+1, j] + self.velocity[i-1, j] +
                        self.velocity[i, j+1] + self.velocity[i, j-1]
                    )
                ) / (1 + 4 * alpha)

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                self.divergence[i, j] = -0.5 * (
                    self.velocity[i+1, j][0] - self.velocity[i-1, j][0] +
                    self.velocity[i, j+1][1] - self.velocity[i, j-1][1]
                ) / self.dx

    @ti.kernel
    def solve_pressure(self):
        for i, j in self.pressure:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                # Gauss-Seidel relaxation with over-relaxation
                w = 1.9  # Over-relaxation parameter
                self.pressure[i, j] = (1 - w) * self.pressure[i, j] + w * (
                    self.divergence[i, j] +
                    self.pressure[i+1, j] + self.pressure[i-1, j] +
                    self.pressure[i, j+1] + self.pressure[i, j-1]
                ) / 4.0
                
            # Neumann boundary conditions
            elif i == 0: self.pressure[i, j] = self.pressure[i+1, j]
            elif i == self.width-1: self.pressure[i, j] = self.pressure[i-1, j]
            elif j == 0: self.pressure[i, j] = self.pressure[i, j+1]
            elif j == self.height-1: self.pressure[i, j] = self.pressure[i, j-1]

    @ti.kernel
    def apply_pressure(self):
        for i, j in self.velocity:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                self.velocity[i, j] -= 0.5 * ti.Vector([
                    self.pressure[i+1, j] - self.pressure[i-1, j],
                    self.pressure[i, j+1] - self.pressure[i, j-1]
                ]) / self.dx

    @ti.kernel
    def force_to_target(self):
        """Force density towards target state."""
        for i, j in self.density:
            if self.transport_field[i, j].norm() > 0:
                target_pos = ti.Vector([float(i), float(j)]) + self.transport_field[i, j]
                target_pos = ti.math.clamp(target_pos, 0.0, ti.Vector([float(self.width-1), float(self.height-1)]))
                i_target = int(target_pos[0])
                j_target = int(target_pos[1])
                # Pull density towards target position strongly
                self.density[i, j] = self.density[i, j] * 0.7 + self.density[i_target, j_target] * 0.3

    def step(self):
        try:
            # Update velocity field
            self.advect_velocity()
            self.velocity.copy_from(self.velocity_temp)
            
            # Direct density advection with transport field
            self.advect_density()
            self.density.copy_from(self.density_temp)
            
            # Apply force towards target
            self.force_to_target()
        except Exception as e:
            print(f"Error in simulation step: {e}")
            return False
        return True

class OptimalTransport:
    def __init__(self):
        pass
        
    def compute_transport_field(self, source, target):
        """Ultra-fast direct transport computation."""
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.spatial import cKDTree
        
        # Get image dimensions
        h, w = source.shape[:2]
        
        # Initialize displacement field
        displacement = np.zeros((h, w, 2), dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(source.shape) == 3:
            source_gray = np.mean(source, axis=2)
            target_gray = np.mean(target, axis=2)
        else:
            source_gray = source.copy()
            target_gray = target.copy()
        
        # Find all significant points (non-black pixels)
        source_points = np.argwhere(source_gray > 0.1)
        target_points = np.argwhere(target_gray > 0.1)
        
        if len(source_points) == 0 or len(target_points) == 0:
            return displacement
            
        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(target_points)
        
        # For each source point, find nearest target point
        for y, x in source_points:
            _, idx = tree.query([y, x])
            ty, tx = target_points[idx]
            
            # Compute displacement vector
            dy = (ty - y)
            dx = (tx - x)
            
            # Apply strong force for faster transition
            magnitude = 50.0
            displacement[y, x] = [dy * magnitude, dx * magnitude]
        
        # Smooth the displacement field
        displacement = gaussian_filter(displacement, sigma=[1.0, 1.0, 0])
        
        return displacement
        
    def compute_direct_transport(self, source, target):
        """Compute transport field between source and target images."""
        return self.compute_transport_field(source, target)
        
    def sinkhorn_transport(self, source, target, epsilon=0.01, num_iters=100):
        """Legacy method, now redirects to direct transport."""
        return self.compute_direct_transport(source, target)
            
    @ti.kernel
    def simple_diffusion(self):
        """Minimal smoothing to maintain fluid appearance."""
        alpha = 0.05  # Small amount of diffusion
        for i, j in self.density:
            if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                self.density_temp[i, j] = self.density[i, j] * (1 - alpha) + alpha * (
                    self.density[i+1, j] + self.density[i-1, j] +
                    self.density[i, j+1] + self.density[i, j-1]
                ) * 0.25
                
    @ti.kernel
    def force_to_target(self):
        """Force density towards target state."""
        for i, j in self.density:
            if self.transport_field[i, j].norm() > 0:
                target_pos = ti.Vector([float(i), float(j)]) + self.transport_field[i, j]
                target_pos = ti.math.clamp(target_pos, 0.0, ti.Vector([float(self.width-1), float(self.height-1)]))
                i_target = int(target_pos[0])
                j_target = int(target_pos[1])
                # Pull density towards target position strongly
                self.density[i, j] = self.density[i, j] * 0.7 + self.density[i_target, j_target] * 0.3

class MorphingWindow(QMainWindow):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        
        # Initialize components
        self.fluid_sim = FluidSimulator(width, height)
        self.transport = OptimalTransport()
        
        # Setup UI
        self.setup_ui()
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(2)  # 500 FPS - maximum possible refresh
        
        # Transition control
        self.transition_progress = 0.0
        self.transition_speed = 0.15  # Extreme speed - complete in ~7 seconds
        
    def setup_ui(self):
        self.setWindowTitle("Fluid Morphing")
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Display area
        self.display_label = QLabel()
        layout.addWidget(self.display_label, stretch=1)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel)
        
        # Speed slider
        speed_label = QLabel("Speed:")
        control_layout.addWidget(speed_label)
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setRange(1, 100)
        speed_slider.setValue(10)
        speed_slider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(speed_slider)
        
    def update_speed(self, value):
        self.transition_speed = value / 10000.0
        
    def create_test_images(self):
        # Create sample source image (red circle)
        source = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.circle(source, (self.width//3, self.height//2), 80, (0, 0, 255), -1)
        cv2.imwrite("input.png", source)

        # Create sample target image (green square)
        target = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(target, 
                     (2*self.width//3 - 60, self.height//2 - 60),
                     (2*self.width//3 + 60, self.height//2 + 60),
                     (0, 255, 0), -1)
        cv2.imwrite("output.png", target)
        
        return source, target

    def load_images(self, source_path, target_path):
        # Try to load images, create test images if files don't exist
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        
        if source is None or target is None:
            print("Creating test images...")
            source, target = self.create_test_images()
        
        source = cv2.resize(source, (self.width, self.height))
        target = cv2.resize(target, (self.width, self.height))
        
        # Convert to float32 and normalize
        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 255.0
        
        # Initialize fluid simulation with source image
        source_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        source_field.from_numpy(source)
        self.fluid_sim.density.copy_from(source_field)
        
        # Compute direct transport field with maximum force
        displacement = self.transport.compute_direct_transport(source, target)
        transport_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.width, self.height))
        transport_field.from_numpy(displacement)
        self.fluid_sim.transport_field.copy_from(transport_field)
        
    def update_simulation(self):
        try:
            # Update simulation
            if not self.fluid_sim.step():
                return
            
            # Get current state
            density = self.fluid_sim.density.to_numpy()
            
            # Convert to QImage for display
            img_array = (np.clip(density, 0, 1) * 255).astype(np.uint8)
            height, width = img_array.shape[:2]
            
            # Create QImage
            bytes_per_line = 3 * width
            qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Display
            pixmap = QPixmap.fromImage(qimg)
            self.display_label.setPixmap(pixmap.scaled(
                self.display_label.width(),
                self.display_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            
            # Update progress
            self.transition_progress = min(1.0, self.transition_progress + self.transition_speed)
            
            # Stop if transition is complete
            if self.transition_progress >= 1.0:
                print("Transition complete")
        except Exception as e:
            print(f"Error in update_simulation: {e}")
            self.timer.stop()

if __name__ == "__main__":
    import sys
    import os
    
    app = QApplication(sys.argv)
    
    # Create window with smaller initial size for better performance
    window = MorphingWindow(256, 256)  # Reduced size for initial testing
    window.resize(800, 600)  # More manageable window size
    window.show()
    
    # Check if custom images are provided as arguments
    if len(sys.argv) > 2 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
        window.load_images(sys.argv[1], sys.argv[2])
    else:
        # Load or create default test images
        window.load_images("input.png", "output.png")
    
    sys.exit(app.exec_())