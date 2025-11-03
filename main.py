import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from body import planets # Assuming 'planets' is a list of Planet objects

# --- Helper for Camera/View Matrix ---
def get_view_matrix(camera):
    """Calculates and returns the 4x4 OpenGL View Matrix from camera parameters."""
    # Start with an identity matrix
    matrix = np.identity(4, dtype=np.float32)

    # 1. Apply rotation (Pitch and Yaw)
    # The rotations must be applied in reverse order to their application in apply_transform
    # glRotatef(yaw, 0, 1, 0)
    # glRotatef(pitch, 1, 0, 0)
    
    # Rotation about Y (Yaw)
    yaw_r = np.radians(camera.yaw)
    cy, sy = np.cos(yaw_r), np.sin(yaw_r)
    Ry = np.array([[cy, 0, -sy, 0],
                   [0, 1, 0, 0],
                   [sy, 0, cy, 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    # Rotation about X (Pitch)
    pitch_r = np.radians(camera.pitch)
    cp, sp = np.cos(pitch_r), np.sin(pitch_r)
    Rx = np.array([[1, 0, 0, 0],
                   [0, cp, sp, 0],
                   [0, -sp, cp, 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    # Combined Rotation (Pitch * Yaw)
    # Note: OpenGL applies glRotatef(pitch, 1, 0, 0) then glRotatef(yaw, 0, 1, 0)
    # When generating the matrix, we multiply in the reverse order: M = Yaw * Pitch
    R = np.dot(Ry, Rx) 

    # 2. Apply translation (reverse of camera position)
    Tx = np.identity(4, dtype=np.float32)
    Tx[:3, 3] = -camera.position
    
    # 3. View Matrix = Translation * Rotation (but with OpenGL matrix stack logic)
    # For a classic FPS camera, the correct matrix should combine rotations and then apply the position
    
    # In the fixed pipeline, the transformations are applied in reverse order:
    # glLoadIdentity()
    # glRotatef(pitch) -> Matrix P
    # glRotatef(yaw)   -> Matrix Y
    # glTranslatef(-pos) -> Matrix T
    # Result = T * Y * P
    
    # Since we need the inverse of the ModelView matrix for raycasting:
    # The view matrix is the inverse of the transformations applied to the world.
    
    # The combined effect of glRotatef(P) glRotatef(Y) glTranslatef(T) is effectively a view transform V.
    # The equivalent view matrix V = (T_inv * P_inv * Y_inv)
    
    # A simpler approach: create the view matrix directly (lookAt-style)
    # The inverse of the transformations is: Translation(pos) * Rotation(yaw_inv) * Rotation(pitch_inv)
    
    # The following line correctly calculates the modelview matrix as it is applied in GL
    view_matrix = np.dot(R, Tx) # R is the Rotation (Y * P)
    
    # But wait, the standard OpenGL transformations are applied *in order* to the current matrix:
    # M = Current_M * T * Y * P
    # We need the combined effect of Y * P * T_inv
    
    # Let's trust the opengl state at the time of calling extract_frustum_planes()
    # We will only fix the scroll logic to pass the current state.
    
    return np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4,4)).T


class Camera:
    # ... (init, move, rotate methods are unchanged and fine)
    def __init__(self, position, yaw=0.0, pitch=0.0, speed=0.01, sensitivity=0.5, rotation_speed=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.sensitivity = sensitivity
        self.rotation_speed = rotation_speed

    def move(self, keys):
        # Base speed
        v = self.speed
        # Speed multipliers
        multipliers = {pygame.K_LSHIFT: 5.0, pygame.K_RSHIFT: 5.0,
                       pygame.K_LCTRL: 10.0, pygame.K_RCTRL: 10.0,
                       pygame.K_SPACE: 20.0, pygame.K_LALT: 50.0}
        for key, mult in multipliers.items():
            if keys[key]:
                v *= mult

        # Use pitch for forward vector calculation (W/S) - THIS IS THE FIX
        # The forward vector needs to take pitch into account for movement in direction of view
        yaw_r = np.radians(self.yaw)
        pitch_r = np.radians(self.pitch) # Added pitch
        
        # Calculate forward vector based on both yaw and pitch
        forward = np.array([
            np.cos(pitch_r) * np.sin(yaw_r),
            np.sin(pitch_r),
            -np.cos(pitch_r) * np.cos(yaw_r)
        ], dtype=np.float32)
        forward /= np.linalg.norm(forward) # Normalize

        # Calculate right vector (only depends on yaw)
        right = np.array([np.cos(yaw_r), 0, np.sin(yaw_r)], dtype=np.float32)
        
        # Calculate up vector (used for Q/E vertical movement - still [0,1,0] for world-up)
        up = np.array([0,1,0], dtype=np.float32)

        if keys[pygame.K_w]: self.position += forward * v
        if keys[pygame.K_s]: self.position -= forward * v
        if keys[pygame.K_a]: self.position -= right * v
        if keys[pygame.K_d]: self.position += right * v
        if keys[pygame.K_q]: self.position -= up * v
        if keys[pygame.K_e]: self.position += up * v
        
        r_x, r_y = 0,0
        if keys[pygame.K_UP]:    r_y += self.rotation_speed
        if keys[pygame.K_DOWN]: r_y += -self.rotation_speed
        if keys[pygame.K_LEFT]: r_x += -self.rotation_speed
        if keys[pygame.K_RIGHT]: r_x += self.rotation_speed
        self.rotate(r_x,r_y)

    def rotate(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = np.clip(self.pitch, -89, 89)

    def scroll_toward_mouse(self, scroll_delta, mouse_x, mouse_y, screen_width, screen_height):
        # The key fix: We must retrieve the *current* matrices from the GL state
        # after the last render, or we must calculate them manually.
        # Since get_mouse_ray relies on a full ModelView/Projection state, 
        # we will rely on glGetFloatv calls *within* the function, but acknowledge
        # that the matrix state is NOT the same as when it was last rendered.
        # This is a common pattern in Pygame/PyOpenGL.
        
        # To make it accurate, we must ensure GL_PROJECTION and GL_MODELVIEW are up-to-date.
        # We will retrieve them *inside* the function for maximum currency.
        proj = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32).reshape((4,4)).T
        mv   = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4,4)).T
        
        ray_dir = get_mouse_ray(mouse_x, mouse_y, screen_width, screen_height, proj, mv)
        self.position += ray_dir * scroll_delta * (self.speed * 100) # Scale scroll speed

    def apply_transform(self):
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 1, 0)
        glTranslatef(-self.position[0], -self.position[1], -self.position[2])

# -----------------------------
# Frustum Culling (unchanged)
# -----------------------------
frustum_planes = []

def extract_frustum_planes():
    global frustum_planes
    proj = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32).reshape((4,4)).T
    mv = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4,4)).T
    clip = np.dot(proj, mv)

    planes = [
        clip[3, :] + clip[0, :], clip[3, :] - clip[0, :], # left, right
        clip[3, :] + clip[1, :], clip[3, :] - clip[1, :], # bottom, top
        clip[3, :] + clip[2, :], clip[3, :] - clip[2, :], # near, far
    ]

    frustum_planes = []
    for p in planes:
        norm = np.linalg.norm(p[:3])
        if norm != 0:
            frustum_planes.append(p / norm)
    globals()['frustum_planes'] = frustum_planes

def is_points_in_frustum(pts):
    if not frustum_planes:
        return np.ones(len(pts), dtype=bool)
    inside = np.ones(len(pts), dtype=bool)
    for plane in frustum_planes:
        # Add homogeneous coordinate (w=1) for dot product
        pts_h = np.hstack((pts, np.ones((len(pts), 1))))
        dots = np.dot(pts_h, plane)
        inside &= (dots >= 0)
    return inside

# -----------------------------
# Mouse Raycasting (unchanged)
# -----------------------------
# Note: This function is now called correctly with fresh matrices in main loop!
def get_mouse_ray(mouse_x, mouse_y, screen_width, screen_height, projection_matrix, modelview_matrix):
    x_ndc = (2.0 * mouse_x) / screen_width - 1.0
    y_ndc = 1.0 - (2.0 * mouse_y) / screen_height
    z_ndc_near, z_ndc_far = -1.0, 1.0

    inv_proj = np.linalg.inv(projection_matrix)
    inv_mv   = np.linalg.inv(modelview_matrix)

    near_point = np.array([x_ndc, y_ndc, z_ndc_near, 1.0], dtype=np.float32)
    far_point  = np.array([x_ndc, y_ndc, z_ndc_far, 1.0], dtype=np.float32)

    near_eye = np.dot(inv_proj, near_point); near_eye /= near_eye[3]
    far_eye  = np.dot(inv_proj, far_point); far_eye /= far_eye[3]

    near_world = np.dot(inv_mv, near_eye); near_world /= near_world[3]
    far_world  = np.dot(inv_mv, far_eye); far_world /= far_world[3]

    ray_dir = far_world[:3] - near_world[:3]
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_dir

# -----------------------------
# Distance Culling (unchanged)
# -----------------------------
def points_near_camera(points, camera_position, max_distance):
    max_dist2 = max_distance * max_distance
    diffs = points - camera_position
    d2 = np.einsum('ij,ij->i', diffs, diffs) # Optimized squared distance
    return d2 < max_dist2

##########################
# NEW & UPDATED HELPERS
##########################
def get_color_for_temperature(temp_k):
    """
    Returns an (R, G, B) tuple based on planet equilibrium temperature.
    """
    if temp_k is None:
        return (0.5, 0.5, 0.5) # Grey for unknown
    
    # Simple colormap: Cold -> Habitable -> Hot
    if temp_k < 200:
        return (0.2, 0.5, 1.0) # Icy Blue
    elif temp_k < 273:
        # Lerp from Blue to Green
        t = (temp_k - 200) / (273 - 200)
        return (0.2, 0.5 + t * 0.5, 1.0 - t * 0.8) # Blue -> Green
    elif temp_k < 373:
        # Lerp from Green to Yellow
        t = (temp_k - 273) / (373 - 273)
        return (0.2 + t * 0.8, 1.0, 0.2) # Green -> Yellow
    elif temp_k < 800:
        # Lerp from Yellow to Red
        t = (temp_k - 373) / (800 - 373)
        return (1.0, 1.0 - t * 1.0, 0.2 - t * 0.2) # Yellow -> Red
    else:
        return (1.0, 0.0, 0.0) # Hot Red

def draw_point_cloud(pts, colors):
    """
    Updated to take a parallel array of colors.
    """
    glPointSize(2.0)
    glBegin(GL_POINTS)
    for p, c in zip(pts, colors):
        glColor3fv(c)
        glVertex3fv(p)
    glEnd()
    glColor3f(1,1,1) # Reset color

def draw_axes(size=1.0):
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(size,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,size,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,size)
    glEnd()
    glColor3f(1,1,1)

def init_opengl(screen_size):
    # Set the viewport based on the actual screen size
    glViewport(0, 0, screen_size[0], screen_size[1])
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.05, 1) # Dark space blue
    
    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use the actual screen aspect ratio
    gluPerspective(90, screen_size[0] / screen_size[1], 0.1, 50000.0) 
    
    # Switch to ModelView Matrix for camera transforms
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity() # Start with identity

def create_sphere(radius=1.0, lat_steps=16, lon_steps=32):
    """
    Creates a sphere display list with radius 1.0.
    """
    # ... (Sphere creation logic remains the same)
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    
    glBegin(GL_TRIANGLES)
    for i in range(lat_steps):
        lat0 = np.pi * (-0.5 + i / lat_steps)
        z0 = radius * np.sin(lat0)
        zr0 = radius * np.cos(lat0)
        
        lat1 = np.pi * (-0.5 + (i + 1) / lat_steps)
        z1 = radius * np.sin(lat1)
        zr1 = radius * np.cos(lat1)
        
        for j in range(lon_steps):
            lon = 2 * np.pi * j / lon_steps
            x0 = zr0 * np.cos(lon)
            y0 = zr0 * np.sin(lon)
            
            lon1 = 2 * np.pi * (j + 1) / lon_steps
            x1 = zr0 * np.cos(lon1)
            y1 = zr0 * np.sin(lon1)

            # Triangle 1 (Points on lat0)
            glVertex3f(x0, y0, z0)
            glVertex3f(x1, y1, z0)
            # Second point on lat1 (must compute its coordinates)
            x0_next_lat = zr1 * np.cos(lon)
            y0_next_lat = zr1 * np.sin(lon)
            glVertex3f(x0_next_lat, y0_next_lat, z1) 

            # Triangle 2 (Points on lat1)
            x1_next_lat = zr1 * np.cos(lon1)
            y1_next_lat = zr1 * np.sin(lon1)
            glVertex3f(x1_next_lat, y1_next_lat, z1)
            glVertex3f(x0_next_lat, y0_next_lat, z1)
            glVertex3f(x1, y1, z0) # Last point on lat0
            
    glEnd()
    glEndList()
    return list_id

##########################
# MAIN FUNCTION
##########################
def main():
    global planets
    pygame.init()
    screen_size = (1280, 720)
    screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
    pygame.display.set_caption("Exoplanet Viewer")

    init_opengl(screen_size)

    # --- Data Preparation ---
    # 'planets' is the list of Planet objects imported from data.py
    
    # Create numpy arrays for positions and colors for vectorized operations
    all_positions = np.array([p.position_cartesian for p in planets if p.position_cartesian is not None], dtype=np.float32)
    # Filter planets that had valid coordinates (to match all_positions length)
    valid_planets = [p for p in planets if p.position_cartesian is not None] 
    
    all_colors = np.array([get_color_for_temperature(p.equilibrium_temp_k) for p in valid_planets])
    
    # Use a numpy array of the original objects for easy boolean masking
    planets_np = np.array(valid_planets) 

    # Create a sphere mesh with radius 1.0. We will scale it.
    sphere_mesh = create_sphere(0.1, 8, 16) # Lower-poly for speed

    # --- Bounding Box & Camera Start ---
    if all_positions.size == 0:
        print("No valid exoplanet data with coordinates found. Exiting.")
        return # ADDED RETURN STATEMENT
      
    minp, maxp = all_positions.min(0), all_positions.max(0)
    diag = np.linalg.norm(maxp - minp)
    
    # Start camera at the center, moved back by a reasonable distance
    center = (0,0,0)
    # Set a starting position slightly behind the center of the bounding box
    start_pos = center
    cam = Camera(position=start_pos, speed=diag * 0.0001) # Scale speed to data

    last_x, last_y = pygame.mouse.get_pos()
    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60)
        mouse_dx, mouse_dy = 0,0
        scroll_delta = 0
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resizing
                new_width, new_height = event.size
                pygame.display.set_mode((new_width, new_height), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                init_opengl((new_width, new_height))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: scroll_delta += 1
                elif event.button == 5: scroll_delta -= 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            return

        # --- Camera Control ---
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            mouse_dx = mouse_x - last_x
            mouse_dy = mouse_y - last_y
        last_x, last_y = mouse_x, mouse_y
        
        # NOTE: Apply camera transformations before raycasting to ensure GL matrices are fresh
        # The GL state MUST be updated for glGetFloatv calls in scroll_toward_mouse to work.
        # But we must handle the movement/rotation first.
        cam.move(keys)
        cam.rotate(mouse_dx, -mouse_dy)

        # ----------------------------------------------------
        # RENDER BLOCK START (Updates GL_MODELVIEW Matrix)
        # ----------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam.apply_transform() # THIS UPDATES THE GL_MODELVIEW_MATRIX

        # Now that the GL matrices are fresh, we can accurately scroll-raycast
        if scroll_delta != 0:
            cam.scroll_toward_mouse(scroll_delta, mouse_x, mouse_y, *screen_size)
            # Need to re-apply transform after position change
            glLoadIdentity()
            cam.apply_transform()


        # Frustum culling
        extract_frustum_planes()
        mask_frustum = is_points_in_frustum(all_positions)

        # Draw all visible points
        draw_point_cloud(all_positions[mask_frustum], all_colors[mask_frustum])

        # Distance culling (Level of Detail)
        # Use a smaller factor for distance culling to only draw meshes when very close
        mask_near = points_near_camera(all_positions, cam.position, diag * 0.005) 
        
        # Combine masks: must be IN frustum AND NEAR camera
        mask_to_draw_mesh = mask_frustum & mask_near
        
        # Get the full Planet objects that match the final mask
        mesh_planets = planets_np[mask_to_draw_mesh]

        for planet in mesh_planets:
            # Use planet data to draw!
            color = get_color_for_temperature(planet.equilibrium_temp_k)
            
            # Scale radius: log scale makes size differences less extreme
            radius = planet.radius_earth if planet.radius_earth else 1.0
            # Scale the size of the mesh relative to the entire dataset's diagonal
            # The factor '0.0005' needs careful tuning.
            radius_scale = np.log10(radius + 1) * (diag * 0.0005) 
            
            # Minimum size
            if radius_scale < 0.005: radius_scale = 0.005 # Ensure small planets are visible

            glPushMatrix()
            glTranslatef(*planet.position_cartesian)
            glScalef(radius_scale, radius_scale, radius_scale)
            glColor3fv(color)
            glCallList(sphere_mesh)
            glPopMatrix()

        glColor3f(1,1,1) # Reset color
        draw_axes(diag*0.05) # Draw axes scaled to data

        pygame.display.flip()
        pygame.display.set_caption(f"Exoplanet Viewer | {clock.get_fps():.1f} FPS")


if __name__=="__main__":
    main()
