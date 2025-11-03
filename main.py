import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


from data import rows, fields
from body import Planet
from images import load_texture


planets = [Planet(row=i) for i in rows]
points = np.array([p.position_cartesian for p in planets], dtype=np.float32)
colors = np.array([p.color for p in planets], dtype=np.float32)


class Camera:
    def __init__(self, position, yaw=0.0, pitch=0.0, speed=0.01, sensitivity=0.5):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.sensitivity = sensitivity

    def move(self, keys, dt):
        v = self.speed * dt

        # Movement modifiers
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            v *= 5.0
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            v *= 10.0
        if keys[pygame.K_SPACE]:
            v *= 10.0
        if keys[pygame.K_LALT]:
            v *= 10.0

        yaw_r = np.radians(self.yaw)
        forward = np.array([np.sin(yaw_r), 0, -np.cos(yaw_r)], dtype=np.float32)
        right = np.array([np.cos(yaw_r), 0, np.sin(yaw_r)], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        if keys[pygame.K_w]: self.position += forward * v
        if keys[pygame.K_s]: self.position -= forward * v
        if keys[pygame.K_a]: self.position -= right * v
        if keys[pygame.K_d]: self.position += right * v
        if keys[pygame.K_q]: self.position -= up * v
        if keys[pygame.K_e]: self.position += up * v

    def rotate(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = np.clip(self.pitch, -89, 89)

    def scroll_toward_mouse(self, scroll_delta, mouse_x, mouse_y, screen_width, screen_height):
        proj = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32).reshape((4,4)).T
        mv = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4,4)).T
        ray_dir = get_mouse_ray(mouse_x, mouse_y, screen_width, screen_height, proj, mv)
        self.position += ray_dir * scroll_delta * 0.5

    def apply_transform(self):
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 1, 0)
        glTranslatef(-self.position[0], -self.position[1], -self.position[2])



frustum_planes = []

def extract_frustum_planes():
    global frustum_planes
    proj = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32).reshape((4,4)).T
    mv = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape((4,4)).T
    clip = np.dot(proj, mv)

    planes = [
        clip[3, :] + clip[0, :],
        clip[3, :] - clip[0, :],
        clip[3, :] + clip[1, :],
        clip[3, :] - clip[1, :],
        clip[3, :] + clip[2, :],
        clip[3, :] - clip[2, :],
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
        dots = np.dot(pts, plane[:3]) + plane[3]
        inside &= (dots >= 0)
    return inside

def get_mouse_ray(mouse_x, mouse_y, screen_width, screen_height, projection_matrix, modelview_matrix):
    x_ndc = (2.0 * mouse_x) / screen_width - 1.0
    y_ndc = 1.0 - (2.0 * mouse_y) / screen_height
    z_ndc_near = -1.0
    z_ndc_far = 1.0

    inv_proj = np.linalg.inv(projection_matrix)
    inv_mv = np.linalg.inv(modelview_matrix)

    near_point = np.array([x_ndc, y_ndc, z_ndc_near, 1.0], dtype=np.float32)
    far_point  = np.array([x_ndc, y_ndc, z_ndc_far, 1.0], dtype=np.float32)

    near_eye = np.dot(inv_proj, near_point); near_eye /= near_eye[3]
    far_eye  = np.dot(inv_proj, far_point);  far_eye /= far_eye[3]
    near_world = np.dot(inv_mv, near_eye); near_world /= near_world[3]
    far_world  = np.dot(inv_mv, far_eye);  far_world /= far_world[3]

    ray_dir = far_world[:3] - near_world[:3]
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_dir

def points_near_camera(points, camera_position, max_distance):
    max_dist2 = max_distance * max_distance
    diffs = points - camera_position
    d2 = np.einsum('ij,ij->i', diffs, diffs)
    return d2 < max_dist2


def draw_point_cloud(pts, cols=None):
    glEnableClientState(GL_VERTEX_ARRAY)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_POINT_SMOOTH)
    glVertexPointerf(pts)
    glPointSize(3)

    if cols is not None:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerf(cols)

    glDrawArrays(GL_POINTS, 0, len(pts))

    glDisableClientState(GL_VERTEX_ARRAY)
    if cols is not None:
        glDisableClientState(GL_COLOR_ARRAY)

def draw_axes(size=1.0):
    glDisable(GL_TEXTURE_2D)
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(size,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,size,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,size)
    glEnd()
    glColor3f(1,1,1)

def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0,0,0,1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90, 800/600, 0.1, 50000.0)
    glMatrixMode(GL_MODELVIEW)

def create_sphere(radius=1.0, lat_steps=16, lon_steps=32, pole_epsilon=1e-6):
    vertices = []
    normals  = []
    tex      = []
    indices  = []


    for i in range(lat_steps + 1):
        theta = np.pi * i / lat_steps

        theta_clamped = np.clip(theta, pole_epsilon, np.pi - pole_epsilon)

        sin_t = np.sin(theta_clamped)
        cos_t = np.cos(theta_clamped)

        lat = np.pi/2 - theta_clamped

        for j in range(lon_steps + 1):
            phi = 2.0 * np.pi * j / lon_steps 

            sin_p = np.sin(phi)
            cos_p = np.cos(phi)

            x = radius * sin_t * cos_p
            y = radius * cos_t
            z = radius * sin_t * sin_p

            vertices.append([x, y, z])

            normals.append([sin_t * cos_p, cos_t, sin_t * sin_p])

            v_merc = 0.5 - (np.log(np.tan(np.pi/4 + lat/2.0)) / (2.0 * np.pi))
            v_merc = float(np.clip(1.0 - v_merc, 0.0, 1.0))

            u = 1.0 - (phi / (2.0 * np.pi))
            if u < 0.0: u += 1.0
            if u > 1.0: u -= 1.0

            tex.append([u, v_merc])

    vertices = np.array(vertices, dtype=np.float32)
    normals  = np.array(normals, dtype=np.float32)
    tex      = np.array(tex, dtype=np.float32)

    # build triangle indices (two triangles per quad)
    for i in range(lat_steps):
        for j in range(lon_steps):
            first  = i * (lon_steps + 1) + j
            second = first + (lon_steps + 1)
            indices.append([first, second, first + 1])
            indices.append([second, second + 1, first + 1])
    indices = np.array(indices, dtype=np.uint32)


    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for tri in indices:
        for idx in tri:
            glNormal3fv(normals[idx])
            glTexCoord2fv(tex[idx])
            glVertex3fv(vertices[idx])
    glEnd()
    glEndList()

    return list_id



def main():
    global points
    pygame.init()
    screen = pygame.display.set_mode((800,600), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Planet Viewer (Manual Camera)")

    init_opengl()

    points = np.array(points, dtype=np.float32).reshape(-1,3)
    sphere_mesh = create_sphere(0.05, 8, 16)

    glEnable(GL_TEXTURE_2D)
    sphere_tex = load_texture("earth.jpg")

    minp, maxp = points.min(0), points.max(0)
    center = (0,0,0)
    diag = np.linalg.norm(maxp - minp)
    cam = Camera(position=center)

    last_x, last_y = pygame.mouse.get_pos()
    clock = pygame.time.Clock()

    while True:
        dt = clock.tick()
        mouse_dx, mouse_dy = 0, 0
        scroll_delta = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    scroll_delta += 1
                elif event.button == 5:
                    scroll_delta -= 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            mouse_dx = mouse_x - last_x
            mouse_dy = mouse_y - last_y
        last_x, last_y = mouse_x, mouse_y

        if scroll_delta != 0:
            if keys[pygame.K_LCTRL]:
                scroll_delta *= 0.25
            cam.scroll_toward_mouse(scroll_delta, mouse_x, mouse_y, 800, 600)

        cam.move(keys, dt)
        cam.rotate(mouse_dx, mouse_dy)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam.apply_transform()

        extract_frustum_planes()
        mask = is_points_in_frustum(points)
        draw_point_cloud(points[mask], colors[mask])

        mask_near = points_near_camera(points, cam.position, 50.0)
        mask_to_draw_mesh = mask & mask_near


        visible_indices = np.where(mask_to_draw_mesh)[0]
        max_meshes = 500
        if len(visible_indices) > max_meshes:
            visible_indices = visible_indices[:max_meshes]

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, sphere_tex)
        for i in visible_indices:
            p = points[i]
            glPushMatrix()
            glTranslatef(*p)
            glColor3fv(colors[i])
            glCallList(sphere_mesh)
            glPopMatrix()

        draw_axes(diag * 0.2)
        pygame.display.flip()
        pygame.display.set_caption(f"{int(clock.get_fps())} FPS; position: {cam.position}")


if __name__ == "__main__":
    main()
