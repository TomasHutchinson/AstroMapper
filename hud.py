import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_text(x, y, text, font, color=(255, 255, 255, 0), center=False):
    # Render Pygame surface
    text_surf = font.render(text, True, color)
    text_surf = text_surf.convert_alpha()
    w, h = text_surf.get_size()
    text_data = pygame.image.tostring(text_surf, "RGBA", True)

    if center: x = x - w/2

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Switch to 2D orthographic projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 800, 600, 0, -1, 1)  # top-left origin
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # Important for RGBA

    glRasterPos2f(x, y + h)  # OpenGL draws from the baseline, not top-left
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Restore matrices
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_BLEND)