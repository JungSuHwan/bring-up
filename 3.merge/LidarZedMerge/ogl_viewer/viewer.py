from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from threading import Lock
import numpy as np
import sys
import array
import math
import ctypes
import pyzed.sl as sl

M_PI = 3.1415926

POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
void main() {
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

POINT_FRAGMENT_SHADER = """
#version 330 core
uniform vec3 u_color;
layout(location = 0) out vec4 color;
void main() {
   color = vec4(u_color, 1);
}
"""


MESH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
    b_color = u_color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FPC_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec4 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
   b_color = u_color;
   gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);
}
"""

VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 b_color;
layout(location = 0) out vec4 color;
void main() {
   color = vec4(b_color,1);
}
"""

class Shader:
    def __init__(self, _vs, _fs):

        self.program_id = glCreateProgram()
        vertex_id = self.compile(GL_VERTEX_SHADER, _vs)
        fragment_id = self.compile(GL_FRAGMENT_SHADER, _fs)

        glAttachShader(self.program_id, vertex_id)
        glAttachShader(self.program_id, fragment_id)
        glBindAttribLocation( self.program_id, 0, "in_vertex")
        glBindAttribLocation( self.program_id, 1, "in_texCoord")
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            if (self.program_id is not None) and (self.program_id > 0) and glIsProgram(self.program_id):
                glDeleteProgram(self.program_id)
            if (vertex_id is not None) and (vertex_id > 0) and glIsShader(vertex_id):
                glDeleteShader(vertex_id)
            if (fragment_id is not None) and (fragment_id > 0) and glIsShader(fragment_id):
                glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        if (vertex_id is not None) and (vertex_id > 0) and glIsShader(vertex_id):
            glDeleteShader(vertex_id)
        if (fragment_id is not None) and (fragment_id > 0) and glIsShader(fragment_id):
            glDeleteShader(fragment_id)

    def compile(self, _type, _src):
        try:
            shader_id = glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            glShaderSource(shader_id, _src)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                if (shader_id is not None) and (shader_id > 0) and glIsShader(shader_id):
                    glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            if (shader_id is not None) and (shader_id > 0) and glIsShader(shader_id):
                glDeleteShader(shader_id)
            raise

    def get_program_id(self):
        return self.program_id

IMAGE_FRAGMENT_SHADER = """
#version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
uniform bool revert;
uniform bool rgbflip;
void main() {
    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);
    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);
    color = vec4(rgbcolor,1);
}
"""

IMAGE_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
    UV = (vert.xy+vec2(1,1))/2;
    gl_Position = vec4(vert, 1);
}
"""

class ImageHandler:
    """
    Class that manages the image stream to render with OpenGL
    """
    def __init__(self):
        self.tex_id = 0
        self.tex_rgb = 0
        self.tex_depth = 0
        self.quad_vb = 0
        self.is_called = 0

    def close(self):
        if self.tex_rgb:
            self.tex_rgb = 0
        if self.tex_depth:
            self.tex_depth = 0

    def initialize(self, _res):    
        self.shader_image = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.tex_id = glGetUniformLocation( self.shader_image.get_program_id(), "texImage")

        g_quad_vertex_buffer_data = np.array([-1, -1, 0,
                                                1, -1, 0,
                                                -1, 1, 0,
                                                -1, 1, 0,
                                                1, -1, 0,
                                                1, 1, 0], np.float32)

        self.quad_vb = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glBufferData(GL_ARRAY_BUFFER, g_quad_vertex_buffer_data.nbytes,
                     g_quad_vertex_buffer_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glEnable(GL_TEXTURE_2D)

        # Create RGB Texture
        self.tex_rgb = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_rgb)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _res.width, _res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        # Create Depth Texture
        self.tex_depth = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_depth)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _res.width, _res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        glBindTexture(GL_TEXTURE_2D, 0)   

    def update_texture(self, tex_id, ptr, w, h):
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, ptr)
        glBindTexture(GL_TEXTURE_2D, 0)            

    def draw(self, tex_id):
        glUseProgram(self.shader_image.get_program_id())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(self.tex_id, 0)

        # invert y axis and color for this image
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "revert"), 1)
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "rgbflip"), 1)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)            
        glUseProgram(0)

class PointHandler:
    def __init__(self):
        self.vbo = None
        self.count = 0
        self.shader = None
        self.color = [1.0, 0.0, 0.0] # Red

    def initialize(self):
        self.shader = Shader(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        self.u_mvp = glGetUniformLocation(self.shader.get_program_id(), "u_mvpMatrix")
        self.u_color = glGetUniformLocation(self.shader.get_program_id(), "u_color")
        self.vbo = glGenBuffers(1)

    def update(self, points):
        # points: list of float [x, y, z, x, y, z, ...]
        if not points:
            self.count = 0
            return
            
        data = np.array(points, dtype=np.float32)
        self.count = len(points) // 3
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mvp, color_override=None):
        if self.count == 0: return
        
        glUseProgram(self.shader.get_program_id())
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, (GLfloat * len(mvp))(*mvp))
        color = color_override if color_override is not None else self.color
        glUniform3fv(self.u_color, 1, color)
        
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        glDrawArrays(GL_POINTS, 0, self.count)
        
        glDisableVertexAttribArray(0)
        glUseProgram(0)


class LineLoopHandler:
    def __init__(self):
        self.vbo = None
        self.count = 0
        self.shader = None
        self.color = [1.0, 0.0, 0.0]
        self.width = 2.0

    def initialize(self):
        self.shader = Shader(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        self.u_mvp = glGetUniformLocation(self.shader.get_program_id(), "u_mvpMatrix")
        self.u_color = glGetUniformLocation(self.shader.get_program_id(), "u_color")
        self.vbo = glGenBuffers(1)

    def update(self, points):
        if not points:
            self.count = 0
            return
        data = np.array(points, dtype=np.float32)
        self.count = len(points) // 3
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mvp, color_override=None, width_override=None):
        if self.count < 3:
            return
        glUseProgram(self.shader.get_program_id())
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, (GLfloat * len(mvp))(*mvp))
        color = color_override if color_override is not None else self.color
        glUniform3fv(self.u_color, 1, color)
        line_w = float(width_override) if width_override is not None else float(self.width)
        glLineWidth(line_w)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_LINE_LOOP, 0, self.count)
        glDisableVertexAttribArray(0)
        glUseProgram(0)


class FillPolygonHandler:
    def __init__(self):
        self.vbo = None
        self.count = 0
        self.shader = None
        self.color = [1.0, 0.0, 0.0]

    def initialize(self):
        self.shader = Shader(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        self.u_mvp = glGetUniformLocation(self.shader.get_program_id(), "u_mvpMatrix")
        self.u_color = glGetUniformLocation(self.shader.get_program_id(), "u_color")
        self.vbo = glGenBuffers(1)

    def update(self, points):
        if not points:
            self.count = 0
            return
        data = np.array(points, dtype=np.float32)
        self.count = len(points) // 3
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mvp, color_override=None):
        if self.count < 3:
            return
        glUseProgram(self.shader.get_program_id())
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, (GLfloat * len(mvp))(*mvp))
        color = color_override if color_override is not None else self.color
        glUniform3fv(self.u_color, 1, color)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.count)
        glDisableVertexAttribArray(0)
        glUseProgram(0)


class GLViewer:
    """
    Class that manages the rendering in OpenGL
    """
    def __init__(self):
        self.available = False
        self.mutex = Lock()
        self.draw_mesh = False
        self.new_chunks = False
        self.chunks_pushed = False
        self.change_state = False
        self.projection = sl.Matrix4f()
        self.projection.set_identity()
        self.znear = 0.5
        self.zfar = 100.
        self.image_handler = ImageHandler()
        self.sub_maps = []
        self.pose = sl.Transform().set_identity()
        self.tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
        self.mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
        self.command_callback = None
        self.control_status_text = ""
        self.layout_left_ratio = 0.42
        self.layout_rgb_ratio = 0.64
        self.lidar_handlers = {}
        self.lidar_alert_handlers = {}
        self.lidar_order = []
        self.lidar_status = {}
        self.robot_overlay_enabled = False
        self.robot_body_fill = None
        self.robot_body_outline = None
        self.robot_heading_fill = None
        self.lidar_palette = [
            [0.20, 0.85, 0.20],  # green
            [0.20, 0.60, 1.00],  # blue
            [1.00, 0.75, 0.20],  # orange
            [0.95, 0.20, 0.95],  # magenta
            [0.20, 0.95, 0.95],  # cyan
            [0.70, 0.85, 1.00],  # light blue
        ]
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pan_sensitivity = 0.005  # meter / pixel
        self.zoom_sensitivity = 0.20  # meter / wheel step
        self.drag_active = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.rgb_aspect_ratio = 16.0 / 9.0
        self.lidar_2d_viewport = (0, 0, 1, 1)
        self.lidar_2d_ortho = (-3.0, 3.0, -6.0, 1.0)  # left, right, bottom(z), top(z)
        self.lidar_2d_hover_valid = False
        self.lidar_2d_hover_x = 0.0
        self.lidar_2d_hover_z = 0.0
        self.lidar_2d_hover_dist = 0.0
        self.lidar_2d_hover_angle_deg = 0.0
        self.lidar_2d_hover_nx = 0.0
        self.lidar_2d_hover_ny = 0.0

    def init(self, _params, _mesh, _create_mesh, show_window=True): 
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = wnd_w*0.9
        height = wnd_h*0.9
     
        if width > _params.image_size.width and height > _params.image_size.height:
            width = _params.image_size.width
            height = _params.image_size.height

        glutInitWindowSize(int(width), int(height))
        glutInitWindowPosition(0, 0) # The window opens at the upper left corner of the screen
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)
        glutCreateWindow(b"ZED Spatial Mapping")
        if not show_window:
            glutHideWindow()
        glViewport(0, 0, int(width), int(height))

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Initialize image renderer
        self.image_handler.initialize(_params.image_size)
        if _params.image_size.height > 0:
            self.rgb_aspect_ratio = float(_params.image_size.width) / float(_params.image_size.height)
        
        self.init_mesh(_mesh, _create_mesh)

        # Compile and create the shader 
        if(self.draw_mesh):
            self.shader_image = Shader(MESH_VERTEX_SHADER, FRAGMENT_SHADER)
        else:
            self.shader_image = Shader(FPC_VERTEX_SHADER, FRAGMENT_SHADER)

        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        self.shader_color_loc = glGetUniformLocation(self.shader_image.get_program_id(), "u_color")
        # Create the rendering camera
        self.set_render_camera_projection(_params)

        self.robot_body_fill = FillPolygonHandler()
        self.robot_body_fill.initialize()
        self.robot_body_fill.color = [0.90, 0.90, 0.95]
        self.robot_body_outline = LineLoopHandler()
        self.robot_body_outline.initialize()
        self.robot_body_outline.color = [0.10, 0.20, 0.95]
        self.robot_body_outline.width = 2.5
        self.robot_heading_fill = FillPolygonHandler()
        self.robot_heading_fill.initialize()
        self.robot_heading_fill.color = [0.95, 0.90, 0.20]

        glLineWidth(1.)
        glPointSize(4.)

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)   

        glutKeyboardUpFunc(self.keyReleasedCallback)
        glutMouseFunc(self.mouse_button_callback)
        glutMotionFunc(self.mouse_motion_callback)
        glutPassiveMotionFunc(self.mouse_passive_motion_callback)
        if hasattr(sys.modules[__name__], "glutMouseWheelFunc"):
            try:
                glutMouseWheelFunc(self.mouse_wheel_callback)
            except Exception:
                pass

        # Register the closing function
        glutCloseFunc(self.close_func)

        self.ask_clear = False
        self.available = True

        # Set color for wireframe
        self.vertices_color = [0.12,0.53,0.84] 
        
        # Ready to start
        self.chunks_pushed = True

    def init_mesh(self, _mesh, _create_mesh):
        self.draw_mesh = _create_mesh
        self.mesh = _mesh

    def set_render_camera_projection(self, _params):
        # Just slightly move up the ZED camera FOV to make a small black border
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        self.projection[(0,0)] = 1. / math.tan(fov_x * .5)
        self.projection[(1,1)] = 1. / math.tan(fov_y * .5)
        self.projection[(2,2)] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        self.projection[(3,2)] = -1.
        self.projection[(2,3)] = -(2. * self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection[(3,3)] = 0.
    
    def print_GL(self, _x, _y, _string):
        glRasterPos(_x, _y)
        for i in range(len(_string)):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(_string[i])))

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF:
            return True
        else:
            return False

    def update_chunks(self):
        self.new_chunks = True
        self.chunks_pushed = False
    
    def chunks_updated(self):
        return self.chunks_pushed

    def clear_current_mesh(self):
        self.ask_clear = True
        self.new_chunks = True

    def _ensure_lidar_handler(self, name):
        if name in self.lidar_handlers:
            return self.lidar_handlers[name]

        handler = PointHandler()
        handler.initialize()
        color_idx = len(self.lidar_order) % len(self.lidar_palette)
        handler.color = self.lidar_palette[color_idx]

        self.lidar_handlers[name] = handler
        self.lidar_order.append(name)
        return handler

    def _ensure_lidar_alert_handler(self, name):
        if name in self.lidar_alert_handlers:
            return self.lidar_alert_handlers[name]
        handler = PointHandler()
        handler.initialize()
        handler.color = [1.0, 0.0, 0.0]  # red for threshold hit points
        self.lidar_alert_handlers[name] = handler
        return handler

    def update_lidar(self, points):
        self.update_lidar_multi([{
            "name": "lidar",
            "points": points if points is not None else [],
            "connected": True,
        }])

    def update_lidar_multi(self, lidar_frames):
        self.mutex.acquire()
        seen_names = set()

        for frame in lidar_frames:
            name = str(frame.get("name", "lidar"))
            points = frame.get("points", [])
            alert_points = frame.get("alert_points", [])
            connected = bool(frame.get("connected", False))
            fps = float(frame.get("fps", 0.0))
            offset = frame.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0})
            yaw_deg = float(frame.get("yaw_deg", 0.0))

            handler = self._ensure_lidar_handler(name)
            alert_handler = self._ensure_lidar_alert_handler(name)
            handler.update(points)
            alert_handler.update(alert_points)
            self.lidar_status[name] = {
                "connected": connected,
                "point_count": len(points) // 3,
                "alert_point_count": len(alert_points) // 3,
                "fps": fps,
                "offset": {
                    "x": float(offset.get("x", 0.0)),
                    "y": float(offset.get("y", 0.0)),
                    "z": float(offset.get("z", 0.0)),
                },
                "yaw_deg": yaw_deg,
            }
            seen_names.add(name)

        for name in self.lidar_order:
            if name not in seen_names:
                self.lidar_handlers[name].update([])
                self.lidar_alert_handlers[name].update([])
                status = self.lidar_status.get(name, {})
                status["point_count"] = 0
                status["alert_point_count"] = 0
                status["fps"] = float(status.get("fps", 0.0))
                self.lidar_status[name] = status

        self.mutex.release()

    def update_view(self, _image, _depth_ptr, _pose, _tracking_state, _mapping_state):     
        self.mutex.acquire()
        if self.available:
            # update image
            # self.image_handler.push_new_image(_image)
            if _image is not None:
                self.image_handler.update_texture(self.image_handler.tex_rgb, ctypes.c_void_p(_image.get_pointer()), _image.get_width(), _image.get_height())
            
            if _depth_ptr is not None and _image is not None:
                self.image_handler.update_texture(self.image_handler.tex_depth, _depth_ptr, _image.get_width(), _image.get_height())

            self.pose = _pose
            self.tracking_state = _tracking_state
            self.mapping_state = _mapping_state
        self.mutex.release()
        copy_state = self.change_state
        self.change_state = False
        return copy_state

    def set_command_callback(self, callback):
        self.command_callback = callback

    def set_control_status(self, text):
        with self.mutex:
            self.control_status_text = str(text) if text is not None else ""

    def set_robot_overlay(self, body_points_world, heading_points_world, enabled=True):
        with self.mutex:
            self.robot_overlay_enabled = bool(enabled)
            if not self.robot_overlay_enabled:
                if self.robot_body_fill is not None:
                    self.robot_body_fill.update([])
                if self.robot_body_outline is not None:
                    self.robot_body_outline.update([])
                if self.robot_heading_fill is not None:
                    self.robot_heading_fill.update([])
                return
            if self.robot_body_fill is not None:
                self.robot_body_fill.update(body_points_world if body_points_world else [])
            if self.robot_body_outline is not None:
                self.robot_body_outline.update(body_points_world if body_points_world else [])
            if self.robot_heading_fill is not None:
                self.robot_heading_fill.update(heading_points_world if heading_points_world else [])

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):      
        if self.available:
            self.available = False
            self.image_handler.close()

    def close_func(self): 
        if self.available:
            self.available = False
            self.image_handler.close()      

    def keyReleasedCallback(self, key, x, y):
        try:
            ch = key.decode("utf-8").lower() if isinstance(key, (bytes, bytearray)) else chr(ord(key)).lower()
        except Exception:
            return
        try:
            key_code = key[0] if isinstance(key, (bytes, bytearray)) else ord(key)
        except Exception:
            key_code = -1

        if ch == "q" or key_code == 27:   # 'q' or ESC
            self.close_func()
            return
        if ch == "r":
            self.reset_pan_zoom()
            return
        if ch == " ":
            self.change_state = True
            return

        if not self.command_callback:
            return

        key_map = {
            "s": "save_spatial_map",
            "n": "select_prev_lidar",
            "m": "select_next_lidar",
            "h": "offset_x_minus",
            "l": "offset_x_plus",
            "u": "offset_y_plus",
            "o": "offset_y_minus",
            "j": "offset_z_minus",
            "k": "offset_z_plus",
            "[": "offset_step_down",
            "]": "offset_step_up",
            "0": "reset_selected_lidar_offset",
            ",": "yaw_minus",
            ".": "yaw_plus",
            ";": "yaw_step_down",
            "'": "yaw_step_up",
            "9": "reset_selected_lidar_yaw",
        }
        action = key_map.get(ch)
        if action:
            try:
                self.command_callback(action, {})
            except Exception:
                pass

    def pan_by_pixels(self, dx, dy):
        with self.mutex:
            self.pan_offset[0] += float(dx) * self.pan_sensitivity
            self.pan_offset[1] -= float(dy) * self.pan_sensitivity

    def zoom_by_steps(self, steps):
        with self.mutex:
            self.pan_offset[2] += float(steps) * self.zoom_sensitivity

    def reset_pan_zoom(self):
        with self.mutex:
            self.pan_offset[:] = 0.0

    def _is_in_3d_viewport(self, mouse_x):
        wnd_w = glutGet(GLUT_WINDOW_WIDTH)
        left_w = int(wnd_w / 3)
        return mouse_x >= left_w

    def mouse_button_callback(self, button, state, x, y):
        # FreeGLUT commonly reports wheel as mouse buttons 3(up) / 4(down).
        if state == GLUT_DOWN and self._is_in_3d_viewport(x):
            if button == 3:
                self.zoom_by_steps(1.0)
                return
            if button == 4:
                self.zoom_by_steps(-1.0)
                return

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN and self._is_in_3d_viewport(x):
                self.drag_active = True
                self.last_mouse_x = x
                self.last_mouse_y = y
            elif state == GLUT_UP:
                self.drag_active = False

    def mouse_wheel_callback(self, wheel, direction, x, y):
        if not self._is_in_3d_viewport(x):
            return
        if direction > 0:
            self.zoom_by_steps(1.0)
        elif direction < 0:
            self.zoom_by_steps(-1.0)

    def mouse_motion_callback(self, x, y):
        self._update_lidar_2d_hover_from_mouse(x, y)
        if not self.drag_active:
            return

        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        self.last_mouse_x = x
        self.last_mouse_y = y

        self.pan_by_pixels(dx, dy)

    def mouse_passive_motion_callback(self, x, y):
        self._update_lidar_2d_hover_from_mouse(x, y)

    def _update_lidar_2d_hover_from_mouse(self, mouse_x, mouse_y):
        wnd_h = int(glutGet(GLUT_WINDOW_HEIGHT))
        if wnd_h <= 0:
            return
        with self.mutex:
            vx, vy, vw, vh = self.lidar_2d_viewport
            if vw <= 0 or vh <= 0:
                self.lidar_2d_hover_valid = False
                return
            y_gl = wnd_h - int(mouse_y) - 1
            inside = (int(mouse_x) >= vx) and (int(mouse_x) < (vx + vw)) and (y_gl >= vy) and (y_gl < (vy + vh))
            if not inside:
                self.lidar_2d_hover_valid = False
                return
            l, r, b, t = self.lidar_2d_ortho
            nx = (float(mouse_x) - float(vx)) / float(vw)
            ny = (float(y_gl) - float(vy)) / float(vh)
            world_x = float(l) + nx * float(r - l)
            world_z = float(b) + ny * float(t - b)
            dist = math.sqrt((world_x * world_x) + (world_z * world_z))
            # Inverse of local mapping used in lidar_thread.py:
            # x = r*sin(theta), z = -r*cos(theta) -> theta = atan2(x, -z)
            angle_deg = math.degrees(math.atan2(world_x, -world_z))
            self.lidar_2d_hover_x = world_x
            self.lidar_2d_hover_z = world_z
            self.lidar_2d_hover_dist = dist
            self.lidar_2d_hover_angle_deg = angle_deg
            self.lidar_2d_hover_nx = nx
            self.lidar_2d_hover_ny = ny
            self.lidar_2d_hover_valid = True

    def draw_lidar_2d_hover_overlay(self):
        vx, vy, vw, vh = self.lidar_2d_viewport
        if vw <= 0 or vh <= 0:
            return
        if not self.lidar_2d_hover_valid:
            return
        glViewport(vx, vy, vw, vh)
        glColor3f(0.92, 0.92, 0.92)
        px = -1.0 + (2.0 * float(self.lidar_2d_hover_nx))
        py = -1.0 + (2.0 * float(self.lidar_2d_hover_ny))
        # Keep label near cursor but inside viewport.
        tx = min(0.72, max(-0.98, px + 0.03))
        ty = min(0.95, max(-0.92, py + 0.05))
        self.print_GL(tx, ty, f"D:{self.lidar_2d_hover_dist:.2f}m A:{self.lidar_2d_hover_angle_deg:+.1f}deg")

    def _lidar_angle_to_world_xz(self, angle_deg, radius):
        rad = math.radians(float(angle_deg))
        x = float(radius) * math.sin(rad)
        z = -float(radius) * math.cos(rad)
        return x, z

    def draw_lidar_2d_fov_overlay(self):
        l, r, b, t = self.lidar_2d_ortho
        a0 = -45.0
        a1 = 225.0

        ray_len = max(0.2, min(abs(l), abs(r), abs(b)) * 0.75)
        arc_r = max(0.2, min(abs(l), abs(r), abs(b), abs(t)) * 0.85)

        x0, z0 = self._lidar_angle_to_world_xz(a0, ray_len)
        x1, z1 = self._lidar_angle_to_world_xz(a1, ray_len)

        # Boundary rays
        glLineWidth(1.5)
        glColor3f(0.95, 0.82, 0.20)
        glBegin(GL_LINES)
        glVertex2f(0.0, 0.0)
        glVertex2f(float(x0), float(z0))
        glVertex2f(0.0, 0.0)
        glVertex2f(float(x1), float(z1))
        glEnd()

        # Scan arc from -45 to 225 deg
        glLineWidth(1.2)
        glColor3f(0.90, 0.70, 0.18)
        glBegin(GL_LINE_STRIP)
        step = 3
        for ang in range(int(a0), int(a1) + 1, step):
            xa, za = self._lidar_angle_to_world_xz(float(ang), arc_r)
            glVertex2f(float(xa), float(za))
        glEnd()

        # Labels near arc endpoints
        tx0, tz0 = self._lidar_angle_to_world_xz(a0, arc_r * 1.12)
        tx1, tz1 = self._lidar_angle_to_world_xz(a1, arc_r * 1.12)
        glColor3f(0.95, 0.90, 0.80)
        self.print_GL(float(tx0), float(tz0), "-45")
        self.print_GL(float(tx1), float(tz1), "225")

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.2, 0.2, 0.2, 1.0) # slightly lighter visual background

            self.mutex.acquire()
            self.update()
            
            wnd_w = glutGet(GLUT_WINDOW_WIDTH)
            wnd_h = glutGet(GLUT_WINDOW_HEIGHT)
            
            # Split Layout
            # Left 1/3 for Images, Right 2/3 for Mesh
            left_w = int(wnd_w * self.layout_left_ratio)
            right_w = wnd_w - left_w
            # 1. Draw Mesh (Right Side)
            glViewport(left_w, 0, right_w, wnd_h)
            self.draw_3d_mesh()
            
            # 2. Draw 2D Images (Left Side)
            rgb_h = int(wnd_h * self.layout_rgb_ratio)
            lidar_h = wnd_h - rgb_h

            # Top-Left: RGB
            rgb_vp = self._fit_viewport_keep_aspect(0, lidar_h, left_w, rgb_h, self.rgb_aspect_ratio)
            glViewport(rgb_vp[0], rgb_vp[1], rgb_vp[2], rgb_vp[3])
            self.image_handler.draw(self.image_handler.tex_rgb)

            # Bottom-Left: LiDAR 2D View
            glViewport(0, 0, left_w, lidar_h)
            self.lidar_2d_viewport = (0, 0, int(left_w), int(lidar_h))
            
            # Setup Orthographic Projection (Top-Down View)
            # Range: +/- 5 meters
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            # Left, Right, Bottom, Top
            # X is Left/Right (-5, 5), Z is Forward/Backward (-10, 0) -> mapped to Y
            # Actually just map X->X, Z->Y for display
            # Range: Wide view
            l, r, b, t = self.lidar_2d_ortho
            glOrtho(l, r, b, t, -1, 1) # X: -3m~3m, Z: -6m~1m (1m behind camera)
            
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Draw Grid
            glLineWidth(1.0)
            glBegin(GL_LINES)
            glColor3f(0.3, 0.3, 0.3)
            # Vertical lines (Range markers)
            for i in range(-5, 2):
                glVertex2f(-5.0, float(i))
                glVertex2f( 5.0, float(i))
            # Horizontal lines (Direction)
            for i in range(-5, 6):
                glVertex2f(float(i), -6.0)
                glVertex2f(float(i),  1.0)
            
            # Forward Vector (Z axis negative)
            glColor3f(0.0, 1.0, 0.0)
            glVertex2f(0.0, 0.0)
            glVertex2f(0.0, -1.0)
            glEnd()
            self.draw_lidar_2d_fov_overlay()
            
            # Draw LiDAR Points (Project 3D points to 2D: X, Z)
            # Since point_handler uses a shader with MVP, we need to manually draw or transform
            # But here we are in fixed function pipeline mode for grid.
            # Let's use simple glBegin(GL_POINTS) for the 2D view as we have access to raw points via buffer? No.
            # We don't have raw points easily here unless we store them.
            # Hack: Use point_handler with an orthographic MVP matrix.
            
            # Top-Down MVP:
            # View: Camera at (0, 10, 0) looking at (0, 0, -3), Up (0, 0, -1)
            # Proj: Ortho
            pass 
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
            # Use PointHandler for 2D View (Top Down)
            # Construct simple ortho matrix manually or use GLU?
            # Creating a matrix manually for the shader:
            # Ortho(-3, 3, -6, 1, -10, 10)
            # X'= X / 3
            # Y'= Z / 3.5 (mapped to -1~1 range)
            
            # Simplified: Let's rotate the view so Z becomes Y.
            # Rotate X by 90 degrees.
            # ModelView:
            # 1 0 0 0
            # 0 0 -1 0
            # 0 1 0 0
            # 0 0 0 1
            
            rot_x_90 = np.array([
                1, 0, 0, 0,
                0, 0, 1, 0, 
                0, -1, 0, 0,
                0, 0, 0, 1
            ], dtype=np.float32)
            
            # Ortho Matrix (Column Major for OpenGL)
            # l=-3, r=3, b=-6, t=1, n=-10, f=10
            l, r, b, t, n, f = -3.0, 3.0, -6.0, 1.0, -10.0, 10.0
            
            ortho_mat = np.array([
                2/(r-l), 0, 0, 0,
                0, 2/(t-b), 0, 0,
                0, 0, -2/(f-n), 0,
                -(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1
            ], dtype=np.float32)
            
            mvp_2d = np.dot(rot_x_90.reshape(4,4), ortho_mat.reshape(4,4)).flatten()
            
            glPointSize(3.0)
            for name in self.lidar_order:
                self.lidar_handlers[name].draw(mvp_2d)
            # Draw threshold-hit points in red on top of normal 2D points.
            glPointSize(4.0)
            for name in self.lidar_order:
                if name in self.lidar_alert_handlers:
                    self.lidar_alert_handlers[name].draw(mvp_2d)
            self.draw_lidar_2d_hover_overlay()

            # Restoring logic
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Disable 3D overlay text rendering.
            # (Keep function for potential future debugging use.)
            # self.print_text(left_w, wnd_h)
            self.draw_control_status(left_w, wnd_h)

            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def _fit_viewport_keep_aspect(self, x, y, w, h, aspect):
        # Fit a source aspect ratio into a destination box using letterboxing.
        box_w = max(1, int(w))
        box_h = max(1, int(h))
        a = float(aspect) if aspect and aspect > 0 else (16.0 / 9.0)

        fit_w = box_w
        fit_h = int(round(float(fit_w) / a))
        if fit_h > box_h:
            fit_h = box_h
            fit_w = int(round(float(fit_h) * a))

        fit_w = max(1, min(box_w, fit_w))
        fit_h = max(1, min(box_h, fit_h))
        off_x = int((box_w - fit_w) * 0.5)
        off_y = int((box_h - fit_h) * 0.5)
        return int(x) + off_x, int(y) + off_y, fit_w, fit_h

    def draw_control_status(self, left_w, wnd_h):
        if not self.control_status_text:
            return
        glViewport(left_w, 0, int(glutGet(GLUT_WINDOW_WIDTH) - left_w), wnd_h)
        glColor3f(0.88, 0.88, 0.88)
        self.print_GL(-0.95, -0.92, self.control_status_text)

    # Update both Mesh and FPC
    def update(self):
        if self.new_chunks:
            if self.ask_clear:
                self.sub_maps = []
                self.ask_clear = False
            
            nb_c = len(self.mesh.chunks)

            if nb_c > len(self.sub_maps): 
                for n in range(len(self.sub_maps),nb_c):
                    self.sub_maps.append(SubMapObj())
            
            # For both Mesh and FPC
            for m in range(len(self.sub_maps)):
                if (m < nb_c) and self.mesh.chunks[m].has_been_updated:
                    if self.draw_mesh:
                        self.sub_maps[m].update_mesh(self.mesh.chunks[m])
                    else:
                        self.sub_maps[m].update_fpc(self.mesh.chunks[m])
                        
            self.new_chunks = False
            self.chunks_pushed = True

    def draw_3d_mesh(self):
        if self.available:
            self.draw_mesh_overlay(include_lidar=True, apply_pan=True)

    def draw_mesh_overlay(self, include_lidar, apply_pan):
        if self.tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
            return
        if self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
            return
        if len(self.sub_maps) == 0:
            return

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Copy pose before inverse to avoid mutating shared pose state.
        tmp = sl.Transform(self.pose)
        tmp.inverse()
        if apply_pan:
            view_np = np.array(tmp.m, dtype=np.float32)
            pan_mat = np.identity(4, dtype=np.float32)
            pan_mat[0, 3] = float(self.pan_offset[0])
            pan_mat[1, 3] = float(self.pan_offset[1])
            pan_mat[2, 3] = float(self.pan_offset[2])
            vpMat = np.dot(np.array(self.projection.m, dtype=np.float32), np.dot(pan_mat, view_np)).flatten()
        else:
            proj = (self.projection * tmp).m
            vpMat = proj.flatten()

        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE, (GLfloat * len(vpMat))(*vpMat))
        glUniform3fv(self.shader_color_loc, 1, (GLfloat * len(self.vertices_color))(*self.vertices_color))

        for m in range(len(self.sub_maps)):
            self.sub_maps[m].draw(self.draw_mesh)

        glUseProgram(0)

        if include_lidar:
            glPointSize(5.0)
            for name in self.lidar_order:
                self.lidar_handlers[name].draw(vpMat)
            glPointSize(7.0)
            for name in self.lidar_order:
                if name in self.lidar_alert_handlers:
                    self.lidar_alert_handlers[name].draw(vpMat)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def print_text(self, left_w, wnd_h):
        if self.available:
            # Shift text to the right viewport roughly
            # OpenGL coordinates for raster pos are -1 to 1... wait, glutBitmapCharacter uses RasterPos
            # glWindowPos might be better or specific glRasterPos with viewport transform
            # But glRasterPos is affected by current ModelViewProjection?
            
            # Simple hack: draw text in the Right Viewport
            glViewport(left_w, 0, wnd_h*2, wnd_h) # Reset viewport to right or full? 
            # Actually print_GL uses -1 to 1 coordinates assuming logic.
            
            # Let's adjust print_GL to take raw window coordinates or stick to simple logic
            # Just put it in the mesh viewport
            glViewport(left_w, 0, int(glutGet(GLUT_WINDOW_WIDTH) - left_w), wnd_h)
            
            # ... existing text logic ...
            if self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                glColor3f(0.15, 0.15, 0.15)
                self.print_GL(-0.95, 0.9, "Spatial Mapping is not enabled.")
            else:
                glColor3f(0.25, 0.25, 0.25)
                self.print_GL(-0.95, 0.9, "Spatial Mapping is running.")
            glColor3f(0.65, 0.65, 0.65)
            self.print_GL(-0.95, 0.97, "Drag(LMB): Pan / Wheel: Zoom / R: Reset")
            self.print_GL(-0.95, 0.04, "Keys: N/M lidar, H/L X, U/O Y, J/K Z, [/] step, 0 off-reset")
            self.print_GL(-0.95, -0.02, "Yaw: ,/. -/+, ;/' step, 9 yaw-reset")

            positional_tracking_state_str = "POSITIONAL TRACKING STATE : "
            spatial_mapping_state_str = "SPATIAL MAPPING STATE : "
            state_str = ""

            # Display spatial mapping state
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                if self.mapping_state == sl.SPATIAL_MAPPING_STATE.OK or self.mapping_state == sl.SPATIAL_MAPPING_STATE.INITIALIZING:
                    glColor3f(0.25, 0.99, 0.25)
                elif self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                    glColor3f(0.55, 0.65, 0.55)
                else:
                    glColor3f(0.95, 0.25, 0.25)
                state_str = spatial_mapping_state_str + str(self.mapping_state)
            else:
                if self.mapping_state != sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                    glColor3f(0.95, 0.25, 0.25)
                    state_str = positional_tracking_state_str + str(self.tracking_state)
                else:
                    glColor3f(0.55, 0.65, 0.55)
                    state_str = spatial_mapping_state_str + str(sl.SPATIAL_MAPPING_STATE.NOT_ENABLED)
            self.print_GL(-0.95, 0.83, state_str)

            # LiDAR legend and status (multi-lidar aware)
            text_y = 0.76
            for name in self.lidar_order:
                color = self.lidar_handlers[name].color
                status = self.lidar_status.get(name, {})
                connected = bool(status.get("connected", False))
                point_count = int(status.get("point_count", 0))
                fps = float(status.get("fps", 0.0))
                up_down = "UP" if connected else "DOWN"
                offset = status.get("offset", {})
                off_x = float(offset.get("x", 0.0))
                off_y = float(offset.get("y", 0.0))
                off_z = float(offset.get("z", 0.0))
                yaw_deg = float(status.get("yaw_deg", 0.0))

                glColor3f(color[0], color[1], color[2])
                self.print_GL(
                    -0.95,
                    text_y,
                    f"LIDAR {name}: {up_down} / pts={point_count} / fps={fps:4.1f} / off=({off_x:+.3f},{off_y:+.3f},{off_z:+.3f}) / yaw={yaw_deg:+.2f}",
                )
                text_y -= 0.06
                if text_y < -0.95:
                    break

class SubMapObj:
    def __init__(self):
        self.current_fc = 0
        self.vboID = None
        self.index = []         # For FPC only
        self.vert = []
        self.tri = []

    def update_mesh(self, _chunk): 
        if(self.vboID is None):
            self.vboID = glGenBuffers(2)

        if len(_chunk.vertices):
            self.vert = _chunk.vertices.flatten()      # transform _chunk.vertices into 1D array 
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize, (GLfloat * len(self.vert))(*self.vert), GL_DYNAMIC_DRAW)
        
        if len(_chunk.triangles):
            self.tri = _chunk.triangles.flatten()      # transform _chunk.triangles into 1D array 
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.tri) * self.tri.itemsize , (GLuint * len(self.tri))(*self.tri), GL_DYNAMIC_DRAW)
            self.current_fc = len(self.tri)

    def update_fpc(self, _chunk): 
        if(self.vboID is None):
            self.vboID = glGenBuffers(2)

        if len(_chunk.vertices):
            self.vert = _chunk.vertices.flatten()      # transform _chunk.vertices into 1D array 
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize, (GLfloat * len(self.vert))(*self.vert), GL_DYNAMIC_DRAW)

            for i in range(len(_chunk.vertices)):
                self.index.append(i)
            
            index_np = np.array(self.index)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(index_np) * index_np.itemsize, (GLuint * len(index_np))(*index_np), GL_DYNAMIC_DRAW)
            self.current_fc = len(index_np)

    def draw(self, _draw_mesh): 
        if self.current_fc:
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            if _draw_mesh == True:
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
            else:
                glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            if len(self.index) > 0:
                glDrawElements(GL_POINTS, self.current_fc, GL_UNSIGNED_INT, None)      
            else:
                glDrawElements(GL_TRIANGLES, self.current_fc, GL_UNSIGNED_INT, None)      

            glDisableVertexAttribArray(0)
