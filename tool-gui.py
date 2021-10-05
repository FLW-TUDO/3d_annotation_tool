#!/usr/bin/env python3
import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import pathlib
import json
import cv2
from tf.transformations import quaternion_matrix

isMacOS = (platform.system() == "Darwin")

left_shift_modifier = False
dist = 0.005
deg = 1

global cloud_path

class Scenes:
    def __init__(self, dataset_path):
        self.scenes_path = os.path.join(dataset_path, 'scenes')
        self.objects_path = os.path.join(dataset_path, 'objects')


class AnnotationScene:
    def __init__(self, scene_num, bin_scene):
        self.bin_scene = bin_scene
        self.scene_num = scene_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_name, transform))

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

    class SceneObject:
        def __init__(self, obj_geometry, obj_name, transform):
            self.obj_geometry = obj_geometry
            self.obj_name = obj_name
            self.transform = transform


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.Material(),
            Settings.UNLIT: rendering.Material(),
            Settings.NORMALS: rendering.Material(),
            Settings.DEPTH: rendering.Material()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
                self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def __init__(self, width, height, scenes):
        self.scenes = scenes
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(False)

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # 3D Annotation tool options
        annotation_objects = gui.CollapsableVert("Annotation Objects", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        self._meshes_available = gui.ListView()
        # mesh_available.set_items(["bottle", "can"])
        self._meshes_used = gui.ListView()
        # mesh_used.set_items(["can_0", "can_1", "can_1", "can_1"])
        add_mesh_button = gui.Button("Add Mesh")
        remove_mesh_button = gui.Button("Remove Mesh")
        add_mesh_button.set_on_clicked(self._add_mesh)
        remove_mesh_button.set_on_clicked(self._remove_mesh)
        annotation_objects.add_child(self._meshes_available)
        annotation_objects.add_child(add_mesh_button)
        annotation_objects.add_child(self._meshes_used)
        annotation_objects.add_child(remove_mesh_button)
        self._settings_panel.add_child(annotation_objects)

        scene_control = gui.CollapsableVert("Scene Control", 0.33 * em,
                                            gui.Margins(em, 0, 0, 0))
        pre_button = gui.Button("Previous")
        next_button = gui.Button("Next")
        pre_button.set_on_clicked(self._on_previous_scene)
        next_button.set_on_clicked(self._on_next_scene)
        generate_save_annotation = gui.Button("generate annotation - save/update")
        generate_save_annotation.set_on_clicked(self._on_generate)
        refine_position = gui.Button("Refine position")
        refine_position.set_on_clicked(self._on_refine)
        scene_control.add_child(generate_save_annotation)
        scene_control.add_child(refine_position)
        scene_control.add_child(pre_button)
        scene_control.add_child(next_button)
        self._settings_panel.add_child(scene_control)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            # file_menu.add_item("Open Annotation Folder", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        # w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._on_point_size(1) # set default size to 1

        self._apply_settings()

        self._annotation_scene = None

        # set callbacks for key control
        self._scene.set_on_key(self._transform)

    def _transform(self, event):
        # TODO pressing the keys too fast still causes problems, is that render process that slow
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        global left_shift_modifier
        if event.key == gui.KeyName.LEFT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                left_shift_modifier = False
            return gui.Widget.EventCallbackResult.HANDLED

        # if ctrl is pressed then increase translation and angle values
        global dist, deg
        if event.key == gui.KeyName.LEFT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                dist = 0.05
                deg = 90
            elif event.type == gui.KeyEvent.UP:
                dist = 0.005
                deg = 1
            return gui.Widget.EventCallbackResult.HANDLED

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_empty_active_meshes()
            return gui.Widget.EventCallbackResult.HANDLED

        def move(x, y, z, rx, ry, rz):
            objects = self._annotation_scene.get_objects()
            active_obj = objects[self._meshes_used.selected_index]
            # translation or rotation
            if x!=0 or y!=0 or z!=0:
                h_transform = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
            else: # elif rx!=0 or ry!=0 or rz!=0:
                center = active_obj.obj_geometry.get_center()
                rot_mat_obj_center = active_obj.obj_geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
                T_neg = np.vstack((np.hstack((np.identity(3), -center.reshape(3,1))), [0, 0, 0 ,1]))
                R = np.vstack((np.hstack((rot_mat_obj_center, [[0],[0],[0]])),[0,0,0,1]))
                T_pos = np.vstack((np.hstack((np.identity(3), center.reshape(3,1))), [0, 0, 0 ,1]))
                h_transform = np.matmul(T_pos, np.matmul(R,T_neg))
            active_obj.obj_geometry.transform(h_transform)
            center = active_obj.obj_geometry.get_center()
            self._scene.scene.remove_geometry(active_obj.obj_name)
            self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry, self.settings.material)
            # update values stored of object
            active_obj.transform = np.matmul(h_transform, active_obj.transform)

        if event.type == gui.KeyEvent.DOWN:  # only move objects with down strokes
            # Refine
            if event.key == gui.KeyName.R:
                self._on_refine()
            # Translation
            if not left_shift_modifier:
                if event.key == gui.KeyName.K:
                    print("j pressed: translate in +ve X direction")
                    move(dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.J:
                    print("k pressed: translate in +ve X direction")
                    move(-dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.H:
                    print("h pressed: translate in +ve Y direction")
                    move(0, dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.L:
                    print("l pressed: translate in -ve Y direction")
                    move(0, -dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("i pressed: translate in +ve Z direction")
                    move(0, 0, dist, 0, 0, 0)
                elif event.key == gui.KeyName.COMMA:
                    print(", pressed: translate in -ve Z direction")
                    move(0, 0, -dist, 0, 0, 0)
            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
            else:
                print("Left-Shift is clicked; rotation mode")
                if event.key == gui.KeyName.L:
                    print("j pressed: rotate around +ve X direction")
                    move(0, 0, 0, deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.H:
                    print("k pressed: rotate around -ve X direction")
                    move(0, 0, 0, -deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("h pressed: rotate around +ve Y direction")
                    move(0, 0, 0, 0, deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.COMMA:
                    print("l pressed: rotate around -ve Y direction")
                    move(0, 0, 0, 0, -deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.J:
                    print("i pressed: rotate around +ve Z direction")
                    move(0, 0, 0, 0, 0, deg * np.pi / 180)
                elif event.key == gui.KeyName.K:
                    print(", pressed: rotate around -ve Z direction")
                    move(0, 0, 0, 0, 0, -deg * np.pi / 180)

        return gui.Widget.EventCallbackResult.HANDLED

    def _on_refine(self):
        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_empty_active_meshes()
            return gui.Widget.EventCallbackResult.HANDLED

        target = self._annotation_scene.bin_scene
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        source = active_obj.obj_geometry

        trans_init = np.identity(4)
        threshold = 0.004
        radius = 0.002
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

        active_obj.obj_geometry.transform(reg.transformation)
        #active_obj.obj_geometry.paint_uniform_color([0,1,0])  # Debug
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry, self.settings.material)
        active_obj.transform = np.matmul(reg.transformation, active_obj.transform)

    def _on_generate(self):
        global cloud_path

        # TODO add this to a json file
        obj_mask_value = {
            # our objects
            'choco_box': 20,
            'corn_can': 40,
            'HDMI_cable': 60,
            'krauter_sauce': 80,
            'pantene_shampoo': 100,
            'white_candle': 120,
            'barilla_spaghetti': 255,
            # YCB like
            'cereal_box': 140,
            'scheuermilch': 160,
            'scissors': 180,
            'tomato_can': 200,
            'waschesteife': 220,
            'red_bowl': 240,
        }
        # write 6D annotation for each object
        json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:05}", "6d.json")
        with open(json_6d_path, 'w') as f:
            pose_data = list()
            for obj in self._annotation_scene.get_objects():
                obj_data = {"type": str(obj.obj_name[:-2]),
                            "instance": str(obj.obj_name[-1]),
                            "translation": obj.transform[0:3, 3].tolist(),
                            "orientation": obj.transform[0:3, 0:3].tolist()
                            }
                pose_data.append(obj_data)
            json.dump(pose_data, f)

        # write cloud segmentation annotation - set of points for each object instance
        json_cloud_annotation_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:05}",
                                                  'cloud_annotation.json')

        cloud = o3d.io.read_point_cloud(cloud_path)
        cloud_annotation_data = list()
        with open(json_cloud_annotation_path, 'w') as f:
            for scene_obj in self._annotation_scene.get_objects():
                # find nearest points for each object and save mask
                obj = scene_obj.obj_geometry
                scene = cloud
                pcd_tree = o3d.geometry.KDTreeFlann(scene)
                seg_points = np.zeros(len(scene.points), dtype=bool)
                for point in obj.points:
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, 0.005)
                    np.asarray(scene.colors)[idx[1:], :] = [0, 1, 0]  # Debug
                    seg_points[idx[1:]] = True
                #o3d.visualization.draw_geometries([scene])  # Debug
                seg_idx = np.where(seg_points == True)[0]
                obj_data = {"type": str(scene_obj.obj_name[:-2]),
                            "instance": str(scene_obj.obj_name[-1]),
                            "point_indices": list(map(str, seg_idx))
                            }
                cloud_annotation_data.append(obj_data)
            json.dump(cloud_annotation_data, f)

            # generate segmented image
            depth_k = np.array([[1778.81005859375, 0.0, 967.9315795898438], [0.0, 1778.870361328125, 572.4088134765625], [0.0, 0.0, 1.0]])

            with open(os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:05}",'scene_transformations.json')) as transformations:
                data = json.load(transformations)
                num_of_views = len(data)

                for count in range(num_of_views):
                    tvec = data[str(count)][2]['translation'] # 3rd tranformation is zivid camera to scene link (bin or table)
                    tvec = np.array([tvec['x'], tvec['y'], tvec['z']], np.float64)

                    rvec = data[str(count)][2]['rotation_quaternion']
                    rvec = np.array([rvec['x'], rvec['y'], rvec['z'], rvec['w']], np.float64)
                    matrix = quaternion_matrix(rvec)
                    matrix = matrix[:3,:3]
                    rvec = cv2.Rodrigues(matrix)
                    rvec = rvec[0]

                    seg_mask = np.zeros((1200, 1944))

                    # sort object from closer to farthest so masks occlusion would be generated correctly
                    dist_to_centers = list()
                    cam_pose = data[str(count)][0]['translation'] # 1st tranformation is  iiwa_link to camera
                    cam_pose= np.array([cam_pose['x'], cam_pose['y'], cam_pose['z']], np.float64)
                    scene_center = data[str(count)][1]['translation'] # 2nd tranformation is iiwa_link to scene
                    scene_center = np.array([scene_center['x'], scene_center['y'], scene_center['z']], np.float64)
                    #pcd_tree = o3d.geometry.KDTreeFlann(cloud)
                    for obj in self._annotation_scene.get_objects():
                        object_center = obj.obj_geometry.get_center() + scene_center
                        dist = np.linalg.norm(cam_pose - object_center)
                        dist_to_centers.append(dist)
                    sort_index = np.flip(np.argsort(np.array(dist_to_centers)))

                    obj_count = 0 # TODO make a json file for all objects and add mask pixel value for all of them
                    obj_list = self._annotation_scene.get_objects()
                    for obj_count in range(len(obj_list)):
                        obj = obj_list[sort_index[obj_count]]
                        project_points = np.array(obj.obj_geometry.points)
                        points_indices = cv2.projectPoints(project_points, rvec, tvec, depth_k, None)
                        points_indices = points_indices[0]
                        points_indices = np.around(points_indices)
                        points_indices = points_indices.astype(np.uint16)
                        obj_mask = np.zeros((1200, 1944), dtype=np.uint8)
                        for index in range(project_points.shape[0]):
                            point = (points_indices[index][0][1], points_indices[index][0][0])
                            try:
                                obj_mask[point] = 255
                            except:
                                pass # TODO print warning

                        # fill gaps in mask
                        closing =  obj_mask
                        kernel = np.ones((2, 2), np.uint8)
                        closing = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
                        closing = cv2.dilate(obj_mask, kernel, iterations=1)

                        #cv2.imshow('closing', closing)
                        #cv2.waitKey()
                        #cv2.destroyAllWindows()

                        # Find contours
                        cnts = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                        c = max(cnts, key=cv2.contourArea)

                        pixel_val = obj_mask_value[obj.obj_name[:-2]]
                        #cv2.fillPoly(closing, pts=[c], color=pixel_val)
                        cv2.fillPoly(seg_mask, pts=[c], color=pixel_val)

                        #closing[closing==255] = 0 # remove all pixels not in main contour

                        #seg_mask = np.maximum(closing, seg_mask) # merge current object to over all object

                        obj_count += 1

                    #cv2.imshow("seg mask", seg_mask)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    cv2.imwrite(os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:05}",
                                             str("seg_mask_") + str(count) + ".png"), seg_mask)

    def _on_empty_active_meshes(self):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("No objects are highlighted in scene meshes"))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_empty_active_meshes_ok(self):
        self.window.close_dialog()

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    # def _on_menu_open(self):
    #    dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
    #                         self.window.theme)
    #    dlg.add_filter(
    #        ".ply .stl .fbx .obj .off .gltf .glb",
    #        "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
    #        ".gltf, .glb)")
    #    dlg.add_filter(
    #        ".xyz .xyzn .xyzrgb .ply .pcd .pts",
    #        "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
    #        ".pcd, .pts)")
    #    dlg.add_filter(".ply", "Polygon files (.ply)")
    #    dlg.add_filter(".stl", "Stereolithography files (.stl)")
    #    dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
    #    dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
    #    dlg.add_filter(".off", "Object file format (.off)")
    #    dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
    #    dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
    #    dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
    #    dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
    #    dlg.add_filter(".xyzrgb",
    #                   "ASCII point cloud files with colors (.xyzrgb)")
    #    dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
    #    dlg.add_filter(".pts", "3D Points files (.pts)")
    #    dlg.add_filter("", "All files")

    #    # A file dialog MUST define on_cancel and on_done functions
    #    dlg.set_on_cancel(self._on_file_dialog_cancel)
    #    dlg.set_on_done(self._on_load_dialog_done)
    #    self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _add_mesh(self):
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]

        def which_count():
            types = [i[:-2] for i in meshes]
            equal_values = [i for i in range(len(types)) if types[i] == self._meshes_available.selected_value]
            count = 0
            if len(equal_values):
                indices = np.array(meshes)
                indices = indices[equal_values]
                indices = [int(x[-1]) for x in indices]
                count = max(indices) + 1
                # TODO change to fill the numbers missing in sequence
            return str(count)

        object_geometry = o3d.io.read_point_cloud(
            self.scenes.objects_path + '/' + self._meshes_available.selected_value + '.pcd')
        init_trans = np.identity(4)
        init_trans[2, 3] = 0.2
        object_geometry.transform(init_trans)
        new_mesh_name = str(self._meshes_available.selected_value) + '_' + which_count()
        self._scene.scene.add_geometry(new_mesh_name, object_geometry, self.settings.material)
        self._annotation_scene.add_obj(object_geometry, new_mesh_name, transform=init_trans)
        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = len(meshes) - 1

    def _remove_mesh(self):
        if not self._annotation_scene.get_objects():
            print("There are no object to be deleted.")
            return
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(self._meshes_used.selected_index)  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)

    def scene_load(self, scenes_path, scene_num):
        global cloud_path
        cloud_path = os.path.join(scenes_path, f"{scene_num:05}", 'assembled_cloud.pcd')

        self._scene.scene.clear_geometry()

        geometry = None
        cloud = None

        try:
            cloud = o3d.io.read_point_cloud(cloud_path)
            cloud = cloud.voxel_down_sample(0.001) # downsample assembled cloud to make gui faster

        except Exception:
            pass
        if cloud is not None:
            print("[Info] Successfully read", cloud_path)
            if not cloud.has_normals():
                cloud.estimate_normals()
            cloud.normalize_normals()
            geometry = cloud
        else:
            print("[WARNING] Failed to read points", cloud_path)

        try:
            self._scene.scene.add_geometry("__model__", geometry,
                                           self.settings.material)
            bounds = geometry.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())
            center = np.array([0,0,0])
            eye = center + np.array([-0.5, 0, 1])
            up = np.array([0, 0, 1])
            self._scene.look_at(center, eye, up)

            self._annotation_scene = AnnotationScene(scene_num, geometry)
            self._meshes_used.set_items([])  # clear list from last loaded scene

            # load values if an annotation already exists
            json_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:05}", '6d.json')
            # if os.path.exists(json_path):
            with open(json_path) as json_file:
                data = json.load(json_file)
                obj_list = list()
                active_meshes = list()
                for obj in data:
                    # add object to annotation_scene object
                    obj_geometry = o3d.io.read_point_cloud(os.path.join(self.scenes.objects_path, obj['type'] + '.pcd'))
                    obj_name = obj['type'] + '_' + obj['instance']
                    translation = np.array(np.array(obj['translation']), dtype=np.float64)
                    orientation = np.array(np.array(obj['orientation']), dtype=np.float64)
                    transform = np.concatenate((orientation, translation.reshape(3, 1)), axis=1)
                    transform = np.concatenate((transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform
                    self._annotation_scene.add_obj(obj_geometry, obj_name, transform)
                    # adding object to the scene
                    obj_geometry.translate(translation)
                    center = obj_geometry.get_center()
                    obj_geometry.rotate(orientation, center=center)
                    self._scene.scene.add_geometry(obj_name, obj_geometry, self.settings.material)

                    active_meshes.append(obj_name)
            self._meshes_used.set_items(active_meshes)

        except Exception as e:
            print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def update_obj_list(self):
        objects_list = os.listdir(self.scenes.objects_path)
        objects_list = [x.split('.')[0] for x in objects_list]
        self._meshes_available.set_items(objects_list)

    def _on_next_scene(self):
        # TODO handle overflow
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num + 1)

    def _on_previous_scene(self):
        # TODO handle underflow
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num - 1)

    def save_annotation(self):
        pass


def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    dataset_path = os.path.join(pathlib.Path().absolute(),
                                'dataset')  # TODO make a gui window that asks for dataset path
    scenes = Scenes(dataset_path)

    w = AppWindow(2048, 1536, scenes)

    start_scene_num = 0  # TODO: change it to load last annotated object from json
    if os.path.exists(scenes.scenes_path) and os.path.exists(scenes.objects_path):
        w.scene_load(scenes.scenes_path, start_scene_num)
        w.update_obj_list()
    else:
        w.window.show_message_box("Error",
                                  "Could not scenes or object meshes folders " + scenes.scenes_path + "/" + scenes.objects_path)
        exit()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
