import random as rnd
import os
import pandas as pd
import csv
import datetime as dt
import trimesh
import numpy as np
from matplotlib.scale import scale_factory
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
#Clase para el volcan
class Volcano:
    def __init__(self,index,name, location, height, latitude, humidity, temp_cima, temp_interior, height_msnm):
        self.index = index
        self.name = name
        self.location = location
        self.height = float(height)
        self.latitude = float(latitude)
        self.humidity = humidity
        self.temp_cima = float(temp_cima)
        self.temp_interior = float(temp_interior)
        self.height_msnm = float(height_msnm)
        now = dt.datetime.now()
        self.timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    def calculate_scale(self):
        None

    def plot_obj_on_axes(self,
                         ax,
                         obj_path,
                         rotate_deg=(0, 0, 0),
                         translate=(0, 0, 0),
                         scale=None,  # <--- IMPORTANTE: Debe ser None por defecto
                         base_color=np.array([0.2, 0.7, 0.2]),
                         alpha=1.0,
                         shading_strength=0.6
                         ):

        # 1. Calcular el factor de escala SI el usuario no dio uno específico
        if scale is None:
            volcano_model_height = 1429.086  # Altura de referencia de tu archivo .obj

            # Altura que DEBE tener el volcán (Cima - Base)
            target_relief = self.height_msnm - self.height

            # Factor de compresión Z
            # En tu caso: 660 / 1429.086 = 0.4618...
            z_factor = target_relief / volcano_model_height

            # Aplicamos escala normal en X/Y, y comprimida en Z
            scale = (2000, 2000*z_factor+100, 2000)

        # 2. Cargar Mesh
        mesh = trimesh.load(obj_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)

        vertices = mesh.vertices.copy()
        faces = mesh.faces

        # 3. Aplicar Transformación
        # Si scale era None, ahora es (1.0, 1.0, 0.46...), así que funcionará
        vertices = vertices * scale

        # ... (resto del código de rotación y traslación igual) ...
        rx, ry, rz = np.radians(rotate_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        vertices = vertices @ R.T

        vertices += np.array(translate)

        # ---- NORMALS SUAVES POR VÉRTICE ----
        if hasattr(mesh, "vertex_normals"):
            vertex_normals = mesh.vertex_normals.copy()
            vertex_normals = vertex_normals @ R.T
        else:
            # fallback: promedio de normales de caras
            vertex_normals = np.zeros(vertices.shape)
            for i, f in enumerate(faces):
                normal = np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]])
                normal /= np.linalg.norm(normal)
                vertex_normals[f] += normal
            vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]

        # ---- SHADING POR VÉRTICES ----
        light_dir = np.array([0, 0, 1.0])
        light_dir /= np.linalg.norm(light_dir)

        tris = vertices[faces]
        vertex_colors = []
        for tri in faces:
            intensity = np.clip(vertex_normals[tri] @ light_dir, 0, 1)
            shade = (1 - shading_strength) + shading_strength * intensity
            color = np.clip(base_color * shade[:, None], 0, 1)
            vertex_colors.append(color)

        # así que hacemos promedio por cara para simular suavizado
        face_colors = [np.mean(c, axis=0) for c in vertex_colors]
        poly = Poly3DCollection(tris, alpha=alpha, linewidths=0)

        poly.set_facecolor(face_colors)
        poly.set_edgecolor((0, 0, 0, 0))
        poly.set_sort_zpos(True)
        ax.add_collection3d(poly)

        # Validación visual en consola
        z_max_actual = np.max(vertices[:, 2])
        print(f"Volcán: {self.name}")
        print(f" - Altura Base (Translate Z): {translate[2]}")
        print(f" - Altura Cima Real (Target): {self.height_msnm}")
        print(f" - Altura Máxima en Gráfica: {z_max_actual}")
        return vertices


class VolcanoManager:
    def __init__(self, csv_path):
        self.volcanoes = self.load_csv(csv_path)
    def load_csv(self, path):
        volcano_list=[]
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = Volcano(
                    index = row['index'],
                    name = row['name'],
                    location = row['location'],
                    height = row['height'],
                    latitude = row['latitude'],
                    humidity = row['humidity'],
                    temp_cima = row['temp_cima'],
                    temp_interior = row['temp_interior'],
                    height_msnm = row['height_msnm']

                )
                volcano_list.append(v)
        return volcano_list
    def get_names(self):
        return [v.name for v in self.volcanoes]
    def get_by_index(self, index):
        return self.volcanoes[index-1]

