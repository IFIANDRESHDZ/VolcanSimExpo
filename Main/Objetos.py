import random as rnd
import os
import pandas as pd
import csv
import datetime as dt
import trimesh
import numpy as np
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

    def plot_obj_on_axes(self,
            ax,
            obj_path,
            rotate_deg=(0, 0, 0),
            translate=(0, 0, 0),
            scale=1.0,
            base_color=np.array([0.2, 0.7, 0.2]),  # verde por defecto
            alpha=1.0,
            shading_strength=0.6
    ):

        mesh = trimesh.load(obj_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)

        vertices = mesh.vertices.copy()
        faces = mesh.faces

        # ---- TRANSFORMACIONES ----
        vertices *= scale
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

        # Matplotlib no soporta color por vértice directo en Poly3DCollection,
        # así que hacemos promedio por cara para simular suavizado
        face_colors = [np.mean(c, axis=0) for c in vertex_colors]
        poly = Poly3DCollection(tris, alpha=alpha, linewidths=0)
        poly.set_facecolor(face_colors)
        poly.set_edgecolor((0, 0, 0, 0))
        ax.add_collection3d(poly)

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


#////////////////////////////////////////////////////////////////////////////////////
#Objeto proyectil dependiente del volcan, crea trayectorias y las puede graficar.
class Proyectil:

    def __init__(self, volcano: Volcano, range_min, range_max) ->None: #Otorga caracteristicas iniciales
        self.volcano = volcano
        self.amount = rnd.randint(range_min, range_max)
        self.state = False
        self.random_shape = True
        self.projectiles = []  # to store projectile data
        now = dt.datetime.now()
        self.timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.samples_data = {'angular': {'Ca': 0.71 }
            ,'Trapezoidal': {'Ca': 0.66}
            ,'Cilindrica': {'Ca': 0.98}
            ,'Irregular': {'Ca': 1.01}
            ,'TipoCaja': {'Ca': 0.74}
            ,'Semi-redondeada': {'Ca': 0.62}
            ,'Cubo frontal': {'Ca': 1.06}
            ,'Cubo vertice': {'Ca': 0.77}
            ,'Cilindrico circular': {'Ca':1.23}
            ,'Esfera': {'Ca': 0.51}}

    def use_random_shape(self):                 #When true, random shapes for trayectories
        shape_name = rnd.choice(list(self.samples_data.keys()))
        data = self.samples_data[shape_name]
        cd = data["Ca"]
        return cd


    def start_simulation(self, dt=0.1) ->None:  #Creates the trayectories for a random number of proyectiles
        from MetodosNum import rk4_complete
        from MetodosNum import euler_complete
        if self.state:
            print('Volcano already erupted')
            return

        self.state = True
        final_points = []
        for i in range(self.amount):
            if self.random_shape:
                ca = self.use_random_shape()
            else:
                shape_name, ca = 'sphere', 0.5
            r = rnd.randrange(27, 100) / 100
            theta = np.radians(rnd.randrange(0, 360))
            azim = np.radians(rnd.randrange(30, 85))
            v = rnd.randrange(100, 280)  # m/s
            trayectory_rk4 = rk4_complete(theta, azim, self.volcano.height_msnm, v, ca,r , dt, self.volcano.height )
            trayectory_euler = euler_complete(theta, azim, self.volcano.height_msnm, v, ca,r , dt, self.volcano.height)
            self.save_rk4(trayectory_rk4, f'{i}_Trayectory_rk4.csv')
            self.save_euler(trayectory_euler, f'{i}_Trayectory_euler.csv')
            final_point = {'x_rk4': trayectory_rk4['x'][-1], 'y_rk4': trayectory_rk4['y'][-1],
                           'x_euler': trayectory_euler['x'][-1], 'y_euler': trayectory_euler['y'][-1]}
            final_points.append(final_point)

        self.save_endpoints(final_points)

    def save_euler(self, data, name):
        folder = os.path.join("../Data", "Trayectory_Data", f"{self.volcano.name}Euler_{self.timestamp}")
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(folder, name), index=False)

    def save_rk4(self, data, name):
        folder = os.path.join("../Data", "Trayectory_Data", f"{self.volcano.name}RK4_{self.timestamp}")
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(folder, name), index=False)

    def save_endpoints(self, data):
        folder = os.path.join("../Data", "Endpoints_Data", f"{self.volcano.name}Endpoints_{self.timestamp}")
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(data)
        path = os.path.join(folder, f"{self.volcano.index}_Endpoints_Data.csv")
        df.to_csv(path, index=False)






