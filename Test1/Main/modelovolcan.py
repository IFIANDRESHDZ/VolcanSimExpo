import trimesh
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def plot_obj_on_axes(
        ax,
        obj_path,
        rotate_deg=(0, 0, 0),
        translate=(0, 0, 0),
        scale=1.0,
        base_color=np.array([0.2, 0.7, 0.2]),   # verde por defecto
        alpha=1.0,
        shading_strength=0.6
    ):
    """
    Carga y dibuja un OBJ en un axis 3D, con shading suave por normales de vértice.
    """
    mesh = trimesh.load(obj_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    vertices = mesh.vertices.copy()
    faces = mesh.faces

    # ---- TRANSFORMACIONES ----
    vertices *= scale

    rx, ry, rz = np.radians(rotate_deg)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
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

