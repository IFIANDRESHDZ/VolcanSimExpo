import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from Objetos import VolcanoManager
from modelovolcan import plot_obj_on_axes
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import os
from Objetos import Proyectil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class VolcanoApp:
    def __init__(self):
        self.metodo_actual = 'euler'
        self.ventana = tk.Tk()
        self.ventana.title('Simulacion de Volcanes')
        self.ventana.geometry('1920x1080')
        # Cargar volcanes desde CSV
        self.volcano_manager = VolcanoManager("../Data/DataSets/Data1.csv")
        self.volcanes = self.volcano_manager.volcanoes  # Lista de objetos Volcano

        # Frame derecho
        self.frame_der = tk.Frame(self.ventana)
        self.frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        #Frame izquierdo
        self.frame_izq = tk.Frame(self.ventana, width=300, bg="lightgray")
        self.frame_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        #Crear grafico vacio
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d", computed_zorder = False)
        self.ax.view_init(elev=30, azim=45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_der)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        #boton Trayectorias
        tk.Button(self.frame_izq, text='Simular trayectorias', width=25, command=self.simular_trayectorias).pack(pady=20)

        tk.Label(self.frame_izq, text='Método:', font=('Arial', 11, 'bold')).pack(pady=5)
        tk.Button(self.frame_izq, text='Euler', bg='red', fg='white', width=20, height=2,
                  command=lambda: self.seleccionar_metodo('euler')).pack(pady=2)
        tk.Button(self.frame_izq, text='Runge-Kutta 4', bg='blue', fg='white', width=20, height=2,
                  command=lambda: self.seleccionar_metodo('rk4')).pack(pady=2)


        tk.Label(self.frame_izq, text='Volcán:', font=('Arial', 11, 'bold')).pack(pady=5)
        self.volcano_dropdown = ttk.Combobox(self.frame_izq,
                                             values=[v.name for v in self.volcanes],
                                             state="readonly")
        self.volcano_dropdown.pack(fill=tk.X)
        self.volcano_dropdown.bind("<<ComboboxSelected>>", self.on_volcano_select)

        # Frame derecho para mapa

        self.label_imagen = tk.Label(self.frame_izq)
        self.label_imagen.pack(fill=tk.BOTH, expand=True)
        # Frame de información
        self.info_frame = tk.Frame(self.frame_der)
        self.info_frame.pack(pady=10, fill=tk.X)
        self.info_labels = {}
        fields = ['Nombre', 'Ubicacion', 'Altura desde el valle', 'Latitud',
                  'Humedad', 'Temp cima', 'Temp interior', 'Altura (msnm)']
        for field in fields:
            lbl = tk.Label(self.info_frame, text=f"{field}: ")
            lbl.pack(anchor='w')
            self.info_labels[field] = lbl

        self.ventana.mainloop()

    def seleccionar_metodo(self, metodo):
        self.metodo_actual = metodo
        # Actualizar mapa si ya hay un volcán seleccionado
        self.actualizar_mapa_seleccionado()

    def on_volcano_select(self, event):
        self.actualizar_info_seleccionado()
        self.actualizar_mapa_seleccionado()

    def actualizar_info_seleccionado(self):
        selected_name = self.volcano_dropdown.get()
        volcano = next((v for v in self.volcanes if v.name == selected_name), None)
        if volcano:
            self.update_volcano_info(volcano)
        self.launch = Proyectil(volcano, 1, 15)

    def actualizar_mapa_seleccionado(self):
        selected_name = self.volcano_dropdown.get()
        volcano = next((v for v in self.volcanes if v.name == selected_name), None)
        if volcano:
            self.mostrar_mapa(volcano)

    def simular_trayectorias(self):
        selected_name = self.volcano_dropdown.get()
        volcano = next((v for v in self.volcanes if v.name == selected_name), None)
        if not volcano:
            return
        self.launch = Proyectil(volcano, 5, 15)  # adjust parameters
        self.launch.start_simulation()
        self.animar_trayectorias(self.launch)

    def mostrar_mapa(self, volcano):
        if not volcano:
            return
        index = self.volcanes.index(volcano)
        if self.metodo_actual == 'euler':
            archivo = f"../Data/risk_maps/euler_maps/risk_map {index+1}.png"
        else:
            archivo = f"../Data/risk_maps/rk4_maps/risk_map_rk4 {index+1}.png"
        try:
            img = Image.open(archivo)
            img = img.resize((600, 500))
            photo = ImageTk.PhotoImage(img)
            self.label_imagen.config(image=photo)
            self.label_imagen.image = photo
        except FileNotFoundError:
            self.label_imagen.config(text="Imagen no encontrada", image='')

    def update_volcano_info(self, volcano):
        self.info_labels['Nombre'].config(text=f"Nombre: {volcano.name}")
        self.info_labels['Ubicacion'].config(text=f"Ubicacion: {volcano.location}")
        self.info_labels['Altura desde el valle'].config(text=f"Altura desde el valle: {volcano.height_msnm-volcano.height} m")
        self.info_labels['Latitud'].config(text=f"Latitud: {volcano.latitude}")
        self.info_labels['Humedad'].config(text=f"Humedad: {volcano.humidity}")
        self.info_labels['Temp cima'].config(text=f"Temp cima: {volcano.temp_cima} °C")
        self.info_labels['Temp interior'].config(text=f"Temp interior: {volcano.temp_interior} °C")
        self.info_labels['Altura (msnm)'].config(text=f"Altura (msnm): {volcano.height_msnm} msnm")

    def animar_trayectorias(self, launch, trail_length = 6.7):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_title("Simulación 3D de trayectorias")
        base = os.path.join("../Data", "Trayectory_Data")
        euler_trajs = []
        rk4_trajs = []
        for i in range(launch.amount):
            file_euler = os.path.join(base, f"{launch.volcano.name}Euler_{launch.timestamp}", f"{i}_Trayectory_euler.csv")
            file_rk4 = os.path.join(base, f"{launch.volcano.name}RK4_{launch.timestamp}", f"{i}_Trayectory_rk4.csv")
            euler_trajs.append(pd.read_csv(file_euler))
            rk4_trajs.append(pd.read_csv(file_rk4))
        color = 'black'
        if self.metodo_actual == 'euler':
            all_trajs = euler_trajs
            color = 'red'
        elif self.metodo_actual == 'rk4':
            all_trajs = rk4_trajs
            color = 'blue'

        # Plot volcano model
        launch.volcano.plot_obj_on_axes(
            self.ax,
            r"../Data/Model/volcano4.obj",
            rotate_deg=(90, 0, 180),
            # El volcán se eleva al crater
            translate=(0, 0, launch.volcano.height),
            scale=2000,
            base_color=np.array([0.5, 0.5, 0.5]),
            alpha=1,
            shading_strength=1
        )
        all_x = np.concatenate([traj['x'].values for traj in all_trajs])
        all_y = np.concatenate([traj['y'].values for traj in all_trajs])
        all_z = np.concatenate([traj['z'].values for traj in all_trajs])
        all_times = np.unique(np.concatenate([traj['t'].values for traj in all_trajs]))


        self.ax.set_xlim(-3000, 3000)
        self.ax.set_ylim(-3000, 3000)
        self.ax.set_zlim(launch.volcano.height, launch.volcano.height + 2000)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        last_time = time.time()
        fps_text = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes)
        sim_time_text = self.ax.text2D(0.7, 0.95, "", transform=self.ax.transAxes)  # top-right corner
        dt_sim = np.mean(np.diff(all_times))  # in seconds

        # Lines for animation
        points = [self.ax.plot([], [], [], 'o', color=color)[0] for _ in all_trajs]
        trails = [self.ax.plot([], [], [], '-', color='gray', lw=3, alpha = 0.6)[0] for _ in all_trajs]
        total_frames = len(all_times)
        intervals = np.diff(all_times, prepend=0) * 1000  # in milliseconds

        def update(frame):
            nonlocal last_time
            now = time.time()
            dt = now - last_time

            last_time = now
            fps = 1.0 / dt if dt > 0 else 0
            fps_text.set_text(f"FPS: {fps:.1f}")

            current_time = all_times[frame]
            sim_time_text.set_text(f"Tiempo: {current_time:.2f} s")  # display simulation time

            for idx, traj in enumerate(all_trajs):
                mask = (traj['t'] <= current_time) & (traj['t'] >= max(current_time - trail_length, 0))

                if mask.any():
                    trails[idx].set_data(traj['x'][mask], traj['y'][mask])
                    trails[idx].set_3d_properties(traj['z'][mask])

                    last_idx = mask[mask].index[-1]
                    points[idx].set_data([traj['x'][last_idx]], [traj['y'][last_idx]])
                    points[idx].set_3d_properties([traj['z'][last_idx]])
                else:
                    trails[idx].set_data([], [])
                    trails[idx].set_3d_properties([])
                    points[idx].set_data([], [])
                    points[idx].set_3d_properties([])

            return points + trails + [fps_text, sim_time_text]

        self.ani = FuncAnimation(self.fig, update, frames=total_frames, interval=50, blit=False)
        self.canvas.draw()


if __name__ == "__main__":
    VolcanoApp()
