import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from Volcano import VolcanoManager
from matplotlib.animation import FuncAnimation
import numpy as np
from Balisticos import ProyectilManager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


class VolcanoApp:
    def __init__(self):
        self.metodo_actual = 'euler'
        self.ventana = tk.Tk()
        self.ventana.title('Simulacion de Volcanes')
        self.ventana.geometry('1080x720')  # Increased size slightly for better visibility

        # --- CONTROL VARIABLES ---
        self.camera_free_var = tk.BooleanVar(value=False)  # Variable for the checkbox

        # Cargar volcanes desde CSV
        self.volcano_manager = VolcanoManager("./Data/DataSets/Data1.csv")
        self.volcanes = self.volcano_manager.volcanoes

        # --- ATTRIBUTES FOR ARROW ---
        self.wind_quiver = None
        self.wind_text_3d = None
        # --------------------------------

        # Frame izquierdo
        self.frame_izq = tk.Frame(self.ventana, width=400, bg="lightgray")
        self.frame_izq.grid(row=0, column=0, rowspan=2, sticky="nsew")  # Changed rowspan to cover full left height

        # Frame derecho (Container for Graph)
        self.frame_up = tk.Frame(self.ventana, bg="white")
        self.frame_up.grid(row=0, column=1, rowspan=2, sticky="nsew")

        # Configure Grid Weights
        self.ventana.grid_columnconfigure(0, weight=1)  # Left side smaller
        self.ventana.grid_columnconfigure(1, weight=3)  # Right side bigger
        self.ventana.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL CONTROLS ---
        tk.Label(self.frame_izq, text='Método:', font=('Arial', 11, 'bold')).pack(pady=5)
        tk.Button(self.frame_izq, text='Euler', bg='red3', fg='white', width=20, height=2,
                  command=lambda: self.seleccionar_metodo('euler')).pack(pady=2)
        tk.Button(self.frame_izq, text='Runge-Kutta 4', bg='RoyalBlue2', fg='white', width=20, height=2,
                  command=lambda: self.seleccionar_metodo('rk4')).pack(pady=2)
        tk.Button(self.frame_izq, text='Simular', width=25, height=4, bg="#e0e0e0",
                  font=("Arial", 10, "bold"),
                  command=self.simular_trayectorias).pack(pady=20)

        tk.Label(self.frame_izq, text='Volcán:', font=('Arial', 11, 'bold')).pack(pady=5)
        self.volcano_dropdown = ttk.Combobox(self.frame_izq,
                                             values=[v.name for v in self.volcanes],
                                             state="readonly")
        self.volcano_dropdown.pack(fill='y')
        self.volcano_dropdown.bind("<<ComboboxSelected>>", self.on_volcano_select)

        # Info Frame
        self.info_frame = tk.Frame(self.frame_izq, relief=tk.GROOVE, borderwidth=2, bg='white')
        self.info_frame.pack(pady=10, fill='x', padx=10)
        self.info_labels = {}
        fields = ['Nombre', 'Ubicacion', 'Altura desde el valle', 'Latitud',
                  'Humedad', 'Temp cima', 'Temp interior', 'Altura (msnm)']
        for field in fields:
            lbl = tk.Label(self.info_frame, text=f"{field}: ", bg='white')
            lbl.pack(anchor='w')
            self.info_labels[field] = lbl

        # Risk Map Image (Left Bottom)
        self.label_imagen = tk.Label(self.frame_izq, bg="lightgray")
        self.label_imagen.pack(fill='both', expand=True, pady=10)

        # --- RIGHT PANEL (GRAPH SIDE) ---

        # 1. TOP SECTION: WIND SLIDERS
        self.top_controls = tk.Frame(self.frame_up, bg="white", bd=1, relief=tk.RAISED)
        self.top_controls.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.wind = tk.DoubleVar(value=0.0)
        self.w_deg = tk.DoubleVar(value=90.0)

        # Grid layout for sliders
        tk.Label(self.top_controls, text="Velocidad Viento (m/s)", bg="white").grid(row=0, column=0, padx=10)
        scale_vel = tk.Scale(self.top_controls, from_=0, to=30, orient='horizontal', variable=self.wind,
                             command=self.update_params, length=200, bg="white")
        scale_vel.grid(row=1, column=0, padx=10)

        tk.Label(self.top_controls, text="Dirección Viento (°)", bg="white").grid(row=0, column=1, padx=10)
        scale_angle = tk.Scale(self.top_controls, from_=0, to=360, orient='horizontal', variable=self.w_deg,
                               command=self.update_params, length=200, bg="white")
        scale_angle.grid(row=1, column=1, padx=10)

        # Labels for values
        self.lbl_vel_val = tk.Label(self.top_controls, text="0.0 m/s", bg="white", font=("Arial", 10, "bold"))
        self.lbl_vel_val.grid(row=2, column=0)
        self.lbl_ang_val = tk.Label(self.top_controls, text="90.0 °", bg="white", font=("Arial", 10, "bold"))
        self.lbl_ang_val.grid(row=2, column=1)

        self.top_controls.columnconfigure(0, weight=1)
        self.top_controls.columnconfigure(1, weight=1)


        self.bottom_controls = tk.Frame(self.frame_up, bg="#e0e0e0", height=50)
        self.bottom_controls.pack(side=tk.BOTTOM, fill=tk.X)

        # --- WIDGETS UNDER GRAPH ---
        # Free Camera Checkbox
        self.cb_camera = tk.Checkbutton(self.bottom_controls, text="Camara Libre (Detener Rotacion)",
                                        variable=self.camera_free_var, bg="#e0e0e0", font=("Arial", 10))
        self.cb_camera.pack(side=tk.LEFT, padx=20, pady=10)

        # Reset View Button
        self.btn_reset = tk.Button(self.bottom_controls, text="Reset Camera View",
                                   command=lambda: self.reset_axis(None))
        self.btn_reset.pack(side=tk.LEFT, padx=10)

        # 3. MIDDLE SECTION: GRAPH
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d", zorder=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_up)
        # This fills whatever space is left between Top Controls and Bottom Controls
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.reset_axis()
        self.ventana.mainloop()

    def update_params(self, event=None):
        self.lbl_vel_val.config(text=f"{self.wind.get():.1f} m/s")
        self.lbl_ang_val.config(text=f"{self.w_deg.get():.1f} °")

    def draw_wind_arrow(self):
        if self.wind_quiver:
            try:
                self.wind_quiver.remove()
            except:
                pass
        if self.wind_text_3d:
            try:
                self.wind_text_3d.remove()
            except:
                pass

        speed = self.wind.get()
        angle_deg = self.w_deg.get()
        angle_rad = np.radians(angle_deg)

        visual_length = 80 + 150 * max(speed, 0)
        u = visual_length * np.cos(angle_rad)
        v = visual_length * np.sin(angle_rad)
        w = 0

        z_lims = self.ax.get_zlim()
        z_pos = z_lims[1] * 0.8

        self.wind_quiver = self.ax.quiver(0, 0, z_pos, u, v, w,
                                          color='black', length=1.0, normalize=False, linewidth=1.5,
                                          arrow_length_ratio=0.2)
        self.wind_text_3d = self.ax.text(u, v, z_pos, f" Viento: {speed:.1f} m/s", color='darkgray', fontweight='bold')

    def seleccionar_metodo(self, metodo):
        self.metodo_actual = metodo
        self.actualizar_mapa_seleccionado()

    def on_volcano_select(self, event):
        self.actualizar_info_seleccionado()
        self.actualizar_mapa_seleccionado()

    def actualizar_info_seleccionado(self):
        selected_name = self.volcano_dropdown.get()
        volcano = next((v for v in self.volcanes if v.name == selected_name), None)
        if volcano:
            self.update_volcano_info(volcano)

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
        wind_speed = self.wind.get()
        wind_direction = self.w_deg.get()
        self.ProyectileManager = ProyectilManager(volcano, 10, 15)
        self.ProyectileManager.all_trayectories(wind_speed, wind_direction)
        self.animar_trayectorias(self.ProyectileManager.trayectory, volcano)

    def mostrar_mapa(self, volcano):
        if not volcano:
            return
        index = self.volcanes.index(volcano)
        if self.metodo_actual == 'euler':
            archivo = f"./Data/risk_maps/euler_maps/risk_map {index + 1}.png"
        else:
            archivo = f"./Data/risk_maps/rk4_maps/risk_map_rk4 {index + 1}.png"
        try:
            img = Image.open(archivo)
            img = img.resize((400, 350))  # Adjusted size to fit left panel
            photo = ImageTk.PhotoImage(img)
            self.label_imagen.config(image=photo)
            self.label_imagen.image = photo
        except FileNotFoundError:
            self.label_imagen.config(text="Imagen no encontrada", image='')

    def update_volcano_info(self, volcano):
        self.info_labels['Nombre'].config(text=f"Nombre: {volcano.name}")
        self.info_labels['Ubicacion'].config(text=f"Ubicacion: {volcano.location}")
        self.info_labels['Altura desde el valle'].config(
            text=f"Altura desde el valle: {volcano.height_msnm - volcano.height} m")
        self.info_labels['Latitud'].config(text=f"Latitud: {volcano.latitude}")
        self.info_labels['Humedad'].config(text=f"Humedad: {volcano.humidity}")
        self.info_labels['Temp cima'].config(text=f"Temp cima: {volcano.temp_cima} °C")
        self.info_labels['Temp interior'].config(text=f"Temp interior: {volcano.temp_interior} °C")
        self.info_labels['Altura (msnm)'].config(text=f"Altura (msnm): {volcano.height_msnm} msnm")

    def reset_axis(self, volcano=None):
        self.ax.clear()
        self.ax.set_title("Simulación 3D de trayectorias")
        self.ax.view_init(elev=10, azim=45)
        self.ax.set_xlabel("X(m)")
        self.ax.set_ylabel("Y(m)")
        self.ax.set_zlabel("Z(m)")
        if volcano:
            height_diff = volcano.height_msnm - volcano.height
            scaling = 4000 if height_diff > 500 else 3500
            scaling_z = 4000 if height_diff > 300 else 1200
            self.ax.set_zlim(volcano.height, volcano.height_msnm + scaling_z)
        else:
            scaling = 4000
            scaling_z = 4000
            self.ax.set_zlim(-scaling_z, scaling_z)
        self.ax.set_xlim(-scaling, scaling)
        self.ax.set_ylim(-scaling, scaling)

    def animar_trayectorias(self, trayectories, volcano):
        # 1. Reset the view
        self.reset_axis(volcano)
        self.draw_wind_arrow()

        elev_start = 10
        azim_start = self.camera_free_var.get()
        ROTATION_SPEED = 0.5
        if self.metodo_actual == 'rk4':
            all_trajs = [t['rk4'] for t in trayectories]
            color = 'blue'
        else:
            all_trajs = [t["euler"] for t in trayectories]
            color = 'red'

        # Plot volcano model
        volcano.plot_obj_on_axes(
            self.ax,
            r"./Data/Model/volcano4.obj",
            rotate_deg=(90, 0, 180),
            translate=(0, 0, volcano.height),
            base_color=np.array([0.5, 0.5, 0.5]),
            alpha=0.67,
            shading_strength=1
        )

        all_times = np.unique(np.concatenate([traj['t'].values for traj in all_trajs]))

        last_time = time.time()
        fps_text = self.ax.text2D(0.02, 0.95, "", color='red', transform=self.ax.transAxes)
        sim_time_text = self.ax.text2D(0.7, 0.95, "", color='red', transform=self.ax.transAxes)

        points = [self.ax.plot([], [], [], 'o', color=color, markersize=6, alpha=1.0)[0] for _ in all_trajs]
        trails = [self.ax.plot([], [], [], '-', color="#FF4500", lw=2, alpha=0.2)[0] for _ in all_trajs]

        total_frames = len(all_times)

        def update(frame):
            nonlocal last_time
            now = time.time()
            dt = now - last_time

            last_time = now
            fps = 1.0 / dt if dt > 0 else 0
            fps_text.set_text(f"FPS: {fps:.1f}")
            current_time = all_times[frame]
            sim_time_text.set_text(f"Tiempo: {current_time:.2f} s")

            if self.camera_free_var.get():
                target_azim = self.ax.azim
            else:
                target_azim = (azim_start + frame * ROTATION_SPEED) % 360

            self.ax.view_init(elev=elev_start, azim=target_azim)


            for idx, traj in enumerate(all_trajs):
                mask = (traj['t'] <= current_time) & (traj['t'] >= max(current_time - 6.7, 0))

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

        self.ani = FuncAnimation(self.fig, update, frames=total_frames, interval=20, blit=False)
        self.canvas.draw()


if __name__ == "__main__":
    VolcanoApp()