import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def animate_trajectories(parent_frame, launch, timestamp, method, trail_length=2):
    base = os.path.join("../Data", "Trayectory_Data")
    euler_trajs = []
    rk4_trajs = []

    # Load all trajectories
    for i in range(launch.amount):
        file_euler = os.path.join(base, f"{launch.volcano.name}Euler_{timestamp}", f"{i}_Trayectory_euler.csv")
        file_rk4   = os.path.join(base, f"{launch.volcano.name}RK4_{timestamp}", f"{i}_Trayectory_rk4.csv")
        euler_trajs.append(pd.read_csv(file_euler))
        rk4_trajs.append(pd.read_csv(file_rk4))

    if method == 1:
        all_trajs = euler_trajs
    elif method == 0:
        all_trajs = rk4_trajs
    else:
        all_trajs = euler_trajs + rk4_trajs

    # Tkinter-embedded figure
    fig = plt.Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Embed ONE canvas
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Plot volcano model
    plot_obj_on_axes(
        ax,
        r"Data/Model/volcano4.obj",
        rotate_deg=(90, 0, 0),
        # El volcán inicia en el valle, pero debe elevarse hasta el cráter
        translate=(0, 0, launch.volcano.height),
        # Escala ajustada al modelo
        scale=2000,
        base_color=np.array([0.5, 0.5, 0.5]),
        alpha=0.67,
        shading_strength=1
    )

    # Compute limits
    all_x = np.concatenate([traj['x'].values for traj in all_trajs])
    all_y = np.concatenate([traj['y'].values for traj in all_trajs])
    all_z = np.concatenate([traj['z'].values for traj in all_trajs])
    max_val = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    max_valz = max(np.abs(all_z))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(launch.volcano.height, max_valz)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    last_time = time.time()
    fps_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # Lines for animation
    points = [ax.plot([], [], [], 'o', color='red')[0] for _ in all_trajs]
    trails = [ax.plot([], [], [], '-', color='gray', lw=3)[0] for _ in all_trajs]

    # Times
    all_times = np.unique(np.concatenate([traj['t'].values for traj in all_trajs]))
    total_frames = len(all_times)

    # Update function
    def update(frame):
        nonlocal last_time
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        fps_text.set_text(f"FPS: {fps:.1f}")

        current_time = all_times[frame]

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

        return points + trails + [fps_text]

    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

    # Prevent garbage collection
    parent_frame.ani = ani
    return ani
