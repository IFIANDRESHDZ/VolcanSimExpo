import random as rnd
import os
import pandas as pd
import numpy as np
from Volcano import Volcano

#////////////////////////////////////////////////////////////////////////////////////
#Objeto proyectil dependiente del volcan, crea trayectorias y las puede graficar.
class ProyectilManager:

    def __init__(self, volcano: Volcano, range_min, range_max):
        self.volcano = volcano
        self.amount = rnd.randint(range_min, range_max)
        self.trayectory = []
        self.timestamp = volcano.timestamp
        self.endpoints = []

    def reset_trayectory(self):
        self.trayectory = []

    def all_trayectories(self, w, w_deg):
        self.reset_trayectory()

        for _ in range(self.amount):
            p = Balistico(self.volcano, w, w_deg)
            traj = p.get_data()                    # dict {'rk4': df, 'euler': df}
            self.trayectory.append({
                'rk4': traj['rk4'],
                'euler': traj['euler']
            })

            self.endpoints.append({
                'x_rk4': traj['rk4']['x'].iloc[-1],
                'y_rk4': traj['rk4']['y'].iloc[-1],
                'x_euler': traj['euler']['x'].iloc[-1],
                'y_euler': traj['euler']['y'].iloc[-1]
            })

        return self.trayectory


class Balistico:
    def __init__(self, volcano:Volcano, wind, wind_deg) -> None:
        self.volcano = volcano
        self.dt = 0.1
        self.trayectory =[]
        self.random_shape = True
        self.samples_data = {'angular': {'Ca': 0.71}
            , 'Trapezoidal': {'Ca': 0.66}
            , 'Cilindrica': {'Ca': 0.98}
            , 'Irregular': {'Ca': 1.01}
            , 'TipoCaja': {'Ca': 0.74}
            , 'Semi-redondeada': {'Ca': 0.62}
            , 'Cubo frontal': {'Ca': 1.06}
            , 'Cubo vertice': {'Ca': 0.77}
            , 'Cilindrico circular': {'Ca': 1.23}
            , 'Esfera': {'Ca': 0.51}}
        self.df_rk4 = None
        self.df_euler = None
        self.roughtness = 0.2
        self.W_peak = wind
        self.wind_deg = wind_deg

    def use_random_shape(self):                 #When true, random shapes for trayectories
        shape_name = rnd.choice(list(self.samples_data.keys()))
        data = self.samples_data[shape_name]
        cd = data["Ca"]
        return cd

    def get_data(self):
        print("---------------------------------------------------------")
        if self.random_shape:
            ca = self.use_random_shape()
        else:
            ca = 0.5
        r = rnd.randrange(27, 100) / 100
        theta = np.radians(rnd.randrange(0, 360))
        azim = np.radians(rnd.randrange(30, 85))
        v = rnd.randrange(100, 280)

        tray_rk4 = self.rk4_complete(theta, azim, v, ca, r)
        tray_euler = self.euler_complete(theta, azim, v, ca, r)

        self.df_rk4 = pd.DataFrame(tray_rk4)
        self.df_euler = pd.DataFrame(tray_euler)

        return {
            "rk4": self.df_rk4,
            "euler": self.df_euler
        }

    def save_euler(self, data, name):
        folder = os.path.join("./Data", "Trayectory_Data", f"{self.volcano.name}Euler_{self.volcano.timestamp}")
        os.makedirs(folder, exist_ok=True)
        data.to_csv(os.path.join(folder, name), index=False)

    def save_rk4(self, data, name):
        folder = os.path.join("./Data", "Trayectory_Data", f"{self.volcano.name}RK4_{self.volcano.timestamp}")
        os.makedirs(folder, exist_ok=True)
        data.to_csv(os.path.join(folder, name), index=False)

    def euler_complete(self, theta, azim, v, ca, r):
        dens = rnd.randrange(2100, 2600)
        m = dens * 4 / 3 * np.pi * pow(r, 3)
        area = np.pi * pow(r, 2)
        et = ex = ey = 0
        ez = self.volcano.height_msnm
        ep = self.rho(ez)

        def eulermethod( vx, vy, vz, x, y, z, t, d):
            wx, wy, wz = self.get_wind(z)
            ax, ay, az = self.a(vx, vy, vz, d, wx, wy, wz)
            vx1 = vx + self.dt * ax
            vy1 = vy + self.dt * ay
            vz1 = vz + self.dt * az
            x1 = x + self.dt * vx1
            y1 = y + self.dt * vy1
            z1 = z + self.dt * vz1
            t1 = t + self.dt
            return vx1, vy1, vz1, x1, y1, z1, t1

        evx = v * np.cos(azim) * np.cos(theta)
        evy = v * np.cos(azim) * np.sin(theta)
        evz = v * np.sin(azim)
        e_x, e_y, e_z, e_t = [ex], [ey], [ez], [et]

        while ez >= self.volcano.height:
            ed = area * ca * ep / (2 * m)
            evx, evy, evz, ex, ey, ez, et = eulermethod( evx, evy, evz, ex, ey, ez, et, ed)
            ep = self.rho(ez)
            e_x.append(ex)
            e_y.append(ey)
            e_z.append(ez)
            e_t.append(et)
        euler_trayectory = {'x': e_x, 'y': e_y, 'z': e_z, 't': e_t}
        return euler_trayectory


    def rk4_complete(self, theta, azim, v, ca, r):
        dens = rnd.randrange(2100, 2600)
        m = dens * 4 / 3 * np.pi * pow(r, 3)
        area = np.pi * pow(r, 2)
        rk_t = rkx = rky = 0
        rkz = self.volcano.height_msnm
        rkp = self.rho(rkz)

        def rk4method(vx1, vy1, vz1, x_pos, y_pos, z_pos, t_in, dc):  # dt, vx, vy, vz, x, y, z, initial time, function D
            wx, wy, wz = self.get_wind(z_pos)
            # Stage 1
            ax1, ay1, az1 = self.a(vx1, vy1, vz1, dc, wx, wy, wz)
            # Stage 2
            vx2, vy2, vz2 = vx1 + self.dt * ax1 / 2, vy1 + self.dt * ay1 / 2, vz1 + self.dt * az1 / 2
            ax2, ay2, az2 = self.a(vx2 + self.dt * ax1 / 2, vy2 + self.dt * ay1 / 2, vz2 + self.dt * az1 / 2, dc, wx, wy, wz)
            # Stage 3
            vx3, vy3, vz3 = vx2 + self.dt * ax2 / 2, vy2 + self.dt * ay2 / 2, vz2 + self.dt * az2 / 2
            ax3, ay3, az3 = self.a(vx3 + self.dt * ax2 / 2, vy3 + self.dt * ay2 / 2, vz3 + self.dt * az2 / 2, dc, wx, wy, wz)
            # Stage 4
            vx4, vy4, vz4 = vx3 + self.dt * ax3, vy3 + self.dt * ay3, vz3 + self.dt * az3
            ax4, ay4, az4 = self.a(vx4 + self.dt * ax2, vy4 + self.dt * ay2, vz4 + self.dt * az2, dc, wx, wy, wz)
            # Calculate velocities
            vx_out = vx1 + (self.dt / 6) * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
            vy_out = vy1 + (self.dt / 6) * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
            vz_out = vz1 + (self.dt / 6) * (az1 + 2 * az2 + 2 * az3 + az4)
            # Calculate positions
            x_out = x_pos + (self.dt / 6) * (vx1 + 2 * vx2 + 2 * vx3 + vx4)
            y_out = y_pos + (self.dt / 6) * (vy1 + 2 * vy2 + 2 * vy3 + vy4)
            z_out = z_pos + (self.dt / 6) * (vz1 + 2 * vz2 + 2 * vz3 + vz4)
            return vx_out, vy_out, vz_out, x_out, y_out, z_out, t_in + self.dt

        rkvx = v * np.cos(azim) * np.cos(theta)
        rkvy = v * np.cos(azim) * np.sin(theta)
        rkvz = v * np.sin(azim)
        x_rk4, y_rk4, z_rk4, t_rk4 = [rkx], [rky], [rkz], [rk_t]

        while rkz >= self.volcano.height:
            rkd = area * ca * rkp / (2 * m)
            rkvx, rkvy, rkvz, rkx, rky, rkz, rk_t = rk4method(rkvx, rkvy, rkvz, rkx, rky, rkz, rk_t, rkd)
            rkp = self.rho(rkz)
            x_rk4.append(rkx)
            y_rk4.append(rky)
            z_rk4.append(rkz)
            t_rk4.append(rk_t)
        rk4_trayectory = {'x': x_rk4, 'y': y_rk4, 'z': z_rk4, 't': t_rk4}

        return rk4_trayectory

    def a(self, vx, vy, vz, d, wx, wy, wz):
        g = 9.81
        v_rel_x = vx - wx
        v_rel_y = vy - wy
        v_rel_z = vz - wz
        v_rel = np.sqrt(pow(v_rel_x, 2) + pow(v_rel_y, 2) + pow(v_rel_z, 2))
        return -d * v_rel * v_rel_x, -d * v_rel * v_rel_y, -g - d * v_rel * v_rel_z

    def get_wind(self, z):

        h = max(z - self.volcano.height, self.roughtness)
        h_ref = max(self.volcano.height_msnm - self.volcano.height, self.roughtness)
        scaling_factor = np.log(h / self.roughtness) / np.log(h_ref / self.roughtness)
        if scaling_factor < 0: scaling_factor = 0
        W_mag = self.W_peak * scaling_factor
        wind_rad = np.radians(self.wind_deg)
        wx = W_mag * np.cos(wind_rad)
        wy = W_mag * np.sin(wind_rad)
        wz = 0
        print(f"Wind: {W_mag} -     Height: {z}")
        return wx, wy, wz

    def rho(self, h):
        L = 0.0065  # K/m
        g = 9.81  # m/s2
        M = 0.0289644  # kg/mol
        R = 8.3144598  # J/(mol K)
        T0 = 288.15  # K
        P0 = 101325  # Pa
        P = P0 * pow((1 - (L * h) / (T0)), (g * M) / (R * L))
        T = T0 - L * h
        Respec = R / M
        rho = P / (Respec * T)
        return rho







