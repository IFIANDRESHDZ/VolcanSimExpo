import numpy as np
import random as rnd

def a(vx, vy, vz, d):
    g = 9.81
    v = np.sqrt(pow(vx, 2) + pow(vy, 2) + pow(vz, 2))
    return -d * v * vx, -d * v * vy, -g - d * v * vz

def rho(h):
    L = 0.0065 #K/m
    g = 9.81 #m/s2
    M = 0.0289644  #kg/mol
    R = 8.3144598 #J/(mol K)
    T0 = 288.15 #K
    P0 = 101325 #Pa
    P = P0 * pow((1-(L*h)/(T0)), (g*M)/(R*L))
    T = T0 - L*h
    Respec = R/M
    rho = P/(Respec*T)
    return rho

def rk4method(dt, vx1, vy1, vz1, x_pos, y_pos, z_pos, t_in, dc): #dt, vx, vy, vz, x, y, z, initial time, function D

    #Stage 1
    ax1, ay1, az1 = a(vx1,vy1,vz1, dc)
    #Stage 2
    vx2,vy2,vz2 = vx1 + dt * ax1/2,vy1 + dt * ay1/2,vz1 + dt * az1/2
    ax2, ay2, az2 = a(vx2+dt*ax1/2,vy2+dt*ay1/2,vz2+dt*az1/2, dc)
    #Stage 3
    vx3,vy3,vz3 = vx2 + dt * ax2/2,  vy2 + dt*ay2/2, vz2 + dt*az2/2
    ax3, ay3, az3 = a(vx3+dt*ax2/2,vy3+dt*ay2/2,vz3+dt*az2/2, dc)
    #Stage 4
    vx4, vy4, vz4= vx3 + dt*ax3, vy3 + dt*ay3, vz3 + dt*az3
    ax4, ay4, az4 = a(vx4+dt*ax2,vy4+dt*ay2,vz4+dt*az2, dc)
    #Calculate velocities
    vx_out = vx1 + (dt/6) * (ax1 + 2* ax2 + 2* ax3 + ax4)
    vy_out = vy1 + (dt/6) * (ay1 + 2* ay2 + 2* ay3 + ay4)
    vz_out = vz1 + (dt/6) * (az1 + 2* az2 + 2* az3 + az4)
    #Calculate positions
    x_out = x_pos + (dt/6) * (vx1 + 2*vx2 + 2*vx3 + vx4)
    y_out = y_pos + (dt/6) * (vy1 + 2*vy2 + 2*vy3 + vy4)
    z_out = z_pos + (dt/6) * (vz1 + 2*vz2 + 2*vz3 + vz4)
    return vx_out, vy_out, vz_out,x_out,  y_out, z_out,t_in+dt

def rk4_complete(theta, azim, height, v, ca, r, dt, valley):
    dens = rnd.randrange(2100, 2600)
    m = dens * 4 / 3 * np.pi * pow(r, 3)
    area = np.pi * pow(r, 2)
    rk_t = rkx = rky = 0
    rkz = height
    rkp = rho(rkz)

    rkvx = v * np.cos(azim) * np.cos(theta)
    rkvy = v * np.cos(azim) * np.sin(theta)
    rkvz = v * np.sin(azim)
    x_rk4, y_rk4, z_rk4, t_rk4 = [rkx], [rky], [rkz], [rk_t]

    while rkz >= valley:
        rkd = area * ca * rkp / (2 * m)
        rkvx, rkvy, rkvz, rkx, rky, rkz, rk_t = rk4method(dt, rkvx, rkvy, rkvz, rkx, rky, rkz, rk_t, rkd)
        rkp = rho(rkz)
        x_rk4.append(rkx)
        y_rk4.append(rky)
        z_rk4.append(rkz)
        t_rk4.append(rk_t)
    rk4_trayectory = {'x': x_rk4, 'y': y_rk4, 'z': z_rk4, 't': t_rk4}

    return rk4_trayectory

def eulermethod(dt, vx, vy, vz, x, y, z, t, d):
    ax, ay, az = a(vx, vy, vz, d)
    vx1 = vx + dt*ax
    vy1 = vy + dt*ay
    vz1 = vz + dt*az
    x1 = x + dt*vx1
    y1 = y + dt*vy1
    z1 = z + dt*vz1
    t1 = t + dt
    return vx1, vy1, vz1, x1, y1, z1, t1

def euler_complete(theta, azim, height, v, ca, r, dt, valley):
    dens = rnd.randrange(2100, 2600)
    m = dens * 4 / 3 * np.pi * pow(r, 3)
    area = np.pi * pow(r, 2)
    et = ex = ey = 0
    ez = height
    ep = rho(ez)

    evx = v * np.cos(azim) * np.cos(theta)
    evy = v * np.cos(azim) * np.sin(theta)
    evz = v * np.sin(azim)
    e_x, e_y, e_z, e_t = [ex], [ey], [ez], [et]

    while ez >= valley:
        ed = area * ca * ep / (2 * m)
        evx, evy, evz, ex, ey, ez, et = eulermethod(dt, evx, evy, evz, ex, ey, ez, et, ed)
        ep = rho(ez)
        e_x.append(ex)
        e_y.append(ey)
        e_z.append(ez)
        e_t.append(et)
    euler_trayectory = {'x': e_x, 'y': e_y, 'z': e_z, 't': e_t}

    return euler_trayectory