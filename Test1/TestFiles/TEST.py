from Objetos import VolcanoManager
from Objetos import Proyectil
from animation import animate_trajectories
import tkinter as tk

path = r'../Data/DataSets/Data1.csv'
manager = VolcanoManager(path)
names = manager.get_names()
print("Volcanes: ")
for i, name in enumerate(names):
    print(f"{i+1}: {name}")

index = int(input("Select an option: "))
method = int(input("Select an method (rk4 - 0) (euler - 1) (both - 3): "))
volcano = manager.get_by_index(index)
amount = int(input("Select a max amount: "))


proyectil = Proyectil(volcano, 1, amount)
timestamp = proyectil.timestamp
print(timestamp)
print(proyectil.amount)
proyectil.start_simulation()
root = tk.Tk()
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

animate_trajectories(frame, proyectil, timestamp, method)

root.mainloop()
print("////////////////////////////////////////////////////")



