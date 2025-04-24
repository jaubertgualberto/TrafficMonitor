import tkinter as tk
from gui.app import VehicleCounterApp

def main():
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.minsize(1000, 600)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1200x700+{(sw-1200)//2}+{(sh-700)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()