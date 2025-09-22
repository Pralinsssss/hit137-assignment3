import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("AI Model GUI")
root.geometry("900x700")

menubar = tk.Menu(root)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=file_menu)
root.config(menu=menubar)

root.mainloop()