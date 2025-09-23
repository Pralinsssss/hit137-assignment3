import tkinter as tk
from tkinter import ttk

class AIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Model GUI")
        self.root.geometry("900x700")
        
        self.create_menu()
        self.create_frames()
        self.create_widgets()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        models_menu = tk.Menu(menubar, tearoff=0)
        models_menu.add_command(label="Load Model")
        menubar.add_cascade(label="Models", menu=models_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About")
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
    def create_frames(self):
        self.model_frame = ttk.LabelFrame(self.root, text="Model Selection")
        self.model_frame.pack(padx=10, pady=5, fill="x")
        
        self.input_frame = ttk.LabelFrame(self.root, text="User Input Section")
        self.input_frame.pack(padx=10, pady=5, fill="x")
        
        self.output_frame = ttk.LabelFrame(self.root, text="Model Output Section")
        self.output_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.info_frame = ttk.LabelFrame(self.root, text="Model Information & Explanation")
        self.info_frame.pack(padx=10, pady=5, fill="both", expand=True)

    def create_widgets(self):
        ttk.Label(self.model_frame, text="Model Selection:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var, state="readonly", width=20)
        model_combo['values'] = ("Text Classification", "Image Classification")
        model_combo.grid(row=0, column=1, padx=5, pady=5)
        model_combo.set("Select Model")
        ttk.Button(self.model_frame, text="Load Model").grid(row=0, column=2, padx=5, pady=5)
        
        input_type = tk.StringVar(value="Text")
        ttk.Radiobutton(self.input_frame, text="Text", variable=input_type, value="Text").pack(side="left", padx=5)
        ttk.Radiobutton(self.input_frame, text="Image", variable=input_type, value="Image").pack(side="left", padx=5)
        ttk.Button(self.input_frame, text="Browse").pack(side="left", padx=5)
        
        self.input_text = tk.Text(self.input_frame, height=5)
        self.input_text.pack(padx=5, pady=5, fill="x")
        
        self.output_text = tk.Text(self.output_frame, height=10)
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        button_frame = tk.Frame(self.output_frame)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="Run Model 1").pack(side="left", padx=5)
        ttk.Button(button_frame, text="Run Model 2").pack(side="left", padx=5)
        
        info_container = tk.Frame(self.info_frame)
        info_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        model_info_frame = ttk.LabelFrame(info_container, text="Selected Model Info:")
        model_info_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.model_info_text = tk.Text(model_info_frame, height=8)
        self.model_info_text.pack(padx=5, pady=5, fill="both")
        
        oop_frame = ttk.LabelFrame(info_container, text="OOP Concepts Explanation:")
        oop_frame.pack(side="right", fill="both", expand=True, padx=5)
        self.oop_text = tk.Text(oop_frame, height=8)
        self.oop_text.pack(padx=5, pady=5, fill="both")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIGUI(root)
    root.mainloop()