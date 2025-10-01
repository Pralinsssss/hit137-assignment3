import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import functools
from transformers import pipeline
from PIL import Image
import torch

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model_name = getattr(args[0], '_model_name', 'unknown') if args else 'unknown'
        print(f"[LOG] Calling {func.__name__} on {model_name}")
        return func(*args, **kwargs)
    return wrapper

class TextClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self._model_name = model_name
        self._pipeline = None
        self._loaded = False

    def load(self):
        try:
            print(f"Loading {self._model_name}...")
            self._pipeline = pipeline("text-classification", model=self._model_name)
            self._loaded = True
            print(f"Successfully loaded {self._model_name}")
            return True
        except Exception as e:
            print(f"Error loading {self._model_name}: {str(e)}")
            raise e

    @measure_time
    @log_call
    def predict(self, text):
        if not self._loaded:
            self.load()
        result = self._pipeline(text)[0]
        return f"Sentiment: {result['label']}\nConfidence: {result['score']:.4f}"

class ImageClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self._model_name = model_name
        self._pipeline = None
        self._loaded = False

    def load(self):
        try:
            print(f"Loading {self._model_name}...")
            self._pipeline = pipeline("image-classification", model=self._model_name)
            self._loaded = True
            print(f"Successfully loaded {self._model_name}")
            return True
        except Exception as e:
            print(f"Error loading {self._model_name}: {str(e)}")
            raise e

    @measure_time
    @log_call
    def predict(self, image_path):
        if not self._loaded:
            self.load()
        result = self._pipeline(image_path)[0]
        return f"Prediction: {result['label']}\nConfidence: {result['score']:.4f}"

class AIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Model GUI - Hugging Face Integration")
        self.root.geometry("900x700")

        self.model_classes = {
            "Text Classification": TextClassifier,
            "Image Classification": ImageClassifier
        }
        
        self.current_model = None
        self.model_instances = {}
        self.is_loading = False

        self.create_menu()
        self.create_frames()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        models_menu = tk.Menu(menubar, tearoff=0)
        models_menu.add_command(label="Load Selected Model", command=self.load_selected_model)
        models_menu.add_command(label="Load All Models", command=self.load_all_models)
        menubar.add_cascade(label="Models", menu=models_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_frames(self):
        self.model_frame = ttk.LabelFrame(self.root, text="Model Selection & Control", padding=10)
        self.model_frame.pack(padx=10, pady=5, fill="x")

        self.input_frame = ttk.LabelFrame(self.root, text="User Input Section", padding=10)
        self.input_frame.pack(padx=10, pady=5, fill="x")

        self.output_frame = ttk.LabelFrame(self.root, text="Model Output Section", padding=10)
        self.output_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.info_frame = ttk.LabelFrame(self.root, text="Information & Concepts", padding=10)
        self.info_frame.pack(padx=10, pady=5, fill="both", expand=True)

    def create_widgets(self):
        self.create_model_selection_widgets()
        self.create_input_widgets()
        self.create_output_widgets()
        self.create_info_widgets()  

    def create_model_selection_widgets(self):
        ttk.Label(self.model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var, 
                                      state="readonly", width=25)
        self.model_combo['values'] = tuple(self.model_classes.keys())
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.model_combo.set("Select Model")
        
        self.load_btn = ttk.Button(self.model_frame, text="Load Selected Model", 
                                 command=self.load_selected_model)
        self.load_btn.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(self.model_frame, text="Load All Models", 
                  command=self.load_all_models).grid(row=0, column=3, padx=5, pady=5)
        
        self.status_label = ttk.Label(self.model_frame, text="Select and load a model to start")
        self.status_label.grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky='w')

    def create_input_widgets(self):
        input_type_frame = ttk.Frame(self.input_frame)
        input_type_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(input_type_frame, text="Input Type:").pack(side="left", padx=5)
        
        self.input_type = tk.StringVar(value="Text")
        ttk.Radiobutton(input_type_frame, text="Text", variable=self.input_type, 
                       value="Text", command=self.on_input_type_changed).pack(side="left", padx=10)
        ttk.Radiobutton(input_type_frame, text="Image", variable=self.input_type, 
                       value="Image", command=self.on_input_type_changed).pack(side="left", padx=10)
        
        self.browse_button = ttk.Button(input_type_frame, text="Browse Image", 
                                      command=self.browse_file)
        self.browse_button.pack(side="left", padx=10)
        self.browse_button.pack_forget()  # Hide initially

        self.input_text = tk.Text(self.input_frame, height=6, wrap='word')
        self.input_text.pack(padx=5, pady=5, fill="x")
        
        self.input_text.insert("1.0", "This is a fantastic movie! I loved every moment of it.")

    def create_output_widgets(self):
        self.output_text = tk.Text(self.output_frame, height=12, wrap='word')
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        button_frame = ttk.Frame(self.output_frame)
        button_frame.pack(pady=10)
        
        self.run_selected_btn = ttk.Button(button_frame, text="Run Selected Model", 
                                         command=self.run_selected_model)
        self.run_selected_btn.pack(side="left", padx=10)
        
        self.run_all_btn = ttk.Button(button_frame, text="Run All Models", 
                                    command=self.run_all_models)
        self.run_all_btn.pack(side="left", padx=10)
        
        ttk.Button(button_frame, text="Clear Output", 
                  command=self.clear_output).pack(side="left", padx=10)

    def create_info_widgets(self):
        """Create split info section with Model Info and OOP Concepts side by side"""
        info_container = ttk.Frame(self.info_frame)
        info_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        model_info_frame = ttk.LabelFrame(info_container, text="Selected Model Information")
        model_info_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.model_info_text = tk.Text(model_info_frame, height=8, wrap='word')
        self.model_info_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.model_info_text.insert("1.0", "No model loaded yet. Please load a model to see information.")
        
        oop_frame = ttk.LabelFrame(info_container, text="OOP Concepts Explanation")
        oop_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        self.oop_text = tk.Text(oop_frame, height=8, wrap='word')
        self.oop_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.oop_text.insert("1.0", "OOP concepts explanation will appear here...\n\nThis section is ready for your team member to add OOP explanations.")

    def on_input_type_changed(self):
        """Show/hide browse button based on input type"""
        if self.input_type.get() == "Image":
            self.browse_button.pack(side="left", padx=10)
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", "Click 'Browse Image' to select an image file")
        else:
            self.browse_button.pack_forget()
            if "Browse" in self.input_text.get("1.0", "1.50"):
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert("1.0", "This is a fantastic movie! I loved every moment of it.")

    def load_selected_model(self):
        selected = self.model_var.get()
        if not selected or selected == "Select Model":
            messagebox.showwarning("No Model Selected", "Please select a model from the dropdown first!")
            return
        
        if self.is_loading:
            messagebox.showinfo("Please Wait", "A model is already loading. Please wait...")
            return

        self.is_loading = True
        self.status_label.config(text=f"Loading {selected}...")
        self.load_btn.config(state="disabled")
        
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", f"Loading {selected}...\n\nPlease wait, this may take a moment...")

        def load_model_thread():
            try:
                model_class = self.model_classes[selected]
                model_instance = model_class()
                model_instance.load()
                
                self.model_instances[selected] = model_instance
                self.current_model = model_instance
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.on_model_loaded(selected, model_instance))
                
            except Exception as e:
                self.root.after(0, lambda: self.on_model_load_error(selected, str(e)))
        
        threading.Thread(target=load_model_thread, daemon=True).start()

    def on_model_loaded(self, model_name, model_instance):
        """Callback when model is successfully loaded"""
        self.is_loading = False
        self.load_btn.config(state="normal")
        self.status_label.config(text=f"{model_name} loaded successfully!")
        
        info = f"MODEL LOADED SUCCESSFULLY!\n\n"
        info += f"Name: {model_name}\n"
        info += f"Model ID: {model_instance._model_name}\n"
        info += f"Status: Ready for predictions!"
        
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", info)
        
        messagebox.showinfo("Model Loaded", f"{model_name} is ready to use!")

    def on_model_load_error(self, model_name, error):
        """Callback when model loading fails"""
        self.is_loading = False
        self.load_btn.config(state="normal")
        self.status_label.config(text=f"Failed to load {model_name}")
        
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", f"ERROR LOADING MODEL\n\nModel: {model_name}\nError: {error}")
        
        messagebox.showerror("Load Error", f"Failed to load {model_name}:\n\n{error}")

    def load_all_models(self):
        """Load all available models"""
        self.status_label.config(text="Loading all models...")
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", "Loading all models...\nThis may take a few minutes...\n\n")
        
        def load_all_thread():
            results = []
            for name, model_class in self.model_classes.items():
                try:
                    if name not in self.model_instances:
                        self.root.after(0, lambda n=name: self.model_info_text.insert(tk.END, f"Loading {n}...\n"))
                        model_instance = model_class()
                        model_instance.load()
                        self.model_instances[name] = model_instance
                        results.append(f"{name}: Loaded successfully")
                    else:
                        results.append(f"{name}: Already loaded")
                except Exception as e:
                    results.append(f"{name}: Failed - {str(e)}")
            
            self.root.after(0, lambda: self.on_all_models_loaded(results))
        
        threading.Thread(target=load_all_thread, daemon=True).start()

    def on_all_models_loaded(self, results):
        """Callback when all models are loaded"""
        self.status_label.config(text="All models loaded!")
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", "ALL MODELS STATUS:\n\n" + "\n".join(results))
        messagebox.showinfo("All Models Loaded", "All models have been loaded successfully!")

    def run_selected_model(self):
        if not self.current_model:
            messagebox.showwarning("No Model Loaded", "Please load a model first!")
            return
            
        input_data = self.input_text.get("1.0", tk.END).strip()
        if not input_data:
            messagebox.showwarning("No Input", "Please provide input text or image path!")
            return
        
        if self.input_type.get() == "Image":
            if not os.path.exists(input_data):
                messagebox.showerror("Invalid Image", "Please select a valid image file using the Browse button!")
                return
        
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "Processing... Please wait...\n")
        self.run_selected_btn.config(state="disabled")
        
        def run_model_thread():
            try:
                result = self.current_model.predict(input_data)
                model_name = self.model_var.get()
                self.root.after(0, lambda: self.on_model_result(model_name, result))
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        threading.Thread(target=run_model_thread, daemon=True).start()

    def on_model_result(self, model_name, result):
        """Callback when model prediction completes"""
        self.run_selected_btn.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", f"{model_name} RESULTS:\n{'='*40}\n{result}\n\n")
        self.output_text.insert(tk.END, f"\nPrediction completed successfully!")

    def on_model_error(self, error):
        """Callback when model prediction fails"""
        self.run_selected_btn.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", f"ERROR:\n{'='*40}\n{error}\n\n")
        messagebox.showerror("Model Error", f"Prediction failed:\n{error}")

    def run_all_models(self):
        input_data = self.input_text.get("1.0", tk.END).strip()
        if not input_data:
            messagebox.showwarning("No Input", "Please provide input text or image path!")
            return
            
        if self.input_type.get() == "Image":
            if not os.path.exists(input_data):
                messagebox.showerror("Invalid Image", "Please select a valid image file using the Browse button!")
                return
        
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "Running all models...\nPlease wait...\n\n")
        self.run_all_btn.config(state="disabled")
        
        def run_all_models_thread():
            results = []
            for name, model_class in self.model_classes.items():
                try:
                    if name not in self.model_instances:
                        model_instance = model_class()
                        model_instance.load()
                        self.model_instances[name] = model_instance
                    
                    model_instance = self.model_instances[name]
                    result = model_instance.predict(input_data)
                    results.append((name, result, None))
                except Exception as e:
                    results.append((name, None, str(e)))
            
            self.root.after(0, lambda: self.on_all_models_result(results))
        
        threading.Thread(target=run_all_models_thread, daemon=True).start()

    def on_all_models_result(self, results):
        """Callback when all models complete"""
        self.run_all_btn.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        
        for name, result, error in results:
            self.output_text.insert(tk.END, f"{name}:\n")
            self.output_text.insert(tk.END, "─" * 50 + "\n")
            if error:
                self.output_text.insert(tk.END, f"Error: {error}\n")
            else:
                self.output_text.insert(tk.END, f"{result}\n")
            self.output_text.insert(tk.END, "\n")

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, file_path)

    def show_about(self):
        about_text = """AI Model GUI - HIT137 Assignment 3

Integrated with Hugging Face Transformers:

• Text Classification: DistilBERT sentiment analysis
• Image Classification: Vision Transformer (ViT)

Features:
✅ Modern GUI with Tkinter
✅ Multi-threading for responsive UI
✅ Error handling and user feedback
✅ Model caching for performance
✅ Real-time progress updates

Built with Python and Transformers"""
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIGUI(root)
    
    # Startup message
    print("AI Model GUI Starting...")
    print("Models will download on first use (requires internet)")
    print("Please be patient during initial model loading")
    
    root.mainloop()