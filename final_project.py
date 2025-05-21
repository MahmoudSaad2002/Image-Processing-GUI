# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:16:16 2025

@author: Mahmoud_Saad
"""


import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== GLOBAL VARIABLES =====================
original_image = None
edited_image = None
image_path = None

# ===================== HELPER FUNCTIONS =====================
def load_image():
    global original_image, image_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        original_image = Image.open(file_path).convert('RGB')
        show_image_on_canvas(original_image, original_canvas)

def resize_image(img, max_size=(300, 300)):
    img.thumbnail(max_size, Image.LANCZOS)
    return img

def show_image_on_canvas(img, canvas):
    img = resize_image(img.copy())
    tk_img = ImageTk.PhotoImage(img)
    canvas.image = tk_img
    canvas.create_image(0, 0, anchor='nw', image=tk_img)

def back_to_main(current_win):
    current_win.destroy()
    main_window()

# ===================== SPATIAL FILTERS WINDOW =====================
def spatial_filters_window():
    window = tk.Toplevel()
    window.title("Spatial Filters")
    window.geometry("800x600")

    tk.Button(window, text="Back to Main", command=lambda: back_to_main(window)).pack(pady=10)

    frame = tk.Frame(window)
    frame.pack()

    global original_canvas, edited_canvas
    original_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    original_canvas.grid(row=0, column=0, padx=10)
    edited_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    edited_canvas.grid(row=0, column=1, padx=10)

    filters = {
        "BLUR": ImageFilter.BLUR,
        "CONTOUR": ImageFilter.CONTOUR,
        "DETAIL": ImageFilter.DETAIL,
        "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
        "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
        "EMBOSS": ImageFilter.EMBOSS,
        "FIND_EDGES": ImageFilter.FIND_EDGES,
        "SHARPEN": ImageFilter.SHARPEN,
        "SMOOTH": ImageFilter.SMOOTH,
        "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
        "MaxFilter": ImageFilter.MaxFilter(3),
        "MinFilter": ImageFilter.MinFilter(3),
        "MedianFilter": ImageFilter.MedianFilter(3),
        "ModeFilter": ImageFilter.ModeFilter(3)
    }
    
    
    def apply_selected_filter():
        global edited_image
        selected = filter_var.get()
        if original_image:
            if selected in filters:
                edited_image = original_image.filter(filters[selected])
            elif selected == "Bilateral Filter":
                img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                filtered = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
                edited_image = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            elif selected == "Laplacian Filter":
                img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
                laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
                laplacian = cv2.convertScaleAbs(laplacian)
                laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
                edited_image = Image.fromarray(laplacian_rgb)
    
            show_image_on_canvas(edited_image, edited_canvas)

    filter_var = tk.StringVar(window)
    filter_var.set("Select Filter")
    filter_menu = ttk.Combobox(window, textvariable=filter_var, 
    values=list(filters.keys()) + ["Bilateral Filter", "Laplacian Filter"])

    #filter_menu = ttk.Combobox(window, textvariable=filter_var, values=list(filters.keys()))
    filter_menu.pack(pady=10)

    tk.Button(window, text="Apply Filter", command=apply_selected_filter).pack()

    def apply_custom_kernel():
        try:
            size = int(kernel_size_var.get())
            if size > 7 or size < 1:
                raise ValueError("Kernel size must be between 1 and 7")

            kernel_values = custom_kernel_text.get("1.0", tk.END).split()
            kernel_values = list(map(float, kernel_values))
            if len(kernel_values) != size * size:
                raise ValueError("Incorrect number of kernel values")

            kernel = np.array(kernel_values).reshape((size, size))
            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            filtered = cv2.filter2D(img_cv, -1, kernel)
            filtered_img = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            show_image_on_canvas(filtered_img, edited_canvas)

        except Exception as e:
            messagebox.showerror("Kernel Error", str(e))

    tk.Label(window, text="Custom Kernel Size (1-7):").pack()
    kernel_size_var = tk.StringVar()
    tk.Entry(window, textvariable=kernel_size_var).pack()

    tk.Label(window, text="Enter Kernel values (row-wise):").pack()
    custom_kernel_text = tk.Text(window, height=5, width=30)
    custom_kernel_text.pack()
    tk.Button(window, text="Apply Custom Kernel", command=apply_custom_kernel).pack(pady=5)

    def adjust_brightness(val):
        if original_image:
            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            brightness = int(val)
            img_bright = cv2.convertScaleAbs(img_cv, alpha=1, beta=brightness)
            bright_img = Image.fromarray(cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB))
            show_image_on_canvas(bright_img, edited_canvas)

    tk.Label(window, text="Adjust Brightness:").pack()
    brightness_slider = tk.Scale(window, from_=-100, to=100, orient='horizontal', command=adjust_brightness)
    brightness_slider.pack()

    def save_edited_image():
        if edited_image:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                edited_image.save(path)

    tk.Button(window, text="Save Image", command=save_edited_image).pack(pady=10)

    if original_image:
        show_image_on_canvas(original_image, original_canvas)


# ===================== HISTOGRAM WINDOW =====================
def histogram_window():
    window = tk.Toplevel()
    window.title("Histogram Processing")
    window.geometry("1000x700")

    tk.Button(window, text="Back to Main", command=lambda: back_to_main(window)).pack(pady=5)

    img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
    processed = img_cv.copy()

    # ======= Layout Frames =======
    top_frame = tk.Frame(window)
    top_frame.pack()

    canvas_fig_frame = tk.Frame(top_frame)
    canvas_fig_frame.grid(row=0, column=0, padx=10)

    image_canvas_frame = tk.Frame(top_frame)
    image_canvas_frame.grid(row=0, column=1, padx=10)

    # ======= Histogram Plot =======
    fig, ax = plt.subplots(figsize=(5, 3))
    canvas_fig = FigureCanvasTkAgg(fig, master=canvas_fig_frame)
    canvas_fig.get_tk_widget().pack()

    # ======= Image Canvas for Processed Image =======
    image_canvas = tk.Canvas(image_canvas_frame, width=300, height=300, bg='gray')
    image_canvas.pack()

    def plot_histogram(image):
        ax.clear()
        ax.hist(image.ravel(), bins=256, range=(0, 256), color='black')
        ax.set_title("Histogram")
        canvas_fig.draw()

    def show_processed_image(img_arr):
        img_pil = Image.fromarray(img_arr)
        img_resized = resize_image(img_pil.copy())
        tk_img = ImageTk.PhotoImage(img_resized)
        image_canvas.image = tk_img
        image_canvas.create_image(0, 0, anchor='nw', image=tk_img)

    def apply_equalization():
        nonlocal processed
        processed = cv2.equalizeHist(img_cv)
        plot_histogram(processed)
        show_processed_image(processed)

    def apply_stretching():
        nonlocal processed
        a, b = np.min(img_cv), np.max(img_cv)
        processed = ((img_cv - a) * (255 / (b - a))).astype(np.uint8)
        plot_histogram(processed)
        show_processed_image(processed)

    def apply_sliding(val):
        nonlocal processed
        shift = int(val)
        processed = np.clip(img_cv + shift, 0, 255).astype(np.uint8)
        plot_histogram(processed)
        show_processed_image(processed)

    tk.Button(window, text="Equalization", command=apply_equalization).pack(pady=5)
    tk.Button(window, text="Stretching", command=apply_stretching).pack(pady=5)
    tk.Label(window, text="Sliding (Shift value):").pack()
    tk.Scale(window, from_=-100, to=100, orient='horizontal', command=apply_sliding).pack()

    def save_histogram():
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            fig.savefig(path)

    def save_processed_image():
        if processed is not None:
            out_img = Image.fromarray(processed)
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                out_img.save(path)

    tk.Button(window, text="Save Histogram", command=save_histogram).pack(pady=5)
    tk.Button(window, text="Save Processed Image", command=save_processed_image).pack(pady=5)

    plot_histogram(img_cv)
    show_processed_image(img_cv)

# ===================== FREQUENCY FILTERS WINDOW =====================

def apply_frequency_filter(img, filter_type, d0=30, n=1):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2 , cols // 2

    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D = np.sqrt((U - ccol)**2 + (V - crow)**2)

    if filter_type == "Gaussian Low Pass":
        H = np.exp(-(D**2) / (2 * (d0**2)))
    elif filter_type == "Gaussian High Pass":
        H = 1 - np.exp(-(D**2) / (2 * (d0**2)))
    elif filter_type == "Butterworth Low Pass":
        H = 1 / (1 + (D / d0)**(2*n))
    elif filter_type == "Butterworth High Pass":
        H = 1 / (1 + (d0 / D)**(2*n))
        H[D == 0] = 0
    elif filter_type == "Ideal Low Pass":
        H = np.zeros_like(D)
        H[D <= d0] = 1
    elif filter_type == "Ideal High Pass":
        H = np.ones_like(D)
        H[D <= d0] = 0
    else:
        H = np.ones_like(D)

    filtered_dft = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return Image.fromarray(img_back)

def frequency_filters_window():
    window = tk.Toplevel()
    window.title("Frequency Domain Filters")
    window.geometry("900x700")

    tk.Button(window, text="Back to Main", command=lambda: back_to_main(window)).pack(pady=10)

    frame = tk.Frame(window)
    frame.pack()

    original_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    original_canvas.grid(row=0, column=0, padx=10)
    result_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    result_canvas.grid(row=0, column=1, padx=10)

    if original_image:
        show_image_on_canvas(original_image, original_canvas)

    tk.Label(window, text="Select Frequency Filter").pack()
    filter_var = tk.StringVar()
    
    filter_choices = [
    "Gaussian Low Pass",
    "Gaussian High Pass",
    "Butterworth Low Pass",
    "Butterworth High Pass",
    "Ideal Low Pass",
    "Ideal High Pass"
    ]

    filter_menu = ttk.Combobox(window, textvariable=filter_var, values=filter_choices)
    filter_menu.set("Select Filter")
    filter_menu.pack()

    tk.Label(window, text="Cutoff Frequency (D0)").pack()
    d0_var = tk.StringVar(value="30")
    tk.Entry(window, textvariable=d0_var).pack()

    tk.Label(window, text="Order (only for Butterworth)").pack()
    order_var = tk.StringVar(value="2")
    tk.Entry(window, textvariable=order_var).pack()

    def apply_filter():
        global edited_image
        if original_image:
            ftype = filter_var.get()
            d0 = float(d0_var.get())
            order = int(order_var.get())
            edited_image = apply_frequency_filter(original_image, ftype, d0, order)
            show_image_on_canvas(edited_image, result_canvas)

    tk.Button(window, text="Apply Filter", command=apply_filter).pack(pady=10)

    def save_freq_image():
        if edited_image:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                edited_image.save(path)

    tk.Button(window, text="Save Image", command=save_freq_image).pack(pady=5)

# ===================== MAIN WINDOW =====================
def main_window():
    root = tk.Tk()
    root.title("ENG/Mahmoud Saad - Image Editor")
    root.geometry("500x600")

    tk.Label(root, text="Image Editor", font=("Arial", 16)).pack(pady=20)

    tk.Button(root, text="Load Image", command=load_image).pack(pady=10)

    tk.Button(root, text="Spatial Filters", command=lambda: [root.destroy(), spatial_filters_window()]).pack(pady=5)
    tk.Button(root, text="Frequency Domain Filters", command=lambda: [root.destroy(), frequency_filters_window()]).pack(pady=5)
    tk.Button(root, text="Histogram Processing", command=lambda: [root.destroy(), histogram_window()]).pack(pady=5)
    tk.Button(root, text="Image Transformation", command=lambda: [root.destroy(), image_transformation_window()]).pack(pady=5)
    #tk.Button(root, text="Image Restoration", command=lambda: messagebox.showinfo("Coming Soon", "This feature will be added soon.")).pack(pady=5)

    tk.Label(root, text="Brightness Adjustment").pack(pady=10)
    tk.Scale(root, from_=-100, to=100, orient='horizontal', command=lambda val: adjust_main_brightness(val)).pack()

    global original_canvas
    original_canvas = tk.Canvas(root, width=300, height=300, bg='gray')
    original_canvas.pack(pady=10)

    root.mainloop()

def adjust_main_brightness(val):
    global original_image, edited_image
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        brightness = int(val)
        img_bright = cv2.convertScaleAbs(img_cv, alpha=1, beta=brightness)
        edited_image = Image.fromarray(cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB))
        show_image_on_canvas(edited_image, original_canvas)

#================================================
def image_transformation_window():
    window = tk.Toplevel()
    window.title("Image Transformation")
    window.geometry("900x600")

    tk.Button(window, text="Back to Main", command=lambda: back_to_main(window)).pack(pady=10)

    frame = tk.Frame(window)
    frame.pack()

    original_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    original_canvas.grid(row=0, column=0, padx=10)
    transformed_canvas = tk.Canvas(frame, width=300, height=300, bg='gray')
    transformed_canvas.grid(row=0, column=1, padx=10)

    if original_image:
        show_image_on_canvas(original_image, original_canvas)

    def apply_transformation():
        global edited_image
        choice = transform_var.get()
        img_np = np.array(original_image.convert("L"))  # Convert to grayscale

        if choice == "Negative":
            transformed = 255 - img_np
        elif choice == "Log":
            try:
                c = float(log_c_var.get())
                transformed = c * np.log1p(img_np)
                transformed = np.uint8(np.clip(transformed * (255.0 / transformed.max()), 0, 255))
            except:
                messagebox.showerror("Input Error", "Please enter a valid value for c.")
                return
        elif choice == "Power Law":
            try:
                gamma = float(power_gamma_var.get())
                normalized = img_np / 255.0
                transformed = np.power(normalized, gamma)
                transformed = np.uint8(transformed * 255)
            except:
                messagebox.showerror("Input Error", "Please enter a valid gamma value.")
                return
        else:
            messagebox.showwarning("Selection Required", "Please select a transformation type.")
            return

        edited_image = Image.fromarray(transformed)
        show_image_on_canvas(edited_image, transformed_canvas)

    transform_var = tk.StringVar()
    transform_menu = ttk.Combobox(window, textvariable=transform_var, values=["Negative", "Log", "Power Law"])
    transform_menu.set("Select Transformation")
    transform_menu.pack(pady=10)

    # Inputs for log and power
    log_c_var = tk.StringVar()
    power_gamma_var = tk.StringVar()

    log_frame = tk.Frame(window)
    tk.Label(log_frame, text="Log Transform c:").pack(side="left")
    tk.Entry(log_frame, textvariable=log_c_var, width=5).pack(side="left")
    log_frame.pack(pady=5)

    power_frame = tk.Frame(window)
    tk.Label(power_frame, text="Power Law γ:").pack(side="left")
    tk.Entry(power_frame, textvariable=power_gamma_var, width=5).pack(side="left")
    power_frame.pack(pady=5)

    tk.Button(window, text="Apply", command=apply_transformation).pack(pady=10)

    def save_transformed_image():
        if edited_image:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                edited_image.save(path)

    tk.Button(window, text="Save Image", command=save_transformed_image).pack(pady=10)

# ===================== START APP =====================
main_window()




