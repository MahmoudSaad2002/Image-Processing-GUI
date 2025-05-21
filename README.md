# 🖼️ Image Processing GUI using Tkinter

A powerful and interactive Python GUI for image processing, designed with **Tkinter**. This tool allows users to apply spatial and frequency domain filters, adjust brightness, manipulate histograms, and even define custom kernels — all in one intuitive interface.

---

## 🚀 Features

- 📂 Load and display images
- 🧠 Apply **spatial filters**:
  - Mean, Median, Laplacian, Sobel, etc.
  - Define your own **custom kernel** (up to 7x7)
- 🌐 Apply **frequency domain filters**:
  - Gaussian Low-Pass / High-Pass
  - Butterworth filters
- 📊 Histogram analysis:
  - View histogram
  - Apply Equalization, Stretching, and Sliding
- 🌞 Adjust image **brightness**
- 💾 Save processed images and histogram plots
- 🧭 Organized GUI with separate pages:
  - Home | Spatial Filters | Frequency Filters | Histogram | Restoration

---

## 🖥️ GUI Preview

![GUI Screenshot](images/gui_preview.png)  
*Example of the spatial filter page with a custom kernel applied.*

---

## 📦 Installation

```bash
git clone https://github.com/your-username/Image-Processing-GUI-by_tk.git
cd Image-Processing-GUI-by_tk
pip install -r requirements.txt
python main.py
