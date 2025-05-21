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
  - Butterworth filters control by n (order) low-pass / high pass 
  - ideal low-pass / high pass
  - 
- 📊 Histogram analysis:
  - View histogram
  - Apply Equalization, Stretching, and Sliding

- transformation :
  - negative
  - log-transformation
  - power-law transformation control by gamma    
- 🌞 Adjust image **brightness**
- 💾 Save processed images and histogram plots
- 🧭 Organized GUI with separate pages:
  - Home | Spatial Filters | Frequency Filters | Histogram | transformation 

- gui visualization

  - ![image](https://github.com/user-attachments/assets/adf72a90-d57f-4c7d-adba-e7b32ca29bb8)
  - ![image](https://github.com/user-attachments/assets/bfaf29fc-b444-4226-9b5e-10b740aa9bcf)
  - ![image](https://github.com/user-attachments/assets/dd1fc99e-f2f4-4da2-be13-a4409a9d27e3)
  - ![image](https://github.com/user-attachments/assets/ed2ae120-2de8-4ca3-90b8-2c5a117345ce)




  

## 📦 Installation

```bash
git clone https://github.com/your-username/Image-Processing-GUI-by_tk.git
cd Image-Processing-GUI-by_tk
pip install -r requirements.txt
python main.py
