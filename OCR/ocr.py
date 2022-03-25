
from PIL import Image
import pytesseract as pt
import tkinter as tk
import tkinter.filedialog as filedialog


class Application(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("图片文本提取")

        pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.img_path = tk.StringVar()

        self.frame = tk.Frame(self)
        self.frame.pack(padx=10, pady=10)

        self.lbl_file = tk.Label(self.frame, text="图像")
        self.lbl_file.grid(row=0, column=0)
        
        self.txt_file = tk.Entry(self.frame, width=50, textvariable=self.img_path)
        self.txt_file.grid(row=0, column=1, sticky=tk.W)
        
        self.btn_file = tk.Button(self.frame, text="选择", command=self.sel_img_file)
        self.btn_file.grid(row=0, column=1, sticky=tk.E)
        
        self.lbl_txt = tk.Label(self.frame, text="文本")
        self.lbl_txt.grid(row=1, column=0)
        
        self.txt_exract = tk.Text(self.frame)
        self.txt_exract.grid(row=1, column=1)
        
        self.btn_extract = tk.Button(self.frame, text="提取文本", command=self.extract_text)
        self.btn_extract.grid(row=2, column=1, sticky=tk.W+tk.E)
        
        
    def sel_img_file(self):
        self.img_path.set(filedialog.askopenfilename(title="选择图片", initialdir="."))


    def extract_text(self):
        if self.img_path:
            img = Image.open(self.img_path.get())
            text = pt.image_to_string(img, lang="chi_sim")
            self.txt_exract.delete(1.0, tk.END)
            self.txt_exract.insert(tk.END, text)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
