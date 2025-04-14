import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

#Captura o objeto por imagem de template

def template_match(template, target, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    """
    Match por template
    """
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(target_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        template_height, template_width = template_gray.shape[:2]

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        return [(top_left, bottom_right, 0)]
    else:
        return []


def upload_image(label):
    filename = filedialog.askopenfilename()
    if filename:
        img = cv2.imread(filename)
        label.config(text=f"Image: {filename}")
        return img


def display_matched_result():
    template_img = template_image_label.cget("text").split(": ")[1]
    target_img = target_image_label.cget("text").split(": ")[1]
    method = method_var.get()
    threshold = float(threshold_entry.get())

    if template_img and target_img:
        template = cv2.imread(template_img)
        target = cv2.imread(target_img)

        method_dict = {
            "TM_CCOEFF": cv2.TM_CCOEFF,
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR": cv2.TM_CCORR,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF": cv2.TM_SQDIFF,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
        }
        selected_method = method_dict[method]

        matches = template_match(template, target, selected_method, threshold)

        for match in matches:
            top_left, bottom_right, _ = match
            cv2.rectangle(target, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow('Matched Regions', target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Tinker
root = tk.Tk()
root.title("Template Matching")

# Imagem de template
template_image_label = tk.Label(root, text="Imagem template:")
template_image_label.grid(row=0, column=0, padx=10, pady=10)
template_image_button = tk.Button(root, text="Insira uma imagem de template",
                                  command=lambda: upload_image(template_image_label))
template_image_button.grid(row=0, column=1, padx=10, pady=10)

# Imagem alvo
target_image_label = tk.Label(root, text="Imagem alvo:")
target_image_label.grid(row=1, column=0, padx=10, pady=10)
target_image_button = tk.Button(root, text="Insira uma imagem-alvo", command=lambda: upload_image(target_image_label))
target_image_button.grid(row=1, column=1, padx=10, pady=10)

# Escolha do método
method_label = tk.Label(root, text="Escolha método:")
method_label.grid(row=2, column=0, padx=10, pady=10)
method_options = ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
method_var = tk.StringVar(root)
method_var.set(method_options[1])  # Default method
method_dropdown = tk.OptionMenu(root, method_var, *method_options)
method_dropdown.grid(row=2, column=1, padx=10, pady=10)

# Threshold
threshold_label = tk.Label(root, text="Threshold:")
threshold_label.grid(row=3, column=0, padx=10, pady=10)
threshold_entry = tk.Entry(root)
threshold_entry.insert(tk.END, "0.8")  # Default threshold
threshold_entry.grid(row=3, column=1, padx=10, pady=10)

# Botão identificar
match_button = tk.Button(root, text="identificar", command=display_matched_result)
match_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()