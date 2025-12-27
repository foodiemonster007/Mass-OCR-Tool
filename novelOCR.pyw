import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from google import genai
from google.genai import types
import re
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import sys
import threading
import subprocess
import importlib.util
import urllib.request
import tempfile
import platform

# --- Dependency Check ---

def check_and_install_packages():
    def is_package_installed(package_name):
        """Check if a package is installed by trying to import it"""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except:
            return False
    
    # Collect all error messages
    errors = []
    console_output = []
    
    def log_error(msg):
        errors.append(msg)
        console_output.append(msg)
    
    def log_console(msg):
        console_output.append(msg)
    
    # First, ensure pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        log_console("pip not found. Attempting to install pip...")
        
        # Method 1: Try ensurepip first
        log_console("Attempting Method 1: Using ensurepip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade", "--user"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_console("✓ Successfully installed pip using ensurepip.")
        except subprocess.CalledProcessError:
            log_console("Method 1 failed. Attempting Method 2: Using get-pip.py...")
            
            # Method 2: Download and run get-pip.py if ensurepip fails
            try:
                # Download and run get-pip.py
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                    temp_path = temp_file.name
                    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", temp_path)
                
                try:
                    subprocess.check_call([sys.executable, temp_path, "--user"], 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log_console("✓ Successfully installed pip using get-pip.py.")
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_path)
                    
            except Exception as e:
                error_msg = f"ERROR: Both methods failed to install pip. Please install it manually. Error: {e}"
                log_error(error_msg)
                # Save error and show console
                with open("error.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(console_output))
                # Keep console open
                if platform.system() == "Windows":
                    input("Press Enter to exit...")
                return False
    
    # Upgrade pip to latest version (silent)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass  # Silent failure for pip upgrade
    
    # Check and install required packages
    required_packages = {
        "opencv-python": "cv2",
        "numpy": "numpy", 
        "Pillow": "PIL",
        "google-genai": "google.genai"
    }
    
    all_installed = True
    
    for pip_name, import_name in required_packages.items():
        if not is_package_installed(import_name):
            all_installed = False
            log_console(f"Package '{pip_name}' not found. Attempting to install...")
            try:
                # Use --user flag to avoid permission issues
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pip_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log_console(f"✓ Successfully installed '{pip_name}'.")
            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name], 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log_console(f"✓ Successfully installed '{pip_name}'.")
                except subprocess.CalledProcessError as e:
                    error_msg = f"ERROR: Failed to install '{pip_name}'. Please install it manually. Error: {e}"
                    log_error(error_msg)
                    all_installed = False
    
    # Check for Tkinter (GUI toolkit)
    try:
        import tkinter
    except ImportError:
        error_msg = "✗ Tkinter is not available."
        log_error(error_msg)
        system = platform.system().lower()
        if system == "windows":
            log_error("Tkinter should be included with Python on Windows.")
            log_error("If missing, reinstall Python and make sure 'tcl/tk and IDLE' is selected.")
        elif system == "darwin":  # macOS
            log_error("On macOS, install tkinter with: brew install python-tk")
        elif system == "linux":
            log_error("On Linux, install tkinter with:")
            log_error("  Ubuntu/Debian: sudo apt-get install python3-tk")
            log_error("  Fedora: sudo dnf install python3-tkinter")
            log_error("  Arch: sudo pacman -S tk")
        all_installed = False
    
    # Check for basic font availability (silent check)
    font_check_passed = True
    try:
        from PIL import ImageFont
        basic_fonts = ["arial.ttf", "Arial.ttf"]
        font_found = False
        for font_name in basic_fonts:
            try:
                font = ImageFont.truetype(font_name, 12)
                font_found = True
                break
            except IOError:
                continue
        if not font_found:
            log_error("✗ Basic fonts (Arial) not found. The program may have display issues.")
            font_check_passed = False
    except Exception as e:
        log_error(f"✗ Could not check fonts: {e}")
        font_check_passed = False
    
    # Verify all critical imports work
    import_issues = []
    critical_imports = [
        ("os", "os"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("PIL.Image", "PIL"),
        ("re", "re"),
        ("time", "time"),
        ("tkinter", "tkinter"),
        ("json", "json"),
        ("sys", "sys"),
        ("threading", "threading"),
        ("subprocess", "subprocess"),
        ("importlib.util", "importlib"),
        ("urllib.request", "urllib"),
        ("tempfile", "tempfile")
    ]
    
    for import_name, package_name in critical_imports:
        try:
            if "." in import_name:
                module_name = import_name.split(".")[0]
                __import__(module_name)
            else:
                __import__(import_name)
        except ImportError as e:
            error_msg = f"✗ Failed to import {import_name}: {e}"
            log_error(error_msg)
            import_issues.append((import_name, package_name))
            all_installed = False
    
    # If there are any errors, show console and save to file
    if errors:
        # Print all console output to make errors visible
        for line in console_output:
            print(line)
        
        # Save to error file
        with open("error.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(console_output))
        
        # Keep console open on Windows
        if platform.system() == "Windows":
            print("\nErrors have been saved to error.txt")
            input("Press Enter to exit...")
        
        return False
    
    # If we get here, everything is installed correctly
    return True


# --- Part 1: Image Processing Functions ---

def non_max_suppression(boxes, scores, overlapThresh):
    # Performs non-maximum suppression to filter overlapping bounding boxes.
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return pick

def create_char_image(font_path, font_size, text):
    # Creates an image of a given text string.
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"# Warning: Font not found at '{font_path}'. Using default.")
        font = ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text)
    img = Image.new('RGB', (right - left, bottom - top), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((-left, -top), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def load_template(path):
    # Loads a template image in grayscale and returns it with its dimensions.
    if not path or not path.strip(): # Skip if path is blank
        return None, 0, 0
    try:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None: raise FileNotFoundError
        h, w = template.shape
        return template, w, h
    except (FileNotFoundError, AttributeError):
        print(f"# Warning: Template not found at '{path}'.")
        return None, 0, 0

def perform_all_replacements_on_image(img, replacements, font_paths):
    # Applies all configured replacements to a single image in memory.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for r in replacements:
        template, t_w, t_h = r['template_data']
        if template is None: continue

        font_key = r.get("font_key", "default")
        selected_font_path = font_paths.get(font_key, font_paths["default"])
        char_img = create_char_image(selected_font_path, r['font_size'], r['text'])
        c_h, c_w, _ = char_img.shape

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        y_coords, x_coords = np.where(result >= r['threshold'])
        
        if not y_coords.any(): continue

        boxes = np.array([[x, y, x + t_w, y + t_h] for x, y in zip(x_coords, y_coords)])
        picked_indices = non_max_suppression(boxes, result[y_coords, x_coords], 0.3)

        for i in picked_indices:
            (x, y) = boxes[i][:2].astype(int)
            img[y:y+t_h, x:x+t_w] = (255, 255, 255) # White out area
            
            paste_x = x + ((t_w - c_w) // 2) + r.get('align_x_offset', 0)
            paste_y = y + (t_h - c_h) // 2
            
            if paste_y >= 0 and paste_x >= 0 and paste_y + c_h <= img.shape[0] and paste_x + c_w <= img.shape[1]:
                img[paste_y:paste_y + c_h, paste_x:paste_x + c_w] = char_img
    return img

def crop_final_whitespace(image_folder, margin=50):
    # Crops whitespace from the top and bottom of processed images.
    print(f"# Cropping final whitespace in {os.path.basename(image_folder)}...")
    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(mask)
        if coords is None: continue

        x, y, w, h = cv2.boundingRect(coords)
        crop_top = max(0, y - margin)
        crop_bottom = min(img.shape[0], y + h + margin)
        cv2.imwrite(path, img[crop_top:crop_bottom, :])

def merge_images(processed_folder, merged_folder, chunk_size=3):
    # Merges processed images vertically into chunks.
    print(f"# Merging images into {os.path.basename(merged_folder)}...")
    try:
        image_files = sorted([f for f in os.listdir(processed_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"# Error: Folder '{processed_folder}' not found.")
        return

    if not image_files: return
    
    os.makedirs(merged_folder, exist_ok=True)
    image_chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    for i, chunk in enumerate(image_chunks):
        images_to_merge = [Image.open(os.path.join(processed_folder, fname)) for fname in chunk]
        max_width = max(img.width for img in images_to_merge)
        total_height = sum(img.height for img in images_to_merge)

        merged_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        current_y = 0
        for img in images_to_merge:
            merged_image.paste(img, (0, current_y))
            current_y += img.height
            img.close()
        
        merged_image.save(os.path.join(merged_folder, f"merged_{i + 1}.png"))

# --- Part 2: OCR Functions ---

def configure_gemini(api_key):
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None


def ocr_images_in_folder(folder_path, client, model_name, prompt, config):
    print(f"\n# OCR Processing folder: {os.path.basename(folder_path)}")

    # Define generation config to make output deterministic
    generation_config = types.GenerateContentConfig(temperature=0)
    
    # Define safety settings to disable all filters
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
    
    # Find all .png/.jpg/.jpeg files and sort them alphabetically.
    try:
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"  - Error: Directory not found at {folder_path}")
        return ""
        
    if not image_files:
        print("  - No images found in this folder.")
        return ""

    all_ocr_texts = []

    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        MAX_RETRIES = 10
        RETRY_DELAY_SECONDS = 31
        ocr_text = ""

        for attempt in range(MAX_RETRIES):
            try:
                print(f"  - Processing image: {image_name} ({attempt + 1}/{MAX_RETRIES})")

                img = Image.open(image_path)

                if config.get("USE_UNSHARP_MASK", False):
                    img = img.filter(
                        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                    )

                if config.get("USE_BINARIZATION", False):
                    ocv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                    _, ocv = cv2.threshold(ocv, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img = Image.fromarray(ocv)

                # Convert PIL image to bytes
                from io import BytesIO
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Construct the content with new library format
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                        ]
                    )
                ]

                # Use the new API
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        safety_settings=safety_settings
                    )
                )

                if not response.text:
                    print("  - Empty response, retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue

                ocr_text = response.text.strip()
                print("  - OCR successful.")
                break

            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ("500", "internal", "unavailable")):
                    print(f"  - Server error, retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"  - Non-retriable error: {e}")
                    break

        if ocr_text:
            all_ocr_texts.append(ocr_text)
        else:
            print(f"  - FAILED: {image_name}")

    return "\n".join(all_ocr_texts)

# --- Part 3: Text Processing Functions ---

def fix_unbalanced_quotes(text):
    # Corrects lines with an odd number of quotation marks.
    def fix_line(line):
        for quote in ('"', "'"):
            stripped = line.strip()
            if stripped.count(quote) % 2 == 0: continue
            if stripped.startswith(quote) and not stripped.endswith(quote):
                line = line.rstrip() + quote
            elif stripped.endswith(quote) and not stripped.startswith(quote):
                whitespace = line[:len(line) - len(line.lstrip())]
                line = whitespace + quote + line.lstrip()
        return line
    return "\n".join(map(fix_line, text.splitlines()))

def remove_blank_lines(text):
    # Removes all empty lines from the text.
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def merge_lines(text, sentence_enders, exception_pattern):
    # Merges lines that are part of the same sentence.
    lines = text.splitlines()
    if not lines: return ""
    parts = []
    for i, current_line in enumerate(lines):
        parts.append(current_line)
        if i == len(lines) - 1: break
        stripped_current = current_line.strip()
        next_line = lines[i + 1]
        stripped_next = next_line.strip()
        force_merge = False
        quote_char = stripped_current[0] if stripped_current and stripped_current[0] in ('"', "'") else None
        if (quote_char and stripped_current.endswith(('.', '?', '!', '…', '-', '~')) and stripped_next.endswith(quote_char) and stripped_current.count(quote_char) == 1):
            force_merge = True
        if force_merge:
            parts.append(' ')
        elif stripped_current.endswith(sentence_enders) or exception_pattern.match(stripped_current):
            parts.append('\n')
        else:
            parts.append(' ')
    return "".join(parts)

def process_and_refine_text_file(filepath, replacements, sentence_enders, exception_pattern):
    # Main function to apply all text processing steps to a file.
    try:
        with open(filepath, 'r', encoding='utf8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    for old, new in replacements.items():
        content = content.replace(old, new)
        
    content = remove_blank_lines(content)
    content = merge_lines(content, sentence_enders, exception_pattern)
    content = fix_unbalanced_quotes(content)

    with open(filepath, 'w', encoding='utf8') as f:
        f.write(content)

    print(f"File '{os.path.basename(filepath)}' has been processed and updated successfully.")

# --- Main Application Logic ---

def run_processing_logic(config):
    # Main script logic, adapted to run from the GUI.
    
    # --- Directory Setup ---
    script_directory = os.getcwd()
    screenshots_folder = os.path.join(script_directory, "screenshots")
    processed_folder = os.path.join(script_directory, "processed")
    rawocr_folder = os.path.join(script_directory, "rawocr")

    # --- Initial Setup ---
    print("--- Initializing Script ---")
    if not os.path.isdir(screenshots_folder):
        print(f"# Error: Input folder not found at '{screenshots_folder}'")
        return
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(rawocr_folder, exist_ok=True)
    
    font_paths = {
        "default": config["FONT_PATH"], 
        "hangul": config["FONT_PATH2"],
        "chinese": config["FONT_PATH3"],
        "japanese": config["FONT_PATH4"]
    }
    
    image_replacements = config["IMAGE_REPLACEMENT_CONFIGS"]
    for conf in image_replacements:
        conf['template_data'] = load_template(conf['template_path'])

    client = configure_gemini(config["GOOGLE_API_KEY"])
    if client is None:
        return

    ocr_model_name = config["OCR_MODEL"]

    # --- Main Processing Loop ---
    for folder_name in sorted(os.listdir(screenshots_folder)):
        input_folder = os.path.join(screenshots_folder, folder_name)
        if not os.path.isdir(input_folder): continue
        
        merged_output_folder = os.path.join(processed_folder, folder_name)
        indiv_folder_path = os.path.join(merged_output_folder, "indiv")
        
        output_filename_base = folder_name
        final_ocr_filepath = os.path.join(rawocr_folder, f"{output_filename_base}.txt")

        if os.path.exists(final_ocr_filepath):
            print(f"\n# Skipping {folder_name}: Final OCR file already exists.")
            continue

        print(f"\n# Starting processing for folder: {folder_name}")

        if not os.path.isdir(indiv_folder_path) or not os.listdir(indiv_folder_path):
            print("--- Starting Part 1: Image Processing ---")
            os.makedirs(indiv_folder_path, exist_ok=True)
            
            print(f"# Copying initial files to {os.path.basename(indiv_folder_path)}...")
            for filename in sorted(os.listdir(input_folder)):
                input_path = os.path.join(input_folder, filename)
                if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(input_path)
                    if img is not None:
                        cv2.imwrite(os.path.join(indiv_folder_path, filename), img)

            print(f"# Performing sequential replacement passes...")
            for conf in image_replacements:
                if not conf.get('template_path', '').strip():
                    continue # Skip if path is blank
                print(f"# > Pass: {conf['name']}")
                for filename in sorted(os.listdir(indiv_folder_path)):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    image_path = os.path.join(indiv_folder_path, filename)
                    img = cv2.imread(image_path)
                    if img is None: continue
                    modified_img = perform_all_replacements_on_image(img, [conf], font_paths)
                    cv2.imwrite(image_path, modified_img)
                
            crop_final_whitespace(indiv_folder_path, margin=50)
            
            merge_images(indiv_folder_path, merged_output_folder)
            
            print("--- Finished Part 1: Image Processing ---\n")
        else:
            print("# Skipping Part 1: Processed images already exist.")

        print("--- Starting Part 2: OCR ---")
        combined_text = ocr_images_in_folder(merged_output_folder, client, ocr_model_name, config["OCR_PROMPT"], config)

        if combined_text:
            # Build dynamic replacements for Part 3
            dynamic_replacements = config["TEXT_REPLACEMENTS"].copy()
            for item in config["IMAGE_REPLACEMENT_CONFIGS"]:
                if item.get("text") and item.get("desired_text"):
                    dynamic_replacements[item["text"]] = item["desired_text"]

            try:
                with open(final_ocr_filepath, 'w', encoding='utf8') as f:
                    f.write(combined_text)
                print(f"Successfully saved OCR text to: {final_ocr_filepath}")
            except IOError as e:
                print(f"Error writing to file {final_ocr_filepath}. Error: {e}")
            
            print("\n--- Starting Part 3: Text Processing and Refinement ---")
            process_and_refine_text_file(final_ocr_filepath, dynamic_replacements, tuple(config["SENTENCE_ENDERS"]), re.compile(config["EXCEPTION_PATTERN"]))
        
        print(f"--- Finished processing for {folder_name} ---")

    print("\n--- Script finished ---")

# --- GUI Application ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Foodiemonster007's OCR Tool")
        self.root.geometry("850x750")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.replacement_entries = []

        self.create_settings_tab()
        self.create_log_tab()
        
        self.load_initial_config()

    def create_settings_tab(self):
        settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(settings_frame, text='Settings')
        
        # --- API Key ---
        api_frame = ttk.LabelFrame(settings_frame, text="API and Model", padding="10")
        api_frame.pack(fill="x", expand=False, pady=5)
        ttk.Label(api_frame, text="Google API Key:").grid(row=0, column=0, sticky="w", pady=2)
        self.api_key_entry = ttk.Entry(api_frame, width=80)
        self.api_key_entry.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(api_frame, text="OCR Model:").grid(row=1, column=0, sticky="w", pady=2)
        self.ocr_model_entry = ttk.Entry(api_frame, width=80)
        self.ocr_model_entry.grid(row=1, column=1, sticky="ew", pady=2)
        api_frame.columnconfigure(1, weight=1)

        # --- Prompts ---
        prompts_frame = ttk.LabelFrame(settings_frame, text="Prompts", padding="10")
        prompts_frame.pack(fill="x", expand=False, pady=5)
        ttk.Label(prompts_frame, text="OCR Prompt:").pack(anchor="w")
        self.ocr_prompt_text = tk.Text(prompts_frame, height=4, width=80, wrap="word")
        self.ocr_prompt_text.pack(fill="x", expand=True)
        
        # --- Exception Pattern ---
        ttk.Label(prompts_frame, text="Chapter Title Pattern (Regex):").pack(anchor="w", pady=(5,0))
        self.exception_pattern_entry = ttk.Entry(prompts_frame, width=80)
        self.exception_pattern_entry.pack(fill="x", expand=True)

        # --- OCR Preprocessing ---
        ocr_proc_frame = ttk.LabelFrame(settings_frame, text="OCR Image Preprocessing", padding="10")
        ocr_proc_frame.pack(fill="x", expand=False, pady=5)
        self.unsharp_var = tk.BooleanVar()
        self.binarization_var = tk.BooleanVar()
        ttk.Checkbutton(ocr_proc_frame, text="Unsharp Mask", variable=self.unsharp_var).pack(side="left", padx=10)
        ttk.Checkbutton(ocr_proc_frame, text="Otsu's Binarization", variable=self.binarization_var).pack(side="left", padx=10)

        # --- Image Replacements ---
        replacements_frame = ttk.LabelFrame(settings_frame, text="Image Replacements", padding="10")
        
        ttk.Label(replacements_frame, text="Delete all image replacements if you don't want to replace symbols with custom text.").pack(anchor="w", pady=(0, 5))
        
        replacement_button_frame = ttk.Frame(replacements_frame)
        replacement_button_frame.pack(fill="x", pady=5)
        self.add_replacement_button = ttk.Button(replacement_button_frame, text="Add More Replacements", command=self.add_replacement_row)
        self.add_replacement_button.pack(side="left", padx=5)
        self.delete_replacement_button = ttk.Button(replacement_button_frame, text="Delete All Replacements", command=self.delete_all_replacements)
        self.delete_replacement_button.pack(side="left", padx=5)

        self.canvas = tk.Canvas(replacements_frame)
        scrollbar = ttk.Scrollbar(replacements_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to canvas for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # --- Buttons ---
        button_frame = ttk.Frame(settings_frame)
        
        right_button_frame = ttk.Frame(button_frame)
        right_button_frame.pack(side="right")
        
        self.set_default_button = ttk.Button(right_button_frame, text="Set Default Config", command=self.set_default_config)
        self.set_default_button.pack(side="left", padx=5)

        self.save_button = ttk.Button(right_button_frame, text="Save Config As...", command=self.save_config_dialog)
        self.save_button.pack(side="left", padx=5)
        
        self.load_button = ttk.Button(right_button_frame, text="Load Config", command=self.load_config_dialog)
        self.load_button.pack(side="left", padx=5)

        self.run_button = ttk.Button(right_button_frame, text="Run Process", command=self.run_process_thread)
        self.run_button.pack(side="left", padx=5)

        # --- PACKING ORDER ---
        button_frame.pack(side="bottom", fill="x", pady=10, anchor="s")
        replacements_frame.pack(fill="both", expand=True, pady=5)

    def _on_mousewheel(self, event):
        # Cross-platform mouse wheel scrolling
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_log_tab(self):
        log_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(log_frame, text='Log')
        
        self.log_text = tk.Text(log_frame, height=30, width=95, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")

    def get_default_config(self):
        return {
            "GOOGLE_API_KEY": "YOUR_API_KEY",
            "FONT_PATH": "arial.ttf",
            "FONT_PATH2": "malgun.ttf",
            "FONT_PATH3": "msyh.ttc",  # Microsoft YaHei for Chinese
            "FONT_PATH4": "msgothic.ttc", # MS Gothic for Japanese
            "IMAGE_REPLACEMENT_CONFIGS": [
                {"name": "Ellipsis", "template_path": "ellipsis.png", "text": "@", "desired_text": "...", "font_size": 36, "threshold": 0.80, "font_key": "default"},
                {"name": "Separator", "template_path": "separator.png", "text": "ABCDEFGHIJKLMNOP", "desired_text": "***", "font_size": 44, "threshold": 0.7, "font_key": "default"}
            ],
            "OCR_MODEL": 'gemini-2.5-flash',
            "OCR_PROMPT": "Perform OCR on this image. The text is in Korean and may include Hanja characters, English letters and repetitive sound effects (e.g., '콰콰콰콰콰' or '스스스스스'). Transcribe the text exactly as it appears, preserving all original characters including Hanja.",
            "SENTENCE_ENDERS": ['.', '?', '!', '…', "'", '"', '-', '*', '~', '。', '？', '！', '」', '』'],
            "EXCEPTION_PATTERN": r'^\s*제\d+회',
            "TEXT_REPLACEMENTS": { '‘': "'", '’': "'", '“': '"', '”': '"', ' ,': ',', ' !': '!', ' ?': '?', '-1': '-!' },
            "USE_UNSHARP_MASK": True,
            "USE_BINARIZATION": False
        }

    def load_initial_config(self):
        # Load the hardcoded default config.json if it exists, otherwise use the internal defaults.
        self.load_config("config.json")

    def load_config(self, filepath=None):
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding='utf-8') as f:
                    self.config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                messagebox.showerror("Error", f"Failed to load config file: {e}")
                self.config = self.get_default_config()
        else:
             self.config = self.get_default_config()

        self.api_key_entry.delete(0, tk.END)
        self.api_key_entry.insert(0, self.config.get("GOOGLE_API_KEY", "YOUR_API_KEY"))
        self.ocr_model_entry.delete(0, tk.END)
        self.ocr_model_entry.insert(0, self.config.get("OCR_MODEL", 'gemini-2.5-pro'))
        self.ocr_prompt_text.delete("1.0", tk.END)
        self.ocr_prompt_text.insert("1.0", self.config.get("OCR_PROMPT", ""))
        self.exception_pattern_entry.delete(0, tk.END)
        self.exception_pattern_entry.insert(0, self.config.get("EXCEPTION_PATTERN", r'^\s*제\d+회'))
        
        self.unsharp_var.set(self.config.get("USE_UNSHARP_MASK", True))
        self.binarization_var.set(self.config.get("USE_BINARIZATION", False))
        
        self.delete_all_replacements()

        for item in self.config.get("IMAGE_REPLACEMENT_CONFIGS", []):
            self.add_replacement_row(item)

    def load_config_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Open Config File",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filepath:
            self.load_config(filepath)

    def add_replacement_row(self, item=None):
        if item is None:
            item = {"name": "New Replacement", "template_path": "", "text": "", "desired_text": "", "font_size": 32, "threshold": 0.8, "font_key": "default"}
        
        entry_frame = ttk.Frame(self.scrollable_frame)
        entry_frame.pack(fill="x", expand=True, pady=5)
        
        entry_widgets = {}

        # --- First Row ---
        row1_frame = ttk.Frame(entry_frame)
        row1_frame.pack(fill="x", expand=True)
        
        delete_button = ttk.Button(row1_frame, text="❌", width=3, command=lambda f=entry_frame: self.delete_replacement_row(f))
        delete_button.grid(row=0, column=0, padx=(0,5))

        ttk.Label(row1_frame, text="Name:").grid(row=0, column=1, sticky="w")
        entry_widgets['name'] = ttk.Entry(row1_frame)
        entry_widgets['name'].grid(row=0, column=2, sticky="ew")
        entry_widgets['name'].insert(0, item.get("name", ""))

        ttk.Label(row1_frame, text="Image Filename:").grid(row=0, column=3, sticky="w", padx=(10,0))
        entry_widgets['path'] = ttk.Entry(row1_frame)
        entry_widgets['path'].grid(row=0, column=4, sticky="ew")
        entry_widgets['path'].insert(0, item.get("template_path", ""))

        ttk.Label(row1_frame, text="Size:").grid(row=0, column=5, sticky="w", padx=(10,0))
        entry_widgets['size'] = ttk.Entry(row1_frame, width=5)
        entry_widgets['size'].grid(row=0, column=6)
        entry_widgets['size'].insert(0, item.get("font_size", 32))

        ttk.Label(row1_frame, text="Threshold:").grid(row=0, column=7, sticky="w", padx=(10,0))
        entry_widgets['threshold'] = ttk.Entry(row1_frame, width=5)
        entry_widgets['threshold'].grid(row=0, column=8)
        entry_widgets['threshold'].insert(0, item.get("threshold", 0.8))

        row1_frame.columnconfigure(2, weight=1)
        row1_frame.columnconfigure(4, weight=1)

        # --- Second Row ---
        row2_frame = ttk.Frame(entry_frame)
        row2_frame.pack(fill="x", expand=True, pady=(2,0))

        ttk.Label(row2_frame, text="Replacement Text:").grid(row=0, column=0, sticky="e", padx=5)
        entry_widgets['text'] = ttk.Entry(row2_frame)
        entry_widgets['text'].grid(row=0, column=1, sticky="ew")
        entry_widgets['text'].insert(0, item.get("text", ""))

        ttk.Label(row2_frame, text="Desired Text:").grid(row=0, column=2, sticky="e", padx=5)
        entry_widgets['desired_text'] = ttk.Entry(row2_frame)
        entry_widgets['desired_text'].grid(row=0, column=3, sticky="ew")
        entry_widgets['desired_text'].insert(0, item.get("desired_text", ""))
        
        row2_frame.columnconfigure(1, weight=1)
        row2_frame.columnconfigure(3, weight=1)
        
        self.replacement_entries.append({"frame": entry_frame, "widgets": entry_widgets})
        
        self.root.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def delete_replacement_row(self, frame_to_delete):
        # Find the entry to remove from the list
        index_to_delete = -1
        for i, entry in enumerate(self.replacement_entries):
            if entry["frame"] == frame_to_delete:
                index_to_delete = i
                break
        
        if index_to_delete != -1:
            self.replacement_entries.pop(index_to_delete)
            frame_to_delete.destroy()
            self.root.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_all_replacements(self):
        for entry in self.replacement_entries:
            entry["frame"].destroy()
        self.replacement_entries.clear()
        self.root.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def save_config_dialog(self):
        filepath = filedialog.asksaveasfilename(
            title="Save Config As",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if not filepath:
            return

        config = self.get_config_from_gui()
        if config:
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Success", f"Configuration saved to {os.path.basename(filepath)}")

    def set_default_config(self):
        config = self.get_config_from_gui()
        if config:
            try:
                with open("config.json", "w", encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Success", "Current settings have been saved as the default (config.json).")
            except IOError as e:
                messagebox.showerror("Error", f"Could not save default config: {e}")

    def run_process_thread(self):
        self.run_button.config(state="disabled")
        self.notebook.select(self.log_text.master)
        
        config = self.get_config_from_gui()
        if config:
            thread = threading.Thread(target=self.run_process, args=(config,))
            thread.start()
        else:
            self.run_button.config(state="normal")

    def get_config_from_gui(self):
        config = self.get_default_config()
        config["GOOGLE_API_KEY"] = self.api_key_entry.get()
        config["OCR_MODEL"] = self.ocr_model_entry.get()
        config["OCR_PROMPT"] = self.ocr_prompt_text.get("1.0", tk.END).strip()
        config["EXCEPTION_PATTERN"] = self.exception_pattern_entry.get()
        config["USE_UNSHARP_MASK"] = self.unsharp_var.get()
        config["USE_BINARIZATION"] = self.binarization_var.get()
        
        new_replacements = []
        try:
            # Regex patterns for CJK characters
            hangul_pattern = re.compile(r'[\uac00-\ud7a3]')
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            japanese_pattern = re.compile(r'[\u3040-\u30ff\u31f0-\u31ff]')

            for entry in self.replacement_entries:
                entry_set = entry["widgets"]
                text = entry_set['text'].get()
                
                font_key = "default"
                if hangul_pattern.search(text):
                    font_key = "hangul"
                elif chinese_pattern.search(text):
                    font_key = "chinese"
                elif japanese_pattern.search(text):
                    font_key = "japanese"

                item_config = {
                    "name": entry_set['name'].get(),
                    "template_path": entry_set['path'].get(),
                    "text": text,
                    "desired_text": entry_set['desired_text'].get(),
                    "font_size": int(entry_set['size'].get()),
                    "threshold": float(entry_set['threshold'].get()),
                    "font_key": font_key
                }
                new_replacements.append(item_config)
            config["IMAGE_REPLACEMENT_CONFIGS"] = new_replacements
            return config
        except (ValueError) as e:
            messagebox.showerror("Error", f"Invalid data in Image Replacements: {e}")
            return None

    def run_process(self, config):
        try:
            run_processing_logic(config)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            self.run_button.config(state="normal")
            messagebox.showinfo("Complete", "Processing has finished.")

class TextRedirector(object):
    # Helper class to redirect stdout/stderr to a Tkinter Text widget.
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.see(tk.END)
    
    def flush(self):
        pass

if __name__ == "__main__":
    check_and_install_packages()
    root = tk.Tk()
    app = App(root)
    root.mainloop()
