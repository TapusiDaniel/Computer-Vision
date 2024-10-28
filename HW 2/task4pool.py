import cv2
import numpy as np
import os
from PIL import Image

def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    s = 0 if cmax == 0 else (diff / cmax) * 100
    v = cmax * 100
    
    return h, s, v

def is_in_range(h, s, v, lower, upper):
    return (lower[0] <= h <= upper[0] and
            lower[1] <= s <= upper[1] and
            lower[2] <= v <= upper[2])

def blue_filter(init_image):
    height, width, _ = init_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    filtered_image = np.zeros_like(init_image)

    # Definirea culorii piscinei în format RGB
    pool_rgb = [19, 240, 250]  # BGR inversat pentru RGB
    pool_h, pool_s, pool_v = rgb_to_hsv(*pool_rgb)

    # Definirea intervalului HSV pentru piscină
    lower_blue = [max(0, pool_h - 5), 50, 50]
    upper_blue = [min(360, pool_h + 5), 100, 100]

    for y in range(height):
        for x in range(width):
            r, g, b = init_image[y, x]
            h, s, v = rgb_to_hsv(r, g, b)
            
            if is_in_range(h, s, v, lower_blue, upper_blue):
                mask[y, x] = 255
                filtered_image[y, x] = init_image[y, x]

    return filtered_image, mask

def contour_finding(init_image, mask_pool):
    # Găsirea contururilor
    contours, _ = cv2.findContours(mask_pool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortarea contururilor după arie (de la cel mai mare la cel mai mic)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_candidate = None
    best_score = 0

    for i, contour in enumerate(contours[:10]):
        # Găsirea dreptunghiului minim rotit
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Folosim np.intp în loc de np.int0 pentru a evita avertismentul
        
        # Calcularea raportului de aspect
        width = rect[1][0]
        height = rect[1][1]
        
        # Verificăm dacă width sau height sunt zero pentru a evita împărțirea la zero
        if width == 0 or height == 0:
            aspect_ratio = 0
        else:
            aspect_ratio = max(width, height) / min(width, height)
        
        area = cv2.contourArea(contour)
        
        print(f"Contur {i+1}: Arie = {area:.2f}, Aspect ratio = {aspect_ratio:.2f}")
        
        # Calculăm procentul de pixeli corespunzători în interiorul conturului
        mask = np.zeros(mask_pool.shape, np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        matching_pixel_count = cv2.countNonZero(cv2.bitwise_and(mask_pool, mask))
        total_pixel_count = cv2.countNonZero(mask)
        matching_percentage = matching_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
        
        score = area * matching_percentage
        
        print(f"Contur {i+1}: Procent potrivire = {matching_percentage:.2f}, Scor = {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_candidate = box

    final_image = init_image.copy()

    if best_candidate is not None:
        # Desenarea conturului dreptunghiului
        cv2.drawContours(final_image, [best_candidate], 0, (0, 255, 0), 2)
        cv2.putText(final_image, "Piscina detectata", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("Piscina detectata cu succes.")
    elif len(contours) == 1:
        # Dacă există doar un contur, îl conturăm chiar dacă nu îndeplinește criteriile
        single_contour = contours[0]
        rect = cv2.minAreaRect(single_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(final_image, [box], 0, (0, 255, 0), 2)
        cv2.putText(final_image, "Posibila piscina", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("S-a detectat un singur contur, posibil o piscină.")
    else:
        print("Nu s-a detectat nicio piscină în imagine.")
    
    return final_image

def process_image(image_path):
    # Citirea imaginii
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicarea filtrului albastru
    filtered_blue, mask_pool = blue_filter(image)

    # Aplicarea filtrului median folosind OpenCV
    filtered_blue = cv2.medianBlur(filtered_blue, 9)

    # Găsirea și desenarea contururilor
    result_image = contour_finding(image, mask_pool)

    # Salvarea imaginilor rezultate
    output_dir = 'results/pool_detection'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f'{output_dir}/detected_pool.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_dir}/pool_mask.jpg', mask_pool)
    
    print(f"Rezultatele au fost salvate în {output_dir}/")

    # Afișarea imaginilor (opțional)
    cv2.imshow("Detected Pool", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Pool Mask", mask_pool)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Rularea funcției principale
if __name__ == "__main__":
    image_path = "/home/danyez87/Master AI/CV/HW 2/Gura_Portitei_Scara_0025.jpg"  # Înlocuiește cu calea către imaginea ta
    process_image(image_path)