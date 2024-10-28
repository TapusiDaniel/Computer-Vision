import cv2
import numpy as np
import os

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

def helipad_filter(init_image):
    height, width, _ = init_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    filtered_image = np.zeros_like(init_image)

    # Definirea culorii helipadului în format RGB
    helipad_rgb = [212, 139, 143]  # BGR inversat pentru RGB
    helipad_h, helipad_s, helipad_v = rgb_to_hsv(*helipad_rgb)

    # Definirea intervalului HSV pentru helipad
    lower_helipad = [max(0, helipad_h - 20), 20, 20]
    upper_helipad = [min(360, helipad_h + 20), 100, 100]

    for y in range(height):
        for x in range(width):
            r, g, b = init_image[y, x]
            h, s, v = rgb_to_hsv(r, g, b)
            
            if is_in_range(h, s, v, lower_helipad, upper_helipad):
                mask[y, x] = 255
                filtered_image[y, x] = init_image[y, x]

    return filtered_image, mask

def contour_finding(init_image, mask_helipad):
    # Găsirea contururilor
    contours, _ = cv2.findContours(mask_helipad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortarea contururilor după arie (de la cel mai mare la cel mai mic)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_candidate = None
    best_score = 0
    best_square = None

    for i, contour in enumerate(contours[:20]):
        # Aproximarea conturului la un poligon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        
        # Verificarea dacă poligonul are între 4 și 8 laturi
        if 4 <= len(approx) <= 8:
            # Calcularea raportului de aspect folosind dreptunghiul încadrant rotit
            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), angle = rect
            aspect_ratio = max(width, height) / min(width, height)
            
            print(f"Contur {i+1}: Laturi = {len(approx)}, Aspect ratio = {aspect_ratio:.2f}")
            
            # Verificăm dacă forma este aproximativ pătrată
            if 0.7 <= aspect_ratio <= 1.5:
                area = cv2.contourArea(contour)
                
                # Calculăm procentul de pixeli corespunzători în interiorul conturului
                mask = np.zeros(mask_helipad.shape, np.uint8)
                cv2.drawContours(mask, [approx], 0, 255, -1)
                matching_pixel_count = cv2.countNonZero(cv2.bitwise_and(mask_helipad, mask))
                total_pixel_count = cv2.countNonZero(mask)
                matching_percentage = matching_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
                
                score = area * matching_percentage
                
                print(f"Contur {i+1}: Arie = {area:.2f}, Procent potrivire = {matching_percentage:.2f}, Scor = {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_candidate = approx
                    
                    # Calculăm pătratul perfect
                    side_length = int(np.sqrt(area))
                    center, _, _ = rect
                    half_side = side_length // 2
                    best_square = np.array([
                        [center[0] - half_side, center[1] - half_side],
                        [center[0] + half_side, center[1] - half_side],
                        [center[0] + half_side, center[1] + half_side],
                        [center[0] - half_side, center[1] + half_side]
                    ], dtype=np.int32)

    final_image = init_image.copy()

    if best_square is not None:
        # Desenarea pătratului perfect
        cv2.drawContours(final_image, [best_square], 0, (0, 255, 0), 2)
        cv2.putText(final_image, "Helipad detectat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("Helipad detectat cu succes.")
    else:
        print("Nu s-a detectat niciun helipad în imagine.")
    
    return final_image

def process_image(image_path):
    # Citirea imaginii
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicarea filtrului pentru helipad
    filtered_helipad, mask_helipad = helipad_filter(image)

    # Aplicarea filtrului median folosind OpenCV
    filtered_helipad = cv2.medianBlur(filtered_helipad, 9)

    # Găsirea și desenarea contururilor
    result_image = contour_finding(image, mask_helipad)

    # Salvarea imaginilor rezultate
    output_dir = 'results/helipad_detection'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f'{output_dir}/detected_helipad.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_dir}/helipad_mask.jpg', mask_helipad)
    
    print(f"Rezultatele au fost salvate în {output_dir}/")

    # Afișarea imaginilor (opțional)
    cv2.imshow("Detected Helipad", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Helipad Mask", mask_helipad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Rularea funcției principale
if __name__ == "__main__":
    image_path = "/home/danyez87/Master AI/CV/HW 2/Gura_Portitei_Scara_0025.jpg"  # Înlocuiește cu calea către imaginea ta
    process_image(image_path)