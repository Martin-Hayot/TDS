import cv2
import numpy as np


def extract_hand(image_path, min_red_difference=20):
    # Charger l'image
    image = cv2.imread(image_path)

    # Convertir l'image de BGR à RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialiser le masque avec des pixels noirs
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Trouver les pixels de la main en comparant les composantes rouge, verte et bleue
    hand_pixels = (image_rgb[:,:,0] > image_rgb[:,:,1] + min_red_difference) & \
                  (image_rgb[:,:,0] > image_rgb[:,:,2] + min_red_difference)

    # Mettre à blanc les pixels correspondant à la main dans le masque
    mask[hand_pixels] = 255

    # Appliquer le masque à l'image originale pour isoler la main
    hand = cv2.bitwise_and(image, image, mask=mask)

    # Afficher l'image originale et l'image avec la main isolée
    cv2.imshow("Original Image", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Isolated Hand", hand)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Appeler la fonction avec le chemin de votre image
extract_hand("./WIN_20231202_15_58_17_Pro.jpg")
