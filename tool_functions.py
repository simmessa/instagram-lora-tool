import os
import instaloader
import os
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
# import dlib # new face detection, no dll :(

# Functions 
class tools():
    def __init__(self):
        """
        Initialize the image captioning model.
        """
        # Processor and model variables for the class
        # cannot use fast because rocM is not supported on windows :(
        print("Is CUDA enabled?",torch.cuda.is_available(),"\n\n")
        
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        # quicker
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", dtype=torch.float16).to("cuda")
        
    def scrape_instagram_images(self, username, max_images=None):
        """
        Scrape Instagram images for a given username.

        Args:
            username (str): The username of the Instagram profile.
            max_images (int): The maximum number of images to scrape (optional).

        Returns:
            None
        """
        # Create a directory to store the scraped images
        dir = f"scrapped/{username}"
        os.makedirs(dir, exist_ok=True)
        
        # Create an instance of the Instaloader class
        loader = instaloader.Instaloader(dirname_pattern=dir)

        # Retrieve the profile metadata for the specified username
        try:
            profile = instaloader.Profile.from_username(loader.context, username)
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"Profile '{username}' does not exist or is not accessible.")
            return    

        # Iterate over the profile's posts and download the images
        count = 0  # Counter for the number of downloaded images
        for post in profile.get_posts():
            # Skip posts that are not images
            if not post.is_video:
                # Download the image
                try:
                    loader.download_post(post, target=os.path.join(dir))
                    print(f"Downloaded image: {post.date_utc}")
                    count += 1

                    # Break the loop if the maximum number of images is reached
                    if max_images is not None and count >= max_images:
                        break
                except Exception as e:
                    print(f"Failed to download image: {post.date_utc}. Error: {str(e)}")

        print("Scraping completed successfully.")


    def get_folder_list(self, path):
        """
        Get a list of subdirectories (folders) in the specified path.

        Args:
            path (str): The path to the directory.

        Returns:
            list: A list of subdirectory names.

        """
        # Get a list of subdirectories (folders) in the specified path
        folder_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return folder_list

    def select_folder(self, folder):
        """
        Process the selected folder and perform any desired actions with it.

        Args:
            folder (str): The name of the selected folder.

        Returns:
            None
        """
        # Process the selected folder
        print(f"Selected folder: {folder}")

    def process_images(self, username):
        """
        Process the images in the specified user's folder by detecting faces, cropping around the largest face,
        and resizing the cropped image while maintaining aspect ratio.

        Args:
            username (str): The username associated with the images.

        Returns:
            Text output

        """
        # Size training size for kohya Kohya
        new_size = (512, 512)
        # Folder where the scrapped images are
        input_folder = f"scrapped/{username}"
        # Create the output folder if it doesn't exist
        output_folder = f"scrapped/{username}/cropped_centered"
        os.makedirs(output_folder, exist_ok=True)

        # for output
        out = ""

        # Load the pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Iterate over the images in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.jpg', '.png')):
                # Read the image
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                # Convert the image to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image with an old method Haar from 2001
                # faces = face_cascade.detectMultiScale(
                #     gray,
                #     scaleFactor=1.1, #1.1
                #     minNeighbors=5,
                #     minSize=(60, 60) #30, 30
                # )

                detector = cv2.FaceDetectorYN.create(
                    "face_detection_yunet_2023mar.onnx",
                    "",
                    (320, 320),
                    0.9,
                    0.3,
                    5000
                )

                tm = cv2.TickMeter()

                tm.start()
                ## [inference]
                # Set input size before inference
                detector.setInputSize((image.shape[1], image.shape[0]))

                faces = detector.detect(image)
                ## [inference]
                tm.stop()

                # If there are faces in image
                # print(type(faces), len(faces), faces)

                # if len(faces) == 1: #only if one face is detected
                if faces[1] is not None: #at least one face is detected

                    # get rectangle from detected face
                    # for (x, y, w, h) in faces[1]:
                    # for (x, y, w, h) in tuple(faces[1][:4]): #omg
                    detected = faces[1][0] # get the array with detection data
                    # print(faces)
                    (xf, yf, wf, hf) = tuple(detected[:4])
                    (x, y, w, h) = (int(xf), int(yf), int(wf), int(hf)) #float to int
                    # print (x, y, w, h)                  

                    # """                    
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green color, thickness 2
                    output_path = os.path.join(output_folder, filename)
                    print("Face detected in image:", filename)
                    cv2.imwrite(output_path, image) #SM avoid distortion if not on guitar :D
                    # Calculate the center of the face
                    center_x = x + w // 2
                    center_y = y + h // 2

                    out = out + f"Detected Face center is: {center_x}, {center_y}\n"
                    # print(out)

                    # Calculate the coordinates for cropping the image around the face
                    half_size = min(image.shape[0], image.shape[1]) // 2
                    x1 = max(center_x - half_size, 0)
                    y1 = max(center_y - half_size, 0)
                    x2 = min(center_x + half_size, image.shape[1])
                    y2 = min(center_y + half_size, image.shape[0])

                    # Calculate the aspect ratio of the original image
                    original_ratio = image.shape[1] / image.shape[0]
                    out = out + f"{image.shape[1]}/{image.shape[0]} | image ratio is: {original_ratio}\n"

                    # Crop the image around the face
                    cropped_image = image[y1:y2, x1:x2]

                    # Calculate the aspect ratio of the cropped image
                    cropped_height, cropped_width = cropped_image.shape[:2]
                    cropped_ratio = cropped_width / cropped_height
                    out = out + f"{cropped_width}/{cropped_height} | cropped ratio is: {cropped_ratio}\n"

                    # Calculate dimensions for maintaining aspect ratio within new_size
                    target_width, target_height = new_size
                    if cropped_ratio > 1:  # landscape
                        new_width = target_width
                        new_height = int(target_width / cropped_ratio)
                    else:  # portrait or square
                        new_height = target_height
                        new_width = int(target_height * cropped_ratio)

                    target_ratio = new_width / new_height
                    out = out + f"{new_width}/{new_height} | target ratio is: {target_ratio}\n"

                    # Resize the cropped image while maintaining aspect ratio
                    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

                    # Create a black background image with target size
                    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)

                    # Calculate position to center the resized image
                    x_offset = (target_width - new_width) // 2
                    y_offset = (target_height - new_height) // 2

                    # Place the resized image on the black background
                    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

                    # Save the processed image to the output folder
                    output_path = os.path.join(output_folder, "crop_" + filename)
                    cv2.imwrite(output_path, background) #SM avoid distortion if not on guitar :D
                    # print("Crop and resize: %s, %s" % (filename, original_ratio))
                    # """
        #finish with output
        return(out)


    def blip_captioning(self, username):
        """
        Generate descriptive captions for the images in the specified user's folder using the BLIP Hugging Face model.
        The captions are stored in separate text files with the same name as the corresponding image.

        Args:
            username (str): The username associated with the images.

        Returns:
            None

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Create the output folder if it doesn't exist
        output_folder = f"scrapped/{username}/cropped_centered"
        os.makedirs(output_folder, exist_ok=True)
        
        # For every image (png or jpg) in input_folder:
        for image_file in os.listdir(input_folder):
            if image_file.endswith(".png") or image_file.endswith(".jpg"):
                # Add images files to list of paths varaible
                input_path = f"scrapped/{username}/cropped_centered"
                print("input_path: " + input_path)
                
                image_path = os.path.join(input_path, image_file)
                # Replace backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                
                print("image_path: " + image_path)
                
                # Generate descriptive captioning with BLIP Hugging Face
                blip_caption = self.generate_blip_caption(username ,image_path)
                
                # Generate a txt file with the same name as the image
                txt_filename = os.path.splitext(image_file)[0] + ".txt"
                txt_filepath = os.path.join(output_folder, txt_filename)
                
                # Store the BLIP captioning in the corresponding txt file,
                # always using the username variable as the first word
                if os.path.exists(txt_filepath):
                    with open(txt_filepath, "w") as txt_file:
                        # Append caption to the existing file
                        txt_file.write(f"{blip_caption}\n")
                else:
                    with open(txt_filepath, "w") as txt_file:
                        # Create a new file with the caption
                        txt_file.write(f"{blip_caption}\n")  

    def generate_blip_caption(self, username, image_path):
        """
        Generate a descriptive caption for the specified image using the BLIP Hugging Face model.

        Args:
            username (str): The username associated with the image.
            image_path (str): The path to the image.

        Returns:
            str: The generated caption for the image.

        """
        # Load the image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Conditional image captioning
        text = f"a image of {username}, "
        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)
        
        max_length = 100
        out = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
        
    def change_image_blip_next(self, username, index_image):
        """
        Change the selected image and corresponding text for the BLIP Hugging Face model to the next image in the sequence.

        Args:
            username (str): The username associated with the images.
            index_image (int): The current index of the selected image.

        Returns:
            tuple: A tuple containing the path of the selected image, the updated index, and the text associated with the image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of image paths in the input folder
        image_paths = [
            os.path.join(input_folder, image_file) 
            for image_file in os.listdir(input_folder) 
            if image_file.endswith(('.png', '.jpg'))
        ]
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        index_image += 1
        
        if 0 <= index_image < len(image_paths):
            # Get the image path of the specified index
            selected_image_path = image_paths[index_image]
            # Read the text from the selected TXT file
            selected_txt_path = txt_paths[index_image]
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
        else:
            # If the index is out of range, reset the index to 0
            index_image = 0
            selected_image_path = image_paths[index_image]
            selected_txt_path = txt_paths[index_image]
            # Read the text from the selected TXT file
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
            
        # Replace backslashes with forward slashes in the file path
        selected_image_path = selected_image_path.replace("\\", "/")
        
        return selected_image_path, index_image, text

    def change_image_blip_previous(self, username, index_image):
        """
        Change the selected image and corresponding text for the BLIP Hugging Face model to the next image in the sequence.

        Args:
            username (str): The username associated with the images.
            index_image (int): The current index of the selected image.

        Returns:
            tuple: A tuple containing the path of the selected image, the updated index, and the text associated with the image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of image paths in the input folder
        image_paths = [
            os.path.join(input_folder, image_file) 
            for image_file in os.listdir(input_folder) 
            if image_file.endswith(('.png', '.jpg'))
        ]
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        index_image -= 1
        
        if 0 <= index_image < len(image_paths):
            # Get the image path of the specified index
            selected_image_path = image_paths[index_image]
             # Read the text from the selected TXT file
            selected_txt_path = txt_paths[index_image]
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
        else:
            # If the index is out of range, reset the index to 0
            index_image = len(image_paths)-1
            selected_image_path = image_paths[index_image]
            selected_txt_path = txt_paths[index_image]
            # Read the text from the selected TXT file
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
            
        # Replace backslashes with forward slashes in the file path
        selected_image_path = selected_image_path.replace("\\", "/")
        
        return selected_image_path, index_image, text

    def change_caption(self, username, caption, index_image):
        """
        Change the caption of the selected image.

        Args:
            username (str): The username associated with the images.
            caption (str): The new caption to be assigned to the selected image.
            index_image (int): The index of the selected image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of txt paths in the input folder
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        # Read the text from the selected TXT file
        selected_txt_path = txt_paths[index_image]
        
        with open(selected_txt_path, "w") as txt_file:
            # Create a new file with the caption
            txt_file.write(f"{caption}\n")
