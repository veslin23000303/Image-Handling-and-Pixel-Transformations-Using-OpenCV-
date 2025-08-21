# Image-Handling-and-Pixel-Transformations-Using-OpenCV-

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- *Name:*  Motta Katta Mounika
- *Register Number:* 212224040202

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('DIPT image-1.jpg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#### 2. Print the image width, height & Channel.
python
img.shape

### OUTPUT:


(1280, 591, 3)

#### 3. Display the image using matplotlib imshow().
python
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray,cmap='grey')
plt.show()


#### 4. Save the image as a PNG file using OpenCV imwrite().
python
img=cv2.imread('DIPT image-1.jpg')
cv2.imwrite('DIPT_image.png',img)

### OUTPUT:

True

#### 5. Read the saved image above as a color image using cv2.cvtColor().
python
img=cv2.imread('DIPT image-1.jpg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
python
plt.imshow(img)
plt.show()
img.shape


#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
python
crop = img_rgb[0:450,200:550] 
plt.imshow(crop[:,:,::-1])
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape


#### 8. Resize the image up by a factor of 2x.
python
res= cv2.resize(crop,(200*2, 200*2))


#### 9. Flip the cropped/resized image horizontally.
python
flip= cv2.flip(res,1)
plt.imshow(flip[:,:,::-1])
plt.title("Flipped Horizontally")
plt.axis("off")

#### 10. Read in the image ('Apollo-11-launch.jpg').
python
img=cv2.imread('DIPT image-2.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb.shape

### OUTPUT:

(508, 603, 3)


#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
python
text = cv2.putText(img_rgb, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
plt.imshow(text, cmap='gray')  
plt.title("New image")
plt.show()


#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
python
rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)

### OUTPUT:

array([[[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       [[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       [[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       ...,

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]],

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]],

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]]], shape=(508, 603, 3), dtype=uint8)

#### 13. Display the final annotated image.
python
plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()


#### 14. Read the image ('Boy.jpg').
python
img =cv2.imread('DIPT image-3.jpg',cv2.IMREAD_COLOR)
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#### 15. Adjust the brightness of the image.
python
m = np.ones(img_rgb.shape, dtype="uint8") * 50


#### 16. Create brighter and darker images.
python
img_brighter = cv2.add(img_rgb, m)  
img_darker = cv2.subtract(img_rgb, m)


#### 17. Display the images (Original Image, Darker Image, Brighter Image).
python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker), plt.title("Darker Image"), plt.axis("off")
plt.show()


#### 18. Modify the image contrast.
python
matrix1 = np.ones(img_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_rgb.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img.astype("float32"), matrix1).clip(0,255).astype("uint8")
img_higher2 = cv2.multiply(img.astype("float32"), matrix2).clip(0,255).astype("uint8")


#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_higher1), plt.title("Higher Contrast (1.1x)"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_higher2), plt.title("Higher Contrast (1.2x)"), plt.axis("off")
plt.show()


#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
python
b, g, r = cv2.split(img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g, cmap='gray'), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r, cmap='gray'), plt.title("Red Channel"), plt.axis("off")
plt.show()


#### 21. Merged the R, G, B , displays along with the original image
python
merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5,5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()


#### 22. Split the image into the H, S, V components & Display the channels.
python
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h, cmap='gray'), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s, cmap='gray'), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v, cmap='gray'), plt.title("Value Channel"), plt.axis("off")
plt.show()

#### 23. Merged the H, S, V, displays along with original image.
python
merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()


## Output:
 *i)* Read and Display an Image.

 *1.Read 'DIPT image-1.jpg' as grayscale and display:
  
   <img width="325" height="517" alt="Screenshot 2025-08-20 012619" src="https://github.com/user-attachments/assets/14d29856-415a-446a-b35f-0fe9a76047c6" />

2.Save image as PNG and display:

<img width="340" height="507" alt="Screenshot 2025-08-20 012625" src="https://github.com/user-attachments/assets/c8bd4660-7bec-411b-bac2-1a6fd3cb93a8" />

3.Cropped image:

<img width="415" height="507" alt="Screenshot 2025-08-20 012634" src="https://github.com/user-attachments/assets/df014fa9-cadb-4166-b565-78f74f7d5a58" />

4.Resize and flip Horizontally:

<img width="507" height="509" alt="Screenshot 2025-08-20 012641" src="https://github.com/user-attachments/assets/0c71bf6e-9aab-4e83-9c92-f843334e3f93" />

5.Read 'DIPT image-2.jpg' and Display the final annotated image:

<img width="678" height="540" alt="Screenshot 2025-08-20 012705" src="https://github.com/user-attachments/assets/bb452d38-cb9a-4375-9da1-399da15f019a" />

 *ii)* Adjust Image Brightness.
 
 1.Create brighter and darker images and display:
 
 <img width="1038" height="434" alt="Screenshot 2025-08-20 012715" src="https://github.com/user-attachments/assets/9c773b4b-b6cc-4ace-a085-3761fc19bd76" />
  
 *iii)* Modify Image Contrast.
  
  1.Modify contrast using scaling factors 1.1 and 1.2

 <img width="1024" height="423" alt="Screenshot 2025-08-20 012724" src="https://github.com/user-attachments/assets/76f727b3-1a8f-45d5-8556-9d9d8ff65bf7" />

 *iv)* Generate Third Image Using Bitwise Operations.
  
1.Split 'Boy.jpg' into B, G, R components and display:
  
  <img width="1127" height="431" alt="Screenshot 2025-08-20 012753" src="https://github.com/user-attachments/assets/fc3a2f9d-df28-4805-ba73-9423ff647a96" />
  
2.Merge the R, G, B channels and display:

  <img width="492" height="537" alt="Screenshot 2025-08-20 012801" src="https://github.com/user-attachments/assets/a91019bc-8792-42d7-9e2a-c4eace975e97" />
  
3.Split the image into H, S, V components and display:

  <img width="1181" height="430" alt="Screenshot 2025-08-20 012809" src="https://github.com/user-attachments/assets/fe155998-611e-4bf0-8a15-b7ef6a8f08fc" />
  
4.Merge the H, S, V channels and display:

  <img width="827" height="532" alt="Screenshot 2025-08-20 012818" src="https://github.com/user-attachments/assets/52f5c724-54b4-4600-8557-9aa8ebe93a3a" />

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.
