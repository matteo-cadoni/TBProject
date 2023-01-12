import numpy as np
#deselect blurry tiles

def is_blurry(image):
    #convert image to grayscale

    #find fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    #find mean of magnitude spectrum
    mean = np.mean(magnitude_spectrum)
    #print(mean)
    #if mean is less than 100 then image is blurry
    if mean < 220:
        return True
    else:
        return False
#plot spectrum of fft
def plot_spectrum(image):
    #convert image to grayscale

    #find fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    #find mean of magnitude spectrum
    mean = np.mean(magnitude_spectrum)
    #print(mean)
    #if mean is less than 100 then image is blurry
    if mean < 200:
        print('blurry')
    else:
        print('not blurry')
    #plot spectrum
    plt.subplot(121),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    #save image

def get_not_blurry_tiles(images_list):
    not_blurry_tiles=[]
    for tile in images_list:
        if is_blurry(tile):
            pass
        else:
            not_blurry_tiles.append(tile)
    return not_blurry_tiles

