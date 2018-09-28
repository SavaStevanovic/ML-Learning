import scipy.misc
image = scipy.misc.imread('./Images/example-image.png', mode='RGB')

print('Image shape', image.shape)
print('Chanel count', image.shape[2])
print('Image data type', image.dtype)
print(image[100:102, 100:102, :])
