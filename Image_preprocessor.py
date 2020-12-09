import multiprocessing, os, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

''' SETTING '''
INPUT_DIR = r''# your image data directory/
OUTPUT_DIR = r''  # output images data directory/
IMAGE_FORMAT = 'jpeg'  # output image format (jpeg, png)
FILE_NAME = 'pre'  # output image file name pre***.jpeg
IMAGE_SIZE = (48, 48)  # output image size
P_NUM = multiprocessing.cpu_count() # number of core
END_POINT = 10  # Number of images to be generated per image
# batch_size = Number of images in 1 label
# The batch size is calculated automatically.

# Set image modify
# See keras api documentation
# https://keras.io/preprocessing/image/
train_data = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True)

''' Execution '''
def check_start():
    global labels
    labels= os.listdir(INPUT_DIR)
    global labels_cnt
    labels_cnt = len(labels)

    no_image = []
    try:
        if not (os.path.isdir(INPUT_DIR)):
            print("Error : Not a directory.")
            print(INPUT_DIR)
            return False
        else:
            if not (labels):
                print('\nError : Input directory is empty.')
                print(INPUT_DIR)
                return False
            else:
                for name in labels:
                    if (len(os.walk(INPUT_DIR + name).__next__()[2]) == 0):
                        no_image.append(name)
        if (no_image):
            print('\nError : There are no images in the sub directory.')
            for name in no_image:
                print('- ' + name)
                return False
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    except :
        print('\nError : It is not the root images directory.')
        print("Check the 'INPUT_DIR' and images directory structure.")
        print(INPUT_DIR)
        return False
        exit()
    else :
        return True

def ready_dir():
    # looking for labels
    try :
        print('\nLooking for labels ...')
        for folder_name in labels:
            print(folder_name)
            os.mkdir(OUTPUT_DIR + folder_name)
        print('\n%d Labels' % labels_cnt)
        print('... Completed ...\n')
    except :
        print('Error : Failed to find and create label.')
        exit()

def gen(folder_name):
    i = 0
    batch_cnt = len(os.walk(INPUT_DIR + folder_name).__next__()[2])
    print('- ' + folder_name + ' start ...')
    try:
        for name in train_data.flow_from_directory(
                directory=INPUT_DIR,
                target_size=IMAGE_SIZE,
                batch_size=batch_cnt,
                save_to_dir=OUTPUT_DIR + folder_name,
                save_format=IMAGE_FORMAT,
                save_prefix=folder_name,
                classes=[folder_name]):
            i += 1
            if i > END_POINT:
                print('-- '+folder_name + ' end ...')
                break
    except Exception as e:
        print('\nError : Image generate error !')
        print(e)
        exit()

def gen_run():
    if P_NUM > labels_cnt:
        core = labels_cnt
    else:
        core = P_NUM

    try:
        p = multiprocessing.Pool(core)
        p.map_async(gen, labels).get()
    except Exception as e:
        print('\nError : Process execution error !')
        print(e)
        exit()

def check_end():
    new_labels = os.listdir(OUTPUT_DIR)
    if (labels == new_labels):
        print('\nAll images generated !')
    else:
        error_dir = list(set(labels) - set(new_labels))
        print('\nError : Some images were not generated. Please check the Input/Output directory.')
        for dir in error_dir:
            print(dir)

if __name__ == '__main__':
    start_time = time.time()

    if(check_start()) :
        ready_dir()
        gen_run()
        check_end()
        running_time = time.time() - start_time
        print('RUNNING TIME : %.2f sec' % running_time)
    else:
        exit()