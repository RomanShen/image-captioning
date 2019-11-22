import os, random, shutil

TEST_PERCENTAGE = 20
names = []


def  copyFile(image_dir, image_tar_dir_test, image_tar_dir_train,):
    image_lists = os.listdir(image_dir)
    lengths_lists = len(image_lists)
    test_names = random.sample(image_lists, int(lengths_lists * TEST_PERCENTAGE / 100))
    for name in image_lists:
        if name in test_names:
            shutil.copy(image_dir + name, image_tar_dir_test + name)
        else:
            shutil.copy(image_dir + name, image_tar_dir_train + name)
        print("now {} is being proceeding.".format(name))


if __name__ == '__main__':
    dataset_path = '/root/dataset/NWPU-RESISC45/'
    classes = os.listdir(dataset_path)
    for name in classes:
        image_dir = dataset_path + name + '/'
        image_tar_dir_test = './dataset/val/' + name + '/'
        image_tar_dir_train = './dataset/train/' + name + '/'
        if not os.path.isdir(image_tar_dir_test):
            os.makedirs(image_tar_dir_test)
        if not os.path.isdir(image_tar_dir_train):
            os.makedirs(image_tar_dir_train)
        copyFile(image_dir, image_tar_dir_test, image_tar_dir_train)