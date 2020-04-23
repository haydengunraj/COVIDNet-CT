import os
import tensorflow as tf


class COVIDXCTDataset:
    def __init__(self, data_dir, image_height=512, image_width=512, shuffle_buffer=1000):
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.shuffle_buffer = shuffle_buffer

    def train_dataset(self, train_split_file='train.txt', batch_size=1):
        return self._make_dataset(train_split_file, batch_size, True)

    def validation_dataset(self, val_split_file='val.txt', batch_size=1):
        return self._make_dataset(val_split_file, batch_size, False)

    def _make_dataset(self, split_file, batch_size, is_training):
        """Creates COVIDX-CT dataset for train or val split"""
        files, classes = self._get_files(split_file)
        count = len(files)
        dataset = tf.data.Dataset.from_tensor_slices((files, classes))

        # Shuffle and repeat in training
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
            dataset = dataset.repeat()

        # Create and apply map function
        load_and_process = self._get_load_and_process_fn(is_training)
        dataset = dataset.map(load_and_process)

        # Batch data
        dataset = dataset.batch(batch_size)

        return dataset, count, batch_size

    def _get_load_and_process_fn(self, is_training):
        """Creates map function for TF dataset"""
        def load_and_process(path, label):
            # Load and stack to 3-channel
            image = tf.image.decode_png(tf.io.read_file(path), channels=1)
            image = tf.image.grayscale_to_rgb(image)

            # Scale to [0, 1] and resize
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = tf.image.resize(image, [self.image_height, self.image_width])

            # Flip L-R in training
            if is_training:
                image = tf.image.random_flip_left_right(image)

            label = tf.cast(label, dtype=tf.int32)

            return {'image': image, 'label': label}

        return load_and_process

    def _get_files(self, split_file):
        """Gets image filenames and classes"""
        files, classes = [], []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                fname, cls = line.strip('\n').split()
                files.append(os.path.join(self.data_dir, fname))
                classes.append(int(cls))
        return files, classes
